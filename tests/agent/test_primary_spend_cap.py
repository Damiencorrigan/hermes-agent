"""PRIMARY-lane daily spend cap enforced inside hermes (2026-09-01 builder).

The fallback fuse gates FALLBACK activation only. The PRIMARY lane — the
provider a session actually runs on — was never capped inside hermes:
hermes-direct DeepSeek bypassed the hard caps entirely (the burn landed on
dsh-harness while the hermes-deepseek ledger lane read $0). This test pins
the enforcing half:

  * a capped-out paid lane refuses at the chat-completion entry points
    BEFORE a client is built (fail-closed, 402-classified);
  * an unreadable guard ALSO refuses (fail-closed — an unreadable cap is
    not permission to spend);
  * providers with no enforced ledger lane are untouched;
  * the refusal carries status_code=402 so error_classifier routes it to
    FailoverReason.billing (retry stops, fuse-gated fallback takes over).
"""

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import agent.chat_completion_helpers as h

DEEPSEEK = SimpleNamespace(provider="deepseek", model="deepseek-v4-flash")
OLLAMA = SimpleNamespace(provider="ollama", model="qwen2.5:14b")


@pytest.fixture(autouse=True)
def _guard_root(tmp_path, monkeypatch):
    """Point the gate at a temp ai-fleet root with a fake spend_guard, and
    UNDO the hermetic escape hatch (conftest sets HERMES_SPEND_GUARD_OFF=1
    for sandboxed tests — these tests exist to prove the gate ACTIVE)."""
    monkeypatch.delenv("HERMES_SPEND_GUARD_OFF", raising=False)
    guard_dir = tmp_path / "ai-fleet"
    guard_dir.mkdir()
    # the hard-cap park file must live under the SAME temp root (no real
    # HOME park may leak into these tests)
    monkeypatch.setattr(h, "_FAILOVER_STATE_PATH",
                        guard_dir / "ops" / "state" / "lane_failover.json")
    (guard_dir / "spend_guard.py").write_text(
        "import sys, json\n"
        "def lane_blocked(repo_root, lane, cfg=None, **kw):\n"
        "    # TEST SHIM: 'BLOCKED' lane always blocked; 'UNREADABLE' raises\n"
        "    if lane == 'hermes-deepseek':\n"
        "        return {'blocked': True, 'lane': lane,\n"
        "                'reason': 'over_daily_spend_cap:test-shim',\n"
        "                'spent_usd': 6.0, 'cap_usd': 5.0}\n"
        "    if lane == 'boom':\n"
        "        raise RuntimeError('test shim crash')\n"
        "    return {'blocked': False, 'lane': lane, 'reason': 'under cap'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(h, "_SPEND_GUARD_FLEET_ROOT", guard_dir)
    # L10 day-total hard cap (2026-09-02): point the live-ledger read at a
    # temp UNDER-cap ledger so the day-total gate stays quiet unless a test
    # says otherwise (the REAL ~/ai-fleet ledger is over $20 today and must
    # never leak into unit tests).
    day_ledger = guard_dir / "bench" / "daily_spend.jsonl"
    day_ledger.parent.mkdir(parents=True, exist_ok=True)
    day_ledger.write_text(
        json.dumps({"day": date.today().isoformat(), "lane": "dsh-harness",
                    "cost_usd": 1.0}) + "\n", encoding="utf-8")
    monkeypatch.setattr(h, "_HARD_CAP_DAY_LEDGER", day_ledger)
    yield guard_dir


# ── the gate itself ─────────────────────────────────────────────────────────

def test_capped_lane_refuses_before_any_client():
    """deepseek primary over cap -> PrimarySpendCapExceeded, 402."""
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    assert "hermes-deepseek" in str(ei.value)
    assert "over_daily_spend_cap" in str(ei.value)
    assert ei.value.status_code == 402, "402 so the retry loop stops + billing fallback"


def test_refusal_is_402_routed_to_billing_fallback():
    """error_classifier maps 402 -> FailoverReason.billing (retry=False)."""
    from agent.error_classifier import classify_api_error, FailoverReason
    exc = h.PrimarySpendCapExceeded("REFUSED by the hermes primary spend cap")
    classified = classify_api_error(exc, provider="deepseek",
                                    model="deepseek-v4-flash")
    assert classified.reason == FailoverReason.billing
    assert classified.status_code == 402


def test_unreadable_guard_fails_CLOSED(tmp_path, monkeypatch):
    """A spend_guard that crashes still refuses — never spend blind."""
    guard_dir = tmp_path / "boom-guard"
    guard_dir.mkdir()
    (guard_dir / "spend_guard.py").write_text(
        "import sys, json\n"
        "def lane_blocked(repo_root, lane, cfg=None, **kw):\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(h, "_SPEND_GUARD_FLEET_ROOT", guard_dir)
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    assert "spend_guard" in str(ei.value).lower()


def test_uncapped_provider_is_untouched():
    """ollama (no enforced lane) is never refused."""
    assert h.enforce_primary_spend_cap(OLLAMA) is None


def test_missing_guard_fails_CLOSED(tmp_path, monkeypatch):
    """No spend_guard at all -> refuse (fail-closed)."""
    monkeypatch.setattr(h, "_SPEND_GUARD_FLEET_ROOT", tmp_path / "nothing")
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    assert "spend_guard_missing" in str(ei.value)


def test_streaming_and_nonstreaming_entries_both_gate(tmp_path, monkeypatch):
    """Both chat-completion entry points call the gate."""
    # cost mode must be off for this test (the real cost_mode.json is mode 3
    # which preempts with CostSavingModeBlocked before the cap is reached —
    # correct production behavior, but it would mask the cap assertion here)
    import agent.cost_mode as cm
    monkeypatch.setattr(cm, "COST_MODE_FILE", tmp_path / "cost_mode_off.json")
    (tmp_path / "cost_mode_off.json").write_text(
        json.dumps({"mode": "off", "set_by": "test"}), encoding="utf-8")
    cm._reset_cache_for_tests()
    calls = []
    real = h.enforce_primary_spend_cap
    monkeypatch.setattr(h, "enforce_primary_spend_cap",
                        lambda a: calls.append(a.provider) or real(a))
    # non-streaming entry (interruptible_api_call) refuses for deepseek
    with pytest.raises(h.PrimarySpendCapExceeded):
        h.interruptible_api_call(DEEPSEEK, {})
    assert "deepseek" in calls
    # streaming entry refuses too
    with pytest.raises(h.PrimarySpendCapExceeded):
        h.interruptible_streaming_api_call(DEEPSEEK, {})
    assert calls.count("deepseek") >= 2


# ── L10 day-total hard cap (2026-09-02 spend-cap-failclosed directive) ───────
# The $20 hard cap is ACCOUNT-wide (fleet_router + dsh-harness + hermes rows
# on one DeepSeek account). The hourly park file and the per-lane caps were
# both blind to it at request time on 2026-09-02 (ledger crossed $20.82 at
# 16:54, park landed 17:54) — these tests pin the LIVE-LEDGER refusal.

def _write_day_ledger(monkeypatch, tmp_path, rows):
    p = tmp_path / "dayledger"
    p.mkdir(exist_ok=True)
    led = p / "daily_spend.jsonl"
    with led.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(h, "_HARD_CAP_DAY_LEDGER", led)
    return led


def test_day_total_2001_refuses_before_spend_guard(tmp_path, monkeypatch):
    """A $20.01 DeepSeek-account day PHYSICALLY refuses the call at request
    time from the live ledger — no waiting for the hourly park file."""
    _write_day_ledger(monkeypatch, tmp_path, [
        {"day": date.today().isoformat(), "lane": "dsh-harness", "cost_usd": 20.01}])
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    msg = str(ei.value)
    assert "hard_cap_day_total" in msg and "$20.01" in msg
    assert ei.value.status_code == 402, "402 so the retry loop stops + billing fallback"


def test_day_total_at_cap_refuses(tmp_path, monkeypatch):
    """Exactly $20.00 is AT the cap -> refused (>=, not >)."""
    _write_day_ledger(monkeypatch, tmp_path, [
        {"day": date.today().isoformat(), "lane": "deepseek", "cost_usd": 20.0}])
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    assert "hard_cap_day_total" in str(ei.value)


def test_day_total_under_cap_falls_through_to_lane_caps(tmp_path, monkeypatch):
    """$19.99 day -> day-total gate quiet; the per-lane spend_guard check
    still runs and its shim verdict is what refuses."""
    _write_day_ledger(monkeypatch, tmp_path, [
        {"day": date.today().isoformat(), "lane": "dsh-harness", "cost_usd": 19.99}])
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    msg = str(ei.value)
    assert "over_daily_spend_cap" in msg
    assert "hard_cap_day_total" not in msg


def test_day_total_counts_only_today(tmp_path, monkeypatch):
    """Yesterday's spend never counts against today's hard cap."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    _write_day_ledger(monkeypatch, tmp_path, [
        {"day": yesterday, "lane": "dsh-harness", "cost_usd": 50.0},
        {"day": date.today().isoformat(), "lane": "dsh-harness", "cost_usd": 1.0}])
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    msg = str(ei.value)
    assert "over_daily_spend_cap" in msg  # lane-shim verdict, not the day total
    assert "hard_cap_day_total" not in msg


def test_day_total_unreadable_ledger_fails_CLOSED(tmp_path, monkeypatch):
    """A missing/unreadable ledger refuses — cannot prove the cap open."""
    monkeypatch.setattr(h, "_HARD_CAP_DAY_LEDGER",
                        tmp_path / "no" / "daily_spend.jsonl")
    with pytest.raises(h.PrimarySpendCapExceeded) as ei:
        h.enforce_primary_spend_cap(DEEPSEEK)
    assert "hard_cap_day_ledger_unreadable" in str(ei.value)
