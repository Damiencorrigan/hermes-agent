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

from types import SimpleNamespace

import pytest

import agent.chat_completion_helpers as h

DEEPSEEK = SimpleNamespace(provider="deepseek", model="deepseek-v4-flash")
OLLAMA = SimpleNamespace(provider="ollama", model="qwen2.5:14b")


@pytest.fixture(autouse=True)
def _guard_root(tmp_path, monkeypatch):
    """Point the gate at a temp ai-fleet root with a fake spend_guard."""
    guard_dir = tmp_path / "ai-fleet"
    guard_dir.mkdir()
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
