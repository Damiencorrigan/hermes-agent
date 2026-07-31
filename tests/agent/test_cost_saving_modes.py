"""Fleet cost-saving modes ENFORCED inside hermes (Damien's 2026-07-31 ruling).

The mode system already existed as a *rule* (``~/ai-fleet/cost_mode.json`` +
``docs/cost-saving-modes.md`` + SOUL §7) — but a rule only binds where an agent
reads it, which is exactly not where unattended spend happens. These tests pin
the enforcing half:

  * mode "off" / a missing state file / a corrupt one → completely invisible
    (FAIL-SAFE — deliberately the OPPOSITE of the spend caps' fail-closed
    choice, because a cost mode is an optional preference, not a safety floor);
  * mode 3 → every metered provider refused, at the chat-completion entry
    points AND at every fallback rung, so no failover can quietly become the
    lane that spends; local ollama still runs;
  * mode 1 → no gating at all (it is a routing preference);
  * the refusal is 402-classified so the retry loop stops instead of hammering.
"""

import json
from types import SimpleNamespace

import pytest

import agent.cost_mode as cm
import agent.chat_completion_helpers as h

# Captured before the autouse fixture re-points it, for the live test at the end.
_REAL_COST_MODE_FILE = cm.COST_MODE_FILE


@pytest.fixture(autouse=True)
def _mode_file(tmp_path, monkeypatch):
    """Point the gate at a temp state file and clear the 60s verdict cache."""
    path = tmp_path / "cost_mode.json"
    monkeypatch.setattr(cm, "COST_MODE_FILE", path)
    cm._reset_cache_for_tests()
    yield path
    cm._reset_cache_for_tests()


def _set_mode(path, mode):
    path.write_text(json.dumps({"mode": mode, "set_by": "damien"}), encoding="utf-8")
    cm._reset_cache_for_tests()


GEMINI = SimpleNamespace(provider="gemini", model="gemini-3.6-flash")
OLLAMA = SimpleNamespace(provider="ollama", model="qwen2.5:14b")


# ── reading the state file ───────────────────────────────────────────────────

def test_missing_state_file_reads_as_off(_mode_file):
    assert not _mode_file.exists()
    assert cm.read_cost_mode() == "off"


@pytest.mark.parametrize("mode", ["off", "1", "2", "3"])
def test_every_valid_mode_round_trips(_mode_file, mode):
    _set_mode(_mode_file, mode)
    assert cm.read_cost_mode() == mode


def test_mode_written_as_a_number_is_accepted(_mode_file):
    """A non-Python writer may emit 3 rather than "3"."""
    _mode_file.write_text(json.dumps({"mode": 3}), encoding="utf-8")
    cm._reset_cache_for_tests()
    assert cm.read_cost_mode() == "3"


def test_corrupt_json_fails_SAFE_not_closed(_mode_file):
    """The whole point of the fail-open choice: a broken OPTIONAL feature must
    not brick every hermes session and cron job."""
    _mode_file.write_text("{not json", encoding="utf-8")
    cm._reset_cache_for_tests()
    assert cm.read_cost_mode() == "off"
    assert cm.enforce_cost_saving_mode(GEMINI) is None


def test_unrecognised_mode_reads_as_off_not_as_a_different_mode(_mode_file):
    _set_mode(_mode_file, "lockdown")
    assert cm.read_cost_mode() == "off"


def test_verdict_is_cached_so_a_busy_loop_does_not_stat_per_call(_mode_file, monkeypatch):
    _set_mode(_mode_file, "3")
    assert cm.read_cost_mode() == "3"
    # Deleting the file cannot change the answer inside the TTL.
    _mode_file.unlink()
    assert cm.read_cost_mode() == "3"


def test_cache_expires(_mode_file, monkeypatch):
    _set_mode(_mode_file, "3")
    assert cm.read_cost_mode() == "3"
    _mode_file.unlink()
    monkeypatch.setattr(cm.time, "monotonic",
                        lambda: cm._cache["expires"] + 1.0)
    assert cm.read_cost_mode() == "off"


# ── mode off / 1 / 2 → no gating of live calls ───────────────────────────────

@pytest.mark.parametrize("mode", ["off", "1", "2"])
def test_modes_below_lockdown_never_block_a_live_call(_mode_file, mode):
    """Modes 1 and 2 are ROUTING preferences the agent applies with judgement
    (SOUL §7); only mode 3 is a hard boundary. Gating live calls in mode 2
    would break the "paid allowed for critical work" half of its definition."""
    _set_mode(_mode_file, mode)
    assert cm.enforce_cost_saving_mode(GEMINI) is None


def test_no_state_file_at_all_never_blocks(_mode_file):
    assert cm.enforce_cost_saving_mode(GEMINI) is None


# ── mode 3 → metered refused, local allowed ──────────────────────────────────

@pytest.mark.parametrize("provider", ["gemini", "deepseek", "kimi", "openai",
                                      "perplexity", "glm", "nous", "openrouter",
                                      "anthropic", "ollama-cloud", "groq"])
def test_mode_3_refuses_every_metered_provider(_mode_file, provider):
    _set_mode(_mode_file, "3")
    with pytest.raises(cm.CostSavingModeBlocked):
        cm.enforce_cost_saving_mode(SimpleNamespace(provider=provider))


@pytest.mark.parametrize("provider", ["ollama", "OLLAMA", "ollama-local",
                                      "lmstudio", "vllm", "llamacpp", "local"])
def test_mode_3_allows_local_zero_cost_providers(_mode_file, provider):
    _set_mode(_mode_file, "3")
    assert cm.enforce_cost_saving_mode(SimpleNamespace(provider=provider)) is None


def test_mode_3_refusal_is_self_explaining(_mode_file):
    """The message has to stand on its own in a log a human reads days later:
    which mode, which file, and the exact sentence that lifts it."""
    _set_mode(_mode_file, "3")
    with pytest.raises(cm.CostSavingModeBlocked) as exc:
        cm.enforce_cost_saving_mode(GEMINI)
    msg = str(exc.value)
    assert "FLEET COST-SAVING MODE 3" in msg
    assert "lockdown" in msg
    assert "gemini" in msg
    assert str(cm.COST_MODE_FILE) in msg
    assert str(cm.COST_MODE_DOC) in msg
    assert "cost saving mode off" in msg


def test_mode_3_classifies_as_billing_so_the_loop_stops_retrying(_mode_file):
    """402 → FailoverReason.billing (retryable=False). Identical to the
    PrimarySpendCapExceeded contract, so no error_classifier edit is needed."""
    from agent.error_classifier import FailoverReason, classify_api_error

    _set_mode(_mode_file, "3")
    with pytest.raises(cm.CostSavingModeBlocked) as exc:
        cm.enforce_cost_saving_mode(GEMINI)
    assert exc.value.status_code == 402
    classified = classify_api_error(exc.value, provider="gemini",
                                    model="gemini-3.6-flash")
    assert classified.reason == FailoverReason.billing
    assert classified.retryable is False


def test_agent_without_a_provider_attribute_is_not_gated(_mode_file):
    _set_mode(_mode_file, "3")
    assert cm.enforce_cost_saving_mode(SimpleNamespace()) is None


# ── the chat-completion entry points actually call it ────────────────────────

def test_non_streaming_entry_point_refuses_before_building_a_client(_mode_file, monkeypatch):
    _set_mode(_mode_file, "3")
    agent = SimpleNamespace(
        provider="gemini", model="gemini-3.6-flash", api_mode="chat_completions",
        platform="cron", _interrupt_requested=False,
        _create_request_openai_client=lambda **kw: pytest.fail("must not build a client"),
    )
    with pytest.raises(cm.CostSavingModeBlocked):
        h.interruptible_api_call(agent, {"model": "gemini-3.6-flash", "messages": []})


def test_streaming_entry_point_refuses_before_dispatch(_mode_file, monkeypatch):
    _set_mode(_mode_file, "3")
    agent = SimpleNamespace(
        provider="gemini", model="gemini-3.6-flash", api_mode="chat_completions",
        platform="cron", _interrupt_requested=False,
        _interruptible_api_call=lambda kw: pytest.fail("must not reach dispatch"),
    )
    with pytest.raises(cm.CostSavingModeBlocked):
        h.interruptible_streaming_api_call(
            agent, {"model": "gemini-3.6-flash", "messages": []})


# ── fallback rungs refuse too (no silent failover into a paid lane) ──────────

def test_fallback_skip_reason_blocks_metered_entries_in_mode_3(_mode_file):
    _set_mode(_mode_file, "3")
    assert cm.fallback_skip_reason("kimi") == "cost_saving_mode_3_lockdown:kimi"


def test_fallback_skip_reason_allows_local_entries_in_mode_3(_mode_file):
    _set_mode(_mode_file, "3")
    assert cm.fallback_skip_reason("ollama") is None


@pytest.mark.parametrize("mode", ["off", "1", "2"])
def test_fallback_skip_reason_is_inert_below_lockdown(_mode_file, mode):
    _set_mode(_mode_file, mode)
    assert cm.fallback_skip_reason("kimi") is None


def test_try_activate_fallback_exhausts_the_whole_chain_in_mode_3(_mode_file, monkeypatch):
    """The end-to-end guarantee: in lockdown no rung answers, so the chain
    exhausts and the refusal surfaces — rather than the session landing on
    whichever paid lane happened to be next."""
    _set_mode(_mode_file, "3")
    monkeypatch.setattr(h, "_fallback_entry_unavailable_without_network",
                        lambda agent, fb: None)

    chain = [
        {"provider": "deepseek", "model": "deepseek-v4-pro"},
        {"provider": "gemini", "model": "gemini-3.6-flash"},
        {"provider": "kimi", "model": "kimi-k3"},
    ]
    agent = SimpleNamespace(
        provider="gemini", model="gemini-3.6-flash", base_url="",
        _fallback_chain=chain, _fallback_index=0, _fallback_activated=False,
        _primary_runtime={"provider": "gemini"}, _rate_limited_until=0,
        _unavailable_fallback_keys=set(),
    )
    agent._try_activate_fallback = lambda reason=None: h.try_activate_fallback(agent, reason)

    assert h.try_activate_fallback(agent) is False
    assert agent._fallback_index == len(chain)
    assert agent.provider == "gemini"  # never switched onto a paid rung


def test_lockdown_means_no_PAID_fallback_not_no_fallback(_mode_file):
    _set_mode(_mode_file, "3")
    assert cm.fallback_skip_reason("kimi") == "cost_saving_mode_3_lockdown:kimi"
    assert cm.fallback_skip_reason("ollama") is None


# ── cron gating semantics ────────────────────────────────────────────────────

def test_cron_mode_3_skips_every_metered_job(_mode_file):
    _set_mode(_mode_file, "3")
    reason = cm.cron_skip_reason("gemini", critical=False)
    assert reason and "mode 3" in reason
    # even a critical job — mode 3 has no exceptions
    assert cm.cron_skip_reason("gemini", critical=True)


def test_cron_mode_3_runs_local_jobs(_mode_file):
    _set_mode(_mode_file, "3")
    assert cm.cron_skip_reason("ollama") is None


def test_cron_mode_2_skips_metered_jobs_that_are_not_critical(_mode_file):
    _set_mode(_mode_file, "2")
    reason = cm.cron_skip_reason("gemini", critical=False)
    assert reason and "mode 2" in reason and "not flagged critical" in reason


def test_cron_mode_2_runs_a_critical_metered_job(_mode_file):
    _set_mode(_mode_file, "2")
    assert cm.cron_skip_reason("gemini", critical=True) is None


def test_cron_mode_2_runs_local_jobs_regardless_of_the_flag(_mode_file):
    _set_mode(_mode_file, "2")
    assert cm.cron_skip_reason("ollama", critical=False) is None


@pytest.mark.parametrize("mode", ["off", "1"])
def test_cron_is_not_gated_below_mode_2(_mode_file, mode):
    _set_mode(_mode_file, mode)
    assert cm.cron_skip_reason("gemini", critical=False) is None


def test_cron_gate_never_raises_on_a_broken_read(_mode_file, monkeypatch):
    monkeypatch.setattr(cm, "read_cost_mode",
                        lambda: (_ for _ in ()).throw(OSError("boom")))
    assert cm.cron_skip_reason("gemini") is None


# ── session banner ───────────────────────────────────────────────────────────

def test_no_banner_when_mode_is_off(_mode_file):
    assert cm.session_banner() is None


@pytest.mark.parametrize("mode", ["1", "2", "3"])
def test_banner_names_the_mode_and_the_doc(_mode_file, mode):
    _set_mode(_mode_file, mode)
    banner = cm.session_banner()
    assert banner == (
        f"⚠ FLEET COST-SAVING MODE {mode} ACTIVE — follow {cm.COST_MODE_DOC}")


def test_system_prompt_injects_the_banner_only_when_a_mode_is_active(_mode_file, monkeypatch):
    import agent.system_prompt as sp

    _set_mode(_mode_file, "2")
    # Build the volatile tier through the real function with a minimal agent.
    agent = SimpleNamespace(
        load_soul_identity=False, skip_context_files=True, valid_tool_names=[],
        context_compressor=None, _memory_store=None, _memory_manager=None,
        _memory_enabled=False, _user_profile_enabled=False,
        pass_session_id=False, session_id=None, model="m", provider="ollama",
        platform="cron", _kanban_worker_guidance=None,
        _task_completion_guidance=False, _parallel_tool_call_guidance=False,
    )
    parts = sp.build_system_prompt_parts(agent)
    assert "⚠ FLEET COST-SAVING MODE 2 ACTIVE" in parts["volatile"]

    _set_mode(_mode_file, "off")
    parts_off = sp.build_system_prompt_parts(agent)
    assert "COST-SAVING MODE" not in parts_off["volatile"]


# ── live integration (no mocking of the real state file) ────────────────────

def test_live_state_file_shape(monkeypatch):
    """Reads the REAL ~/ai-fleet/cost_mode.json. Asserts only the shape —
    whether a mode is active depends on what Damien last said."""
    monkeypatch.setattr(cm, "COST_MODE_FILE", _REAL_COST_MODE_FILE)
    cm._reset_cache_for_tests()
    assert cm.read_cost_mode() in cm.VALID_MODES
