"""Cron gating by the fleet cost-saving modes (Damien's 2026-07-31 ruling).

A cron job is the one place a cost-saving mode cannot bind by persuasion —
there is no agent in the loop to read ``~/ai-fleet/docs/cost-saving-modes.md``
and choose the cheap lane. So run_job gates itself:

  * mode 3 — every metered job is SKIPPED;
  * mode 2 — metered jobs are SKIPPED unless the job carries
    ``"critical": true``;
  * modes 1 / off — untouched.

Two properties are load-bearing and asserted below:
  1. a gated tick constructs NO agent (so it makes no inference call and costs
     nothing), exactly like the #44585 drift guard; and
  2. a gated tick is a SILENT SUCCESS, not an error — a mode is an operator's
     deliberate pause, and erroring would spam the failure-alerting path on
     every tick for the whole duration of a lockdown.

These tests drive the real run_job path (mocked AIAgent +
resolve_runtime_provider against a temp HERMES_HOME), mirroring
tests/cron/test_cron_provider_pin.py.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agent.cost_mode as cm
from cron.scheduler import SILENT_MARKER, run_job


@pytest.fixture(autouse=True)
def _mode_file(tmp_path, monkeypatch):
    path = tmp_path / "cost_mode.json"
    monkeypatch.setattr(cm, "COST_MODE_FILE", path)
    cm._reset_cache_for_tests()
    yield path
    cm._reset_cache_for_tests()


def _set_mode(path, mode):
    path.write_text(json.dumps({"mode": mode}), encoding="utf-8")
    cm._reset_cache_for_tests()


def _job(**overrides):
    job = {
        "id": "cost-mode-test",
        "name": "cost mode test",
        "prompt": "hello",
        "model": "gemini-3.6-flash",
        "provider": "gemini",
        "provider_snapshot": None,
        "model_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, provider, tmp_path):
    """Drive run_job with the resolved runtime provider pinned to ``provider``.

    Returns (success, output, final_response, error, agent_constructed).
    """
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": provider,
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, output, final_response, error = run_job(job)
        return success, output, final_response, error, mock_agent_cls.called


# ── mode 3 ───────────────────────────────────────────────────────────────────

def test_mode_3_skips_a_metered_job_without_constructing_an_agent(_mode_file, tmp_path):
    _set_mode(_mode_file, "3")
    success, output, final, error, constructed = _run(_job(), "gemini", tmp_path)
    assert constructed is False          # no agent → no inference call → $0
    assert success is True               # a pause, not a failure
    assert error is None
    assert final == SILENT_MARKER        # nothing delivered
    assert "skipped by cost saving mode" in output
    assert "mode 3" in output


def test_mode_3_skips_even_a_critical_job(_mode_file, tmp_path):
    """Lockdown has no exceptions — that is what distinguishes it from mode 2."""
    _set_mode(_mode_file, "3")
    _, _, _, _, constructed = _run(_job(critical=True), "gemini", tmp_path)
    assert constructed is False


def test_mode_3_still_runs_a_local_ollama_job(_mode_file, tmp_path):
    _set_mode(_mode_file, "3")
    _, _, _, _, constructed = _run(
        _job(provider="ollama", model="qwen2.5:14b"), "ollama", tmp_path)
    assert constructed is True


# ── mode 2 ───────────────────────────────────────────────────────────────────

def test_mode_2_skips_a_metered_job_that_is_not_critical(_mode_file, tmp_path):
    _set_mode(_mode_file, "2")
    success, output, final, error, constructed = _run(_job(), "gemini", tmp_path)
    assert constructed is False
    assert success is True and error is None and final == SILENT_MARKER
    assert "mode 2" in output


def test_mode_2_runs_a_job_flagged_critical(_mode_file, tmp_path):
    _set_mode(_mode_file, "2")
    _, _, _, _, constructed = _run(_job(critical=True), "gemini", tmp_path)
    assert constructed is True


def test_mode_2_treats_a_falsy_critical_flag_as_absent(_mode_file, tmp_path):
    _set_mode(_mode_file, "2")
    _, _, _, _, constructed = _run(_job(critical=False), "gemini", tmp_path)
    assert constructed is False


def test_mode_2_runs_a_local_job_without_the_flag(_mode_file, tmp_path):
    _set_mode(_mode_file, "2")
    _, _, _, _, constructed = _run(
        _job(provider="ollama", model="qwen2.5:14b"), "ollama", tmp_path)
    assert constructed is True


# ── modes 1 / off / no file → untouched ──────────────────────────────────────

@pytest.mark.parametrize("mode", ["off", "1"])
def test_modes_below_2_do_not_gate_cron(_mode_file, tmp_path, mode):
    _set_mode(_mode_file, mode)
    _, _, _, _, constructed = _run(_job(), "gemini", tmp_path)
    assert constructed is True


def test_missing_state_file_does_not_gate_cron(_mode_file, tmp_path):
    """FAIL-SAFE: an OPTIONAL feature whose state file is absent must not stop
    the fleet. (Deliberately the opposite of the spend caps' fail-closed.)"""
    assert not _mode_file.exists()
    _, _, _, _, constructed = _run(_job(), "gemini", tmp_path)
    assert constructed is True


def test_corrupt_state_file_does_not_gate_cron(_mode_file, tmp_path):
    _mode_file.write_text("{{{ not json", encoding="utf-8")
    cm._reset_cache_for_tests()
    _, _, _, _, constructed = _run(_job(), "gemini", tmp_path)
    assert constructed is True


# ── the anti-swap invariant ──────────────────────────────────────────────────

def test_a_gated_job_is_never_rerouted_onto_another_provider(_mode_file, tmp_path):
    """SKIP, never SWAP. Silently re-routing a scheduled job onto a cheaper
    provider is the exact drift the #44585 guard exists to prevent."""
    _set_mode(_mode_file, "3")
    job = _job()
    _run(job, "gemini", tmp_path)
    assert job["provider"] == "gemini"
    assert job["model"] == "gemini-3.6-flash"
