"""Cron silent-failover fix (S74): pinned providers must never be substituted.

Background: a cron job that EXPLICITLY pins ``provider`` is a deliberate
routing choice. When that provider fails to resolve (AuthError), run_job used
to silently walk the operator's configured fallback chain and substitute
another provider — violating fleet rule 1/8 (fail loud, never silently fall
back). The job then recorded a misleading last_error from the substituted
provider (e.g. a parked/unfunded gemini 429) instead of the true reason
("provider dgx-ollama unresolvable").

This regression suite locks in:
  (a) cron execution resolves config custom providers exactly like interactive
      sessions — the pinned custom provider name is passed to
      resolve_runtime_provider and the resolved runtime reaches AIAgent;
  (b) a PINNED provider that fails to resolve RAISES with the true reason and
      NEVER walks the fallback chain — resolve_runtime_provider is called
      exactly once (with the pinned provider), and no substitute provider is
      constructed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job
from hermes_cli.auth import AuthError


def _pinned_job(**overrides):
    job = {
        "id": "pin-failover-test",
        "name": "pin failover test",
        "prompt": "hello",
        "model": "gpt-oss:120b",
        "provider": "dgx-ollama",
        "provider_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, resolver_side_effect, tmp_path, config_yaml="model:\n  default: deepseek-v4-flash\n"):
    """Drive run_job with a stubbed resolver.

    Returns (success, output, final_response, error, agent_kwargs, requested_calls).
    """
    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    fake_db = MagicMock()
    requested_calls = []

    def _resolver(*args, **kwargs):
        requested_calls.append(kwargs.get("requested"))
        return resolver_side_effect(*args, **kwargs)

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolver), \
         patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, output, final_response, error = run_job(job)
        agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None

    return success, output, final_response, error, agent_kwargs, requested_calls


class TestPinnedProviderNoSilentSubstitution:
    def test_pinned_unresolvable_provider_raises_true_reason(self, tmp_path):
        """(b) Pinned provider that fails to resolve → job fails with the TRUE
        reason naming the pinned provider; the fallback chain is never walked."""

        def _raise_auth(**kwargs):
            raise AuthError("Unknown provider 'dgx-ollama'")

        success, output, _final, error, agent_kwargs, requested = _run(
            _pinned_job(), _raise_auth, tmp_path,
        )

        assert success is False
        assert error is not None
        # True reason must name the pinned provider, not a substituted one.
        assert "dgx-ollama" in error
        assert "Unknown provider" in error
        # No agent constructed with a substituted provider.
        assert agent_kwargs is None
        # resolve_runtime_provider was called exactly once — with the PINNED
        # provider, never with fallback-chain entries (deepseek etc.).
        assert requested == ["dgx-ollama"]

    def test_pinned_provider_auth_failure_never_substitutes(self, tmp_path):
        """(b) Even when a fallback chain IS configured, a pinned job's auth
        failure must NOT ride it: resolver called once with the pin only."""
        config_yaml = (
            "model:\n"
            "  default: deepseek-v4-flash\n"
            "fallback_providers:\n"
            "  - provider: deepseek\n"
            "    model: deepseek-v4-flash\n"
            "  - provider: gemini\n"
            "    model: gemini-3.6-flash\n"
        )

        def _raise_auth(**kwargs):
            raise AuthError("No usable credentials found for provider 'dgx-ollama'")

        success, output, _final, error, agent_kwargs, requested = _run(
            _pinned_job(), _raise_auth, tmp_path, config_yaml=config_yaml,
        )

        assert success is False
        assert error is not None
        assert "dgx-ollama" in error
        # The configured fallback chain must NOT be consulted for a pinned job.
        assert requested == ["dgx-ollama"]
        assert "deepseek" not in requested
        assert agent_kwargs is None

    def test_pinned_custom_provider_resolves_like_interactive(self, tmp_path):
        """(a) A pinned custom provider (dgx-ollama) resolves exactly like an
        interactive session: pin name forwarded, resolved runtime reaches
        AIAgent with the custom provider's base_url."""
        runtime = {
            "api_key": "ollama",
            "base_url": "http://192.168.0.214:11434/v1",
            "provider": "custom",
            "api_mode": "chat_completions",
            "requested_provider": "dgx-ollama",
            "source": "custom_provider:dgx-ollama",
        }

        def _resolve(**kwargs):
            assert kwargs["requested"] == "dgx-ollama"
            assert kwargs["target_model"] == "gpt-oss:120b"
            return runtime

        success, _output, _final, error, agent_kwargs, requested = _run(
            _pinned_job(), _resolve, tmp_path,
        )

        assert success is True
        assert error is None
        # Every resolution request must be the pinned provider (the preflight
        # probe resolves once more with the same pin) — never a substituted one.
        assert requested, "resolver should have been called"
        assert set(requested) == {"dgx-ollama"}
        assert agent_kwargs is not None
        assert agent_kwargs["provider"] == "custom"
        assert agent_kwargs["base_url"] == "http://192.168.0.214:11434/v1"
        assert agent_kwargs["requested_provider"] == "dgx-ollama"
        assert agent_kwargs["model"] == "gpt-oss:120b"
