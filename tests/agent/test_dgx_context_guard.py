"""Tests for agent/dgx_context_guard.py (2026-08-26, HERMES-BOUND mission).

See agent/dgx_context_guard.py's module docstring for the mechanism: it
imports ai-fleet's ops/dgx_context_guard.py BY PATH and shares its exact
lock-file pools, so a fleet_router.py caller and a Hermes profile gateway
draw from ONE pool, not two independently-capped ones. These tests fabricate
a fake ai-fleet checkout (a minimal ops/dgx_context_guard.py stand-in) so
they never touch the real ~/ai-fleet checkout or its live lock state.
"""

from __future__ import annotations

import sys
import textwrap
from types import SimpleNamespace

import pytest


# ── is_dgx_target / host:port matching ──────────────────────────────────────


class TestIsDgxTarget:
    def test_matches_default_dgx_host_port(self, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        import agent.dgx_context_guard as m
        assert m.is_dgx_target("http://192.168.0.214:30000/v1") is True

    def test_does_not_match_other_host(self, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        import agent.dgx_context_guard as m
        assert m.is_dgx_target("https://openrouter.ai/api/v1") is False

    def test_does_not_match_dgx_host_different_port(self, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        import agent.dgx_context_guard as m
        assert m.is_dgx_target("http://192.168.0.214:11434/v1") is False

    def test_empty_base_url_is_not_a_target(self, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        import agent.dgx_context_guard as m
        assert m.is_dgx_target("") is False
        assert m.is_dgx_target(None) is False

    def test_env_override_extends_allowlist(self, monkeypatch):
        monkeypatch.setenv("HERMES_DGX_GUARD_HOSTS", "10.0.0.5:9000,192.168.0.214:30000")
        import agent.dgx_context_guard as m
        assert m.is_dgx_target("http://10.0.0.5:9000/v1") is True
        assert m.is_dgx_target("http://192.168.0.214:30000/v1") is True
        assert m.is_dgx_target("http://192.168.0.214:11434/v1") is False


# ── estimate_prompt_tokens ────────────────────────────────────────────────


class TestEstimatePromptTokens:
    def test_scales_with_message_length(self):
        import agent.dgx_context_guard as m
        small = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        big = {"messages": [{"role": "user", "content": "x" * 350_000}], "tools": []}
        assert m.estimate_prompt_tokens(big) > m.estimate_prompt_tokens(small)

    def test_uses_chars_per_token_3_5_ratio(self):
        import agent.dgx_context_guard as m
        payload = {"messages": [{"role": "user", "content": "a" * 3500}], "tools": []}
        tokens = m.estimate_prompt_tokens(payload)
        # serialized JSON adds structural overhead on top of the 3500 chars,
        # so the estimate must be >= 1000 (3500/3.5), not exactly equal.
        assert tokens >= 1000

    def test_tools_count_toward_estimate(self):
        import agent.dgx_context_guard as m
        no_tools = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        with_tools = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x", "description": "y" * 5000}}],
        }
        assert m.estimate_prompt_tokens(with_tools) > m.estimate_prompt_tokens(no_tools)

    def test_never_raises_on_malformed_payload(self):
        import agent.dgx_context_guard as m
        # A circular reference can't be json.dumps'd — must fall back, not raise.
        bad = {}
        bad["self"] = bad
        assert m.estimate_prompt_tokens({"messages": bad, "tools": []}) >= 0


# ── module import failure (fail loud, no silent no-op) ─────────────────────


class TestGuardUnavailable:
    def test_missing_ai_fleet_root_raises(self, monkeypatch, tmp_path):
        import agent.dgx_context_guard as m
        monkeypatch.setattr(m, "_AI_FLEET_ROOT_CANDIDATES", (str(tmp_path / "nope"),))
        monkeypatch.setattr(m, "_guard_module", None)
        with pytest.raises(m.DgxContextGuardUnavailable):
            m._load_guard_module()

    def test_dgx_request_guard_noop_for_non_dgx_target_even_without_ai_fleet(
        self, monkeypatch, tmp_path
    ):
        """The whole point of the allowlist gate: an unrelated provider must
        never even attempt the ai-fleet import, so a broken/missing
        checkout can't break every other provider's requests."""
        import agent.dgx_context_guard as m
        monkeypatch.setattr(m, "_AI_FLEET_ROOT_CANDIDATES", (str(tmp_path / "nope"),))
        monkeypatch.setattr(m, "_guard_module", None)
        agent = SimpleNamespace(base_url="https://openrouter.ai/api/v1")
        with m.dgx_request_guard(agent, {"messages": []}):
            pass  # must not raise


# ── dgx_request_guard against a fake shared guard module ────────────────────


@pytest.fixture
def fake_ai_fleet(tmp_path, monkeypatch):
    """Build a minimal ops/dgx_context_guard.py under tmp_path and point
    agent.dgx_context_guard at it, so these tests exercise the real
    import-by-path + dgx_slot() call contract without touching the
    developer's actual ~/ai-fleet checkout or its live lock state."""
    fleet_root = tmp_path / "fake-ai-fleet"
    ops_dir = fleet_root / "ops"
    ops_dir.mkdir(parents=True)
    (fleet_root / "__init__.py").write_text("")
    (ops_dir / "__init__.py").write_text("")
    (ops_dir / "dgx_context_guard.py").write_text(textwrap.dedent(
        """
        import contextlib

        calls = []

        class DGXContextGuardTimeoutError(RuntimeError):
            pass

        class _Cfg:
            def __init__(self):
                self.max_concurrent_workers = 6
                self.max_concurrent_big_prompt = 2
                self.big_prompt_token_threshold = 80_000

        def load_config():
            return _Cfg()

        @contextlib.contextmanager
        def dgx_slot(prompt_tokens, config=None, timeout_s=None, caller_label="fleet_router"):
            calls.append({
                "prompt_tokens": prompt_tokens,
                "timeout_s": timeout_s,
                "caller_label": caller_label,
            })
            yield
        """
    ))

    import agent.dgx_context_guard as m
    monkeypatch.setattr(m, "_AI_FLEET_ROOT_CANDIDATES", (str(fleet_root),))
    monkeypatch.setattr(m, "_guard_module", None)
    # sys.path / sys.modules leak across tests otherwise (ops.* gets cached).
    monkeypatch.syspath_prepend(str(fleet_root))
    for name in list(sys.modules):
        if name == "ops" or name.startswith("ops."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    yield m
    for name in list(sys.modules):
        if name == "ops" or name.startswith("ops."):
            sys.modules.pop(name, None)


class TestDgxRequestGuard:
    def test_noop_for_non_dgx_provider(self, fake_ai_fleet):
        m = fake_ai_fleet
        agent = SimpleNamespace(base_url="https://openrouter.ai/api/v1")
        with m.dgx_request_guard(agent, {"messages": [{"role": "user", "content": "hi"}]}):
            pass
        guard = m._load_guard_module()
        assert guard.calls == []

    def test_acquires_shared_pool_for_dgx_target(self, fake_ai_fleet, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        monkeypatch.setenv("HERMES_PROFILE", "fleet-overseer")
        m = fake_ai_fleet
        agent = SimpleNamespace(base_url="http://192.168.0.214:30000/v1")
        with m.dgx_request_guard(agent, {"messages": [{"role": "user", "content": "hi"}], "tools": []}):
            pass
        guard = m._load_guard_module()
        assert len(guard.calls) == 1
        call = guard.calls[0]
        assert call["caller_label"] == "hermes:fleet-overseer"
        assert call["timeout_s"] == m.WORKER_WAIT_TIMEOUT_S

    def test_big_prompt_uses_long_timeout(self, fake_ai_fleet, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        monkeypatch.setenv("HERMES_PROFILE", "ta-desk")
        m = fake_ai_fleet
        agent = SimpleNamespace(base_url="http://192.168.0.214:30000/v1")
        big_content = "x" * 400_000  # well over 80,000 tokens at chars/3.5
        with m.dgx_request_guard(agent, {"messages": [{"role": "user", "content": big_content}], "tools": []}):
            pass
        guard = m._load_guard_module()
        assert len(guard.calls) == 1
        call = guard.calls[0]
        assert call["timeout_s"] == m.BIG_PROMPT_WAIT_TIMEOUT_S
        assert call["prompt_tokens"] > 80_000

    def test_profile_name_defaults_when_env_unset(self, fake_ai_fleet, monkeypatch):
        monkeypatch.delenv("HERMES_DGX_GUARD_HOSTS", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
        m = fake_ai_fleet
        agent = SimpleNamespace(base_url="http://192.168.0.214:30000/v1")
        with m.dgx_request_guard(agent, {"messages": [], "tools": []}):
            pass
        guard = m._load_guard_module()
        assert guard.calls[0]["caller_label"] == "hermes:default"

    def test_timeout_error_propagates_unchanged(self, fake_ai_fleet, monkeypatch, tmp_path):
        """A DGXContextGuardTimeoutError from the shared guard must reach the
        caller unmodified -- Hermes' own fallback ladder (OpenRouter, etc.)
        decides what happens next, this module never intercepts it."""
        fleet_root = tmp_path / "fake-ai-fleet-timeout"
        ops_dir = fleet_root / "ops"
        ops_dir.mkdir(parents=True)
        (fleet_root / "__init__.py").write_text("")
        (ops_dir / "__init__.py").write_text("")
        (ops_dir / "dgx_context_guard.py").write_text(textwrap.dedent(
            """
            import contextlib

            class DGXContextGuardTimeoutError(RuntimeError):
                pass

            class _Cfg:
                def __init__(self):
                    self.max_concurrent_workers = 6
                    self.max_concurrent_big_prompt = 2
                    self.big_prompt_token_threshold = 80_000

            def load_config():
                return _Cfg()

            @contextlib.contextmanager
            def dgx_slot(prompt_tokens, config=None, timeout_s=None, caller_label="fleet_router"):
                raise DGXContextGuardTimeoutError("timed out waiting for a slot")
                yield  # pragma: no cover - unreachable, keeps this a generator
            """
        ))
        m = fake_ai_fleet
        monkeypatch.setattr(m, "_AI_FLEET_ROOT_CANDIDATES", (str(fleet_root),))
        monkeypatch.setattr(m, "_guard_module", None)
        for name in list(sys.modules):
            if name == "ops" or name.startswith("ops."):
                sys.modules.pop(name, None)
        monkeypatch.syspath_prepend(str(fleet_root))

        guard = m._load_guard_module()
        agent = SimpleNamespace(base_url="http://192.168.0.214:30000/v1")
        with pytest.raises(guard.DGXContextGuardTimeoutError):
            with m.dgx_request_guard(agent, {"messages": [], "tools": []}):
                pass  # pragma: no cover - must not be reached


# ── wired into the real chokepoint methods ──────────────────────────────────


class TestChokepointWiring:
    """run_agent.py's _interruptible_api_call / _interruptible_streaming_api_call
    are the two call sites conversation_loop.py uses for every live turn
    (interactive, cron, delegated) -- confirm both import and use the guard."""

    def test_interruptible_api_call_source_uses_dgx_request_guard(self):
        import inspect
        import run_agent
        src = inspect.getsource(run_agent.AIAgent._interruptible_api_call)
        assert "dgx_request_guard" in src

    def test_interruptible_streaming_api_call_source_uses_dgx_request_guard(self):
        import inspect
        import run_agent
        src = inspect.getsource(run_agent.AIAgent._interruptible_streaming_api_call)
        assert "dgx_request_guard" in src
