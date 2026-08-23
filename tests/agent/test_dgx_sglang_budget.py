"""Tests for the dgx-sglang last-resort hard token-budget clamp.

See agent/dgx_sglang_budget.py for the mechanism this protects against:
2026-08-23's live "The input (144706 tokens) is longer than the model's
context length (131072 tokens)" HTTP 400 on the DGX SGLang canary
(provider "dgx-sglang", model RadixArk/Qwen3.8-27B-NVFP4, context_length
131072). Pre-API compression (agent/conversation_compression.py) triggers
at a much lower threshold (~30,000 tokens) but is not guaranteed to run on
every request-assembly path, so this clamp is the unconditional backstop
that runs immediately before every dgx-sglang send.
"""

from __future__ import annotations

import logging

import pytest

from agent.dgx_sglang_budget import (
    DGX_SGLANG_CONTEXT_MARGIN_TOKENS,
    DGX_SGLANG_DEFAULT_CONTEXT_LENGTH,
    DGX_SGLANG_PROVIDER_NAME,
    DgxSglangTokenBudgetError,
    clamp_messages_to_token_budget,
    enforce_dgx_sglang_token_budget,
    is_dgx_sglang_route,
)


def _counter(text_tokens_per_char: int = 1):
    """A trivial deterministic token counter: 1 token per message, unless
    overridden — tests that need per-message weight pass their own callable.
    """

    def _count(messages):
        return sum(len(str(m.get("content", ""))) for m in messages)

    return _count


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class _FakeCompressor:
    def __init__(self, context_length):
        self.context_length = context_length


class _FakeAgent:
    def __init__(self, provider, base_url, context_length=None, max_tokens=None):
        self.provider = provider
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.context_compressor = (
            _FakeCompressor(context_length) if context_length is not None else None
        )


# ── is_dgx_sglang_route ──────────────────────────────────────────────────


def test_is_dgx_sglang_route_matches_provider_name():
    assert is_dgx_sglang_route("dgx-sglang", "http://192.168.0.214:30000/v1")
    assert is_dgx_sglang_route("DGX-SGLANG", "http://anything/v1")


def test_is_dgx_sglang_route_matches_host_port_fallback():
    assert is_dgx_sglang_route("custom", "http://192.168.0.214:30000/v1")


def test_is_dgx_sglang_route_rejects_other_providers():
    assert not is_dgx_sglang_route("openrouter", "https://openrouter.ai/api/v1")
    assert not is_dgx_sglang_route("deepseek", "https://api.deepseek.com/v1")
    # Same host, different port must NOT match (e.g. the dgx-ollama route).
    assert not is_dgx_sglang_route("dgx-ollama", "http://192.168.0.214:11434/v1")


# ── clamp_messages_to_token_budget (pure logic) ──────────────────────────


def test_clamp_under_budget_passes_untouched():
    messages = [
        _msg("system", "sys"),
        _msg("user", "a" * 10),
        _msg("assistant", "b" * 10),
        _msg("user", "c" * 10),
    ]
    trimmed, dropped_count, dropped_tokens = clamp_messages_to_token_budget(
        messages,
        max_tokens=100,
        ceiling_tokens=1000,
        token_counter=_counter(),
    )
    assert trimmed is messages
    assert dropped_count == 0
    assert dropped_tokens == 0


def test_clamp_drops_oldest_turns_until_it_fits_and_logs(caplog):
    # system(10) + old_user(50) + old_asst(50) + mid_user(50) + last_user(10)
    # = 170 content tokens. Ceiling 100, max_tokens 10 -> budget for prompt
    # is 90. Oldest-first drop: old_user(50) -> 120 remaining still > 90;
    # drop old_asst(50) -> 70 remaining <= 90 -> fits after dropping 2 turns.
    messages = [
        _msg("system", "s" * 10),
        _msg("user", "u" * 50),
        _msg("assistant", "a" * 50),
        _msg("user", "m" * 50),
        _msg("user", "l" * 10),
    ]
    trimmed, dropped_count, dropped_tokens = clamp_messages_to_token_budget(
        messages,
        max_tokens=10,
        ceiling_tokens=100,
        token_counter=_counter(),
    )
    assert dropped_count == 2
    assert dropped_tokens == 100
    # System prompt preserved, latest turn preserved, middle turns dropped
    # oldest-first.
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "l" * 10
    assert [m["content"] for m in trimmed] == ["s" * 10, "m" * 50, "l" * 10]
    remaining_tokens = sum(len(m["content"]) for m in trimmed)
    assert remaining_tokens + 10 <= 100


def test_enforce_logs_one_warning_naming_turns_and_tokens_dropped(caplog):
    # This exercises the real tokenizer (estimate_messages_tokens_rough),
    # not the test's toy counter used for the pure clamp_messages_to_
    # token_budget tests above — so sizes here are deliberately generous
    # (~1000-token turns against a ~50-token ceiling) rather than hand-
    # computed to an exact token count.
    # ceiling = context_length - DGX_SGLANG_CONTEXT_MARGIN_TOKENS(2048) = 50.
    agent = _FakeAgent(
        provider=DGX_SGLANG_PROVIDER_NAME,
        base_url="http://192.168.0.214:30000/v1",
        context_length=DGX_SGLANG_CONTEXT_MARGIN_TOKENS + 50,
        max_tokens=10,
    )
    messages = [
        _msg("system", "s" * 4),
        _msg("user", "u" * 4000),
        _msg("assistant", "a" * 4000),
        _msg("user", "l" * 4),
    ]
    api_kwargs = {"messages": messages, "max_tokens": 10}
    with caplog.at_level(logging.WARNING, logger="agent.dgx_sglang_budget"):
        result = enforce_dgx_sglang_token_budget(agent, api_kwargs)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "dropped" in warnings[0].message
    assert "2 oldest turn" in warnings[0].message
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][-1]["content"] == "l" * 4
    assert len(result["messages"]) < len(messages)


def test_clamp_raises_when_system_plus_last_turn_alone_exceed_budget():
    # No trimmable middle: system + last user alone already exceed ceiling.
    messages = [
        _msg("system", "s" * 5000),
        _msg("user", "u" * 5000),
    ]
    with pytest.raises(DgxSglangTokenBudgetError):
        clamp_messages_to_token_budget(
            messages,
            max_tokens=10,
            ceiling_tokens=1000,
            token_counter=_counter(),
        )


def test_clamp_raises_when_no_middle_turns_exist_at_all():
    # A single message (no system, no room to trim) that alone busts budget.
    messages = [_msg("user", "u" * 5000)]
    with pytest.raises(DgxSglangTokenBudgetError):
        clamp_messages_to_token_budget(
            messages,
            max_tokens=10,
            ceiling_tokens=1000,
            token_counter=_counter(),
        )


# ── enforce_dgx_sglang_token_budget (agent-facing wrapper) ───────────────


def test_enforce_is_noop_for_other_providers():
    agent = _FakeAgent(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        context_length=131072,
        max_tokens=4096,
    )
    huge_messages = [_msg("system", "s")] + [
        _msg("user", "x" * 100000) for _ in range(5)
    ]
    api_kwargs = {"messages": huge_messages, "max_tokens": 4096}
    result = enforce_dgx_sglang_token_budget(agent, dict(api_kwargs))
    assert result["messages"] is huge_messages


def test_enforce_falls_back_to_default_context_length_when_compressor_missing():
    agent = _FakeAgent(
        provider=DGX_SGLANG_PROVIDER_NAME,
        base_url="http://192.168.0.214:30000/v1",
        context_length=None,  # no compressor attached
        max_tokens=100,
    )
    small_messages = [_msg("system", "s"), _msg("user", "hi")]
    api_kwargs = {"messages": small_messages, "max_tokens": 100}
    result = enforce_dgx_sglang_token_budget(agent, api_kwargs)
    # Well under DGX_SGLANG_DEFAULT_CONTEXT_LENGTH - margin -> untouched.
    assert result["messages"] == small_messages


def test_margin_and_default_context_length_constants_match_spec():
    # Task spec: ceiling = context_length(131072) - margin(2048) = 129024.
    assert DGX_SGLANG_DEFAULT_CONTEXT_LENGTH == 131072
    assert DGX_SGLANG_CONTEXT_MARGIN_TOKENS == 2048
    assert DGX_SGLANG_DEFAULT_CONTEXT_LENGTH - DGX_SGLANG_CONTEXT_MARGIN_TOKENS == 129024
