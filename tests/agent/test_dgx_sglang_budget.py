"""Tests for the dgx-sglang last-resort hard token-budget clamp.

See agent/dgx_sglang_budget.py for the mechanism this protects against:
2026-08-23's live "The input (144706 tokens) is longer than the model's
context length (131072 tokens)" HTTP 400 on the DGX SGLang canary
(provider "dgx-sglang", model RadixArk/Qwen3.8-27B-NVFP4, context_length
131072). Pre-API compression (agent/conversation_compression.py) triggers
at a much lower threshold (~30,000 tokens) but does not run on every
request-assembly path, so this clamp is the unconditional backstop that
runs immediately before every dgx-sglang send.

An adversarial review of the first version of this fix (2026-08-23) added
the coverage below for three gaps: (1) two send paths besides
build_api_kwargs also reach dgx-sglang — the iteration-limit summary call
and the auxiliary/fallback router's shared kwargs-builder — both are
covered here via enforce_dgx_sglang_token_budget /
enforce_dgx_sglang_token_budget_raw; (2) dropping messages by raw index can
orphan a tool_call/tool_result pair, covered by the atomicity tests below;
(3) the flat 2048-token margin cannot absorb the rough token counter's
~20-25% undercount on BPE-dense payloads, covered by the percentage-margin
tests below.
"""

from __future__ import annotations

import logging

import pytest

from agent.dgx_sglang_budget import (
    DGX_SGLANG_CONTEXT_MARGIN_PERCENT,
    DGX_SGLANG_CONTEXT_MARGIN_TOKENS,
    DGX_SGLANG_DEFAULT_CONTEXT_LENGTH,
    DGX_SGLANG_PROVIDER_NAME,
    DgxSglangTokenBudgetError,
    clamp_messages_to_token_budget,
    enforce_dgx_sglang_token_budget,
    enforce_dgx_sglang_token_budget_raw,
    is_dgx_sglang_route,
)


def _counter():
    """Deterministic token counter for the pure arithmetic tests: 1 token
    per content character. Real send-site tests use the module's real
    estimator (agent.model_metadata.estimate_messages_tokens_rough) via
    enforce_dgx_sglang_token_budget[_raw] instead.
    """

    def _count(messages):
        return sum(len(str(m.get("content", ""))) for m in messages)

    return _count


def _identity_sanitize(messages: list) -> list:
    return messages


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _tool_call_msg(content: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }


def _tool_result_msg(call_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


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


# ── clamp_messages_to_token_budget (pure arithmetic, sanitize disabled) ──


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
        sanitize_fn=_identity_sanitize,
    )
    assert trimmed is messages
    assert dropped_count == 0
    assert dropped_tokens == 0


def test_clamp_drops_oldest_turns_until_it_fits():
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
        sanitize_fn=_identity_sanitize,
    )
    assert dropped_count == 2
    assert dropped_tokens == 100
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "l" * 10
    assert [m["content"] for m in trimmed] == ["s" * 10, "m" * 50, "l" * 10]
    remaining_tokens = sum(len(m["content"]) for m in trimmed)
    assert remaining_tokens + 10 <= 100


def test_clamp_raises_when_system_plus_last_turn_alone_exceed_budget():
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
            sanitize_fn=_identity_sanitize,
        )


def test_clamp_raises_when_no_middle_turns_exist_at_all():
    messages = [_msg("user", "u" * 5000)]
    with pytest.raises(DgxSglangTokenBudgetError):
        clamp_messages_to_token_budget(
            messages,
            max_tokens=10,
            ceiling_tokens=1000,
            token_counter=_counter(),
            sanitize_fn=_identity_sanitize,
        )


def test_clamp_terminates_when_sanitize_keeps_re_inflating():
    # Pathological sanitize_fn that adds a message back every time one is
    # removed -- must still terminate (bounded by len(messages)) and raise
    # rather than loop forever or silently return an over-budget result.
    def _re_inflate(messages):
        return messages + [_msg("user", "x" * 100)]

    messages = [
        _msg("system", "s"),
        _msg("user", "u" * 200),
        _msg("user", "u" * 200),
        _msg("user", "last"),
    ]
    with pytest.raises(DgxSglangTokenBudgetError):
        clamp_messages_to_token_budget(
            messages,
            max_tokens=10,
            ceiling_tokens=50,
            token_counter=_counter(),
            sanitize_fn=_re_inflate,
        )


# ── Finding 2: tool_call / tool_result atomicity (real sanitize_fn) ──────


def test_clamp_does_not_orphan_a_tool_call_when_default_sanitize_runs():
    # [system, userA, assistant(tool_calls=[c1]), tool(c1), userB, last].
    # Dropping strictly by raw index would remove userA then the
    # assistant(tool_calls) message, leaving tool(c1) orphaned -- a live
    # SGLang 400 ("No tool call found for function call output"). The
    # default sanitize_fn (agent.agent_runtime_helpers.sanitize_api_messages)
    # must repair that on every iteration.
    messages = [
        _msg("system", "s" * 4),
        _msg("user", "u" * 1000),  # userA -- oldest trimmable
        _tool_call_msg("a" * 1000, "call_1"),  # assistant w/ tool_calls
        _tool_result_msg("call_1", "r" * 10),  # its paired tool result
        _msg("user", "u" * 10),  # userB
        _msg("user", "final ask"),  # last turn -- always protected
    ]
    # Real estimator via the char-per-token toy counter is irrelevant here;
    # what matters is that dropping enough to fit forces past the
    # assistant/tool_call pair. ceiling small enough to force >=2 drops.
    trimmed, dropped_count, dropped_tokens = clamp_messages_to_token_budget(
        messages,
        max_tokens=10,
        ceiling_tokens=50,
        token_counter=_counter(),
        # sanitize_fn left as the default (real sanitize_api_messages).
    )
    assert dropped_count >= 2

    # No orphaned tool result: every remaining tool message's tool_call_id
    # matches a tool_calls entry on some remaining assistant message (or the
    # tool message itself was dropped by sanitize as an orphan).
    surviving_call_ids = {
        tc.get("id")
        for m in trimmed
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    for m in trimmed:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in surviving_call_ids

    # No orphaned tool_call either: every assistant tool_calls id has a
    # matching tool result somewhere in the trimmed list (sanitize injects a
    # stub result rather than leaving it dangling).
    result_ids = {m.get("tool_call_id") for m in trimmed if m.get("role") == "tool"}
    for cid in surviving_call_ids:
        assert cid in result_ids

    # System prompt and the final turn are still exactly preserved.
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "final ask"


def test_clamp_drops_orphaned_tool_result_when_only_the_call_is_removed():
    # Directly exercises sanitize's "drop orphaned tool result" rule: force
    # a single drop that removes ONLY the assistant(tool_calls) message
    # (by making it, uniquely, the oldest trimmable message) and confirm the
    # now-orphaned tool result doesn't survive in the final output.
    messages = [
        _msg("system", "s"),
        _tool_call_msg("a" * 5000, "call_1"),
        _tool_result_msg("call_1", "r"),
        _msg("user", "final"),
    ]
    trimmed, dropped_count, _ = clamp_messages_to_token_budget(
        messages,
        max_tokens=10,
        ceiling_tokens=20,
        token_counter=_counter(),
    )
    assert dropped_count >= 1
    assert all(m.get("role") != "tool" or m.get("tool_call_id") != "call_1" for m in trimmed)
    assert all(
        m.get("role") != "assistant" or not any(
            tc.get("id") == "call_1" for tc in (m.get("tool_calls") or [])
        )
        for m in trimmed
    )


# ── Finding 3: percentage-of-prompt margin (not a flat fraction of context) ──


def test_margin_uses_percent_of_prompt_when_it_exceeds_the_flat_floor():
    from agent.dgx_sglang_budget import _effective_margin_tokens

    # A 50,000-token prompt: 18% of it (9000) exceeds the 2048 flat floor,
    # so the percentage term must win.
    prompt_tokens = 50_000
    expected = round(prompt_tokens * DGX_SGLANG_CONTEXT_MARGIN_PERCENT)
    assert expected > DGX_SGLANG_CONTEXT_MARGIN_TOKENS
    assert _effective_margin_tokens(prompt_tokens) == expected


def test_margin_falls_back_to_flat_floor_for_small_prompts():
    from agent.dgx_sglang_budget import _effective_margin_tokens

    # A tiny prompt: 18% of 100 is 18, far under the 2048 floor.
    assert _effective_margin_tokens(100) == DGX_SGLANG_CONTEXT_MARGIN_TOKENS


def test_large_prompt_that_flat_margin_would_have_missed_gets_clamped(caplog):
    # Reconstructs the shape of the reviewed gap: a prompt whose ROUGH
    # estimate sits just under (context_length - flat 2048) but whose real
    # tokenizer would have exceeded it once BPE-dense payload undercount is
    # accounted for. The percentage margin must trigger a clamp where a flat
    # 2048 margin would have let it through untouched.
    agent = _FakeAgent(
        provider=DGX_SGLANG_PROVIDER_NAME,
        base_url="http://192.168.0.214:30000/v1",
        context_length=DGX_SGLANG_DEFAULT_CONTEXT_LENGTH,
        max_tokens=1000,
    )
    # ~127500 rough-estimated tokens (under context_length - 2048 = 129024,
    # so a flat-margin design would treat this as fitting) but well over
    # context_length - (18% of ~127500 ~= 22950) = ~108122.
    big_user_content = "u" * (127_000 * 4)
    messages = [_msg("system", "s" * 4), _msg("user", big_user_content), _msg("user", "final")]
    with caplog.at_level(logging.WARNING, logger="agent.dgx_sglang_budget"):
        result = enforce_dgx_sglang_token_budget(agent, {"messages": messages, "max_tokens": 1000})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert result["messages"] != messages


# ── enforce_dgx_sglang_token_budget (agent-object send sites) ────────────


def test_enforce_logs_one_warning_naming_turns_and_tokens_dropped(caplog):
    # This exercises the real tokenizer (estimate_messages_tokens_rough),
    # not the toy counter used above -- sizes are deliberately generous
    # (~1000-token turns) rather than hand-computed to an exact count.
    agent = _FakeAgent(
        provider=DGX_SGLANG_PROVIDER_NAME,
        base_url="http://192.168.0.214:30000/v1",
        context_length=2500,
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
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][-1]["content"] == "l" * 4
    assert len(result["messages"]) < len(messages)


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
    assert result["messages"] == small_messages


# ── enforce_dgx_sglang_token_budget_raw (auxiliary_client._build_call_kwargs) ──


def test_enforce_raw_is_noop_for_other_providers():
    huge_messages = [_msg("system", "s")] + [_msg("user", "x" * 100000) for _ in range(5)]
    api_kwargs = {"messages": huge_messages, "max_tokens": 4096}
    result = enforce_dgx_sglang_token_budget_raw(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4-flash",
        api_kwargs=dict(api_kwargs),
    )
    assert result["messages"] is huge_messages


def test_enforce_raw_clamps_dgx_sglang_fallback_style_call(monkeypatch, caplog):
    # Simulates agent.auxiliary_client._build_call_kwargs's caller shape:
    # no agent object, just provider/base_url/model strings -- the path the
    # adversarial review flagged as unguarded (context-compression's own
    # summarization call, and same-provider retry / cross-provider
    # fallback, all funnel kwargs through _build_call_kwargs).
    import agent.dgx_sglang_budget as budget_mod

    monkeypatch.setattr(budget_mod, "_resolve_context_length_raw", lambda *a, **k: 2500)

    messages = [
        _msg("system", "s" * 4),
        _msg("user", "u" * 4000),
        _msg("assistant", "a" * 4000),
        _msg("user", "l" * 4),
    ]
    api_kwargs = {"messages": messages, "max_tokens": 10}
    with caplog.at_level(logging.WARNING, logger="agent.dgx_sglang_budget"):
        result = enforce_dgx_sglang_token_budget_raw(
            provider="dgx-sglang",
            base_url="http://192.168.0.214:30000/v1",
            model="RadixArk/Qwen3.8-27B-NVFP4",
            api_kwargs=api_kwargs,
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert len(result["messages"]) < len(messages)
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][-1]["content"] == "l" * 4


def test_resolve_context_length_raw_prefers_cache_then_config_then_default(monkeypatch):
    from agent import dgx_sglang_budget as budget_mod

    # 1. Cache hit wins outright.
    monkeypatch.setattr(
        "agent.model_metadata.get_cached_context_length", lambda model, base_url: 99999
    )
    assert (
        budget_mod._resolve_context_length_raw(
            "RadixArk/Qwen3.8-27B-NVFP4", "http://192.168.0.214:30000/v1", "dgx-sglang"
        )
        == 99999
    )

    # 2. Cache miss falls through to live config's declared context_length.
    monkeypatch.setattr(
        "agent.model_metadata.get_cached_context_length", lambda model, base_url: None
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "providers": {
                "dgx-sglang": {
                    "models": {"RadixArk/Qwen3.8-27B-NVFP4": {"context_length": 131072}}
                }
            }
        },
    )
    assert (
        budget_mod._resolve_context_length_raw(
            "RadixArk/Qwen3.8-27B-NVFP4", "http://192.168.0.214:30000/v1", "dgx-sglang"
        )
        == 131072
    )

    # 3. Both miss -> hardcoded default.
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert (
        budget_mod._resolve_context_length_raw("unknown-model", "http://x/v1", "dgx-sglang")
        == DGX_SGLANG_DEFAULT_CONTEXT_LENGTH
    )


def test_margin_and_default_context_length_constants_match_original_spec():
    # The original task spec's flat computation (131072 - 2048 = 129024)
    # still holds as the FLOOR case (small prompt, percent term doesn't
    # win) -- see test_margin_falls_back_to_flat_floor_for_small_prompts
    # above for the dynamic behavior on larger prompts.
    assert DGX_SGLANG_DEFAULT_CONTEXT_LENGTH == 131072
    assert DGX_SGLANG_CONTEXT_MARGIN_TOKENS == 2048
    assert DGX_SGLANG_DEFAULT_CONTEXT_LENGTH - DGX_SGLANG_CONTEXT_MARGIN_TOKENS == 129024
    assert 0.15 <= DGX_SGLANG_CONTEXT_MARGIN_PERCENT <= 0.20
