"""Last-resort hard token-budget clamp for the DGX SGLang canary route.

Context
-------
The DGX SGLang canary serves ``RadixArk/Qwen3.8-27B-NVFP4`` at
``http://192.168.0.214:30000/v1`` under provider name ``dgx-sglang`` with a
declared context window of 131072 tokens (see
``~/.hermes/profiles/*/config.yaml`` -> ``providers.dgx-sglang.models.
"RadixArk/Qwen3.8-27B-NVFP4".context_length``).

Hermes already runs a "Pre-API compression" pass (see
``agent/conversation_compression.py`` / ``agent/context_compressor.py``)
that proactively summarizes history once a request's estimated token count
crosses a configured threshold (commonly ~30,000 tokens — far below this
model's real 131072-token window). That pass is a *soft* budget guard tuned
for cost/quality, not a hard ceiling tied to any one model's actual context
size, and it is not guaranteed to run on every retry / fallback / error-
recovery path that can (re)assemble a request. On 2026-08-23 that gap let a
144,706-token prompt reach the provider and get rejected with HTTP 400:
"The input (144706 tokens) is longer than the model's context length
(131072 tokens)."

This module is the hard backstop: it runs unconditionally, immediately
before every outgoing chat-completions request is handed to the DGX SGLang
canary (wired into ``agent.chat_completion_helpers.build_api_kwargs``,
*after* compression has already had its chance), and truncates the oldest
conversation turns — never the system prompt, never the latest turn — until
``prompt_tokens + max_tokens`` fits under ``context_length - margin``. It
touches no other provider: the gate is keyed on the ``dgx-sglang`` provider
name (with a base_url fallback for the same host:port), so OpenRouter,
DeepSeek, and every other route are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Named constants (task requirement: no bare hardcodes at the call site) ──

# Provider name as declared in ~/.hermes/profiles/*/config.yaml (model.provider
# / providers.<name>).  This is the primary gate.
DGX_SGLANG_PROVIDER_NAME = "dgx-sglang"

# base_url host:port fallback gate, for routes that reach the same canary
# under a different/unregistered provider label. Matches
# "http://192.168.0.214:30000/v1".
DGX_SGLANG_HOST = "192.168.0.214"
DGX_SGLANG_PORT = "30000"

# Fallback context window when the provider's declared context_length isn't
# available at runtime (e.g. context_compressor not yet initialized for this
# model). Mirrors providers.dgx-sglang.models."RadixArk/Qwen3.8-27B-NVFP4"
# .context_length in the live profile configs.
DGX_SGLANG_DEFAULT_CONTEXT_LENGTH = 131072

# Safety margin subtracted from context_length before comparing against
# prompt_tokens + max_tokens. Leaves room for tokenizer-estimate slop
# between hermes's rough counter and the provider's real tokenizer.
DGX_SGLANG_CONTEXT_MARGIN_TOKENS = 2048

# Fallback output-token budget when neither the outgoing request nor the
# agent declares one (mirrors the ~4096-token defaults used elsewhere in the
# send path, e.g. conversation_loop.py's output-cap retry boost).
DGX_SGLANG_DEFAULT_MAX_TOKENS = 4096


class DgxSglangTokenBudgetError(RuntimeError):
    """Raised when a request cannot be made to fit the dgx-sglang budget.

    This means the system prompt plus the single latest turn already exceed
    ``context_length - margin`` on their own — there is nothing left that
    is safe to trim. Sending anyway would reproduce the exact 400 this
    clamp exists to prevent, so the caller must not silently proceed.
    """


def is_dgx_sglang_route(provider: Optional[str], base_url: Optional[str]) -> bool:
    """Return True when this request targets the DGX SGLang canary.

    Keys on the provider name first (the canonical, config-declared
    identity); falls back to a host:port match on the base URL so an
    unregistered/aliased route to the same canary is still caught. Any
    other provider/base_url returns False, leaving it untouched.
    """
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == DGX_SGLANG_PROVIDER_NAME:
        return True

    if not base_url:
        return False
    try:
        parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    except ValueError:
        return False
    host = (parsed.hostname or "").strip().lower()
    port = str(parsed.port) if parsed.port is not None else ""
    return host == DGX_SGLANG_HOST and port == DGX_SGLANG_PORT


def clamp_messages_to_token_budget(
    messages: list,
    *,
    max_tokens: int,
    ceiling_tokens: int,
    token_counter: Callable[[list], int],
) -> tuple[list, int, int]:
    """Drop the oldest trimmable turns until the request fits the ceiling.

    Preserves every leading ``system``-role message and the final message
    in ``messages`` (the latest turn the model must respond to) — those are
    never candidates for removal. Turns are dropped oldest-first from the
    remaining middle section, one message at a time, until
    ``prompt_tokens + max_tokens <= ceiling_tokens``.

    Returns ``(trimmed_messages, dropped_count, dropped_tokens)``. When the
    input already fits, returns the input unchanged with
    ``dropped_count == 0``.

    Raises ``DgxSglangTokenBudgetError`` when even the protected head
    (system messages) plus the protected tail (the latest turn) alone
    cannot fit under the ceiling — there is nothing left this function is
    permitted to trim.
    """
    if not messages:
        return messages, 0, 0

    per_message_tokens = [token_counter([msg]) for msg in messages]
    total_tokens = sum(per_message_tokens)

    if total_tokens + max_tokens <= ceiling_tokens:
        return messages, 0, 0

    protected_head = 0
    while (
        protected_head < len(messages)
        and isinstance(messages[protected_head], dict)
        and messages[protected_head].get("role") == "system"
    ):
        protected_head += 1

    # Always protect the final message (the latest turn awaiting a
    # response) — even when it is the only message, or the same message as
    # the protected head (a single system-only request).
    protected_tail_start = max(len(messages) - 1, protected_head)

    dropped_count = 0
    dropped_tokens = 0
    idx = protected_head
    while total_tokens + max_tokens > ceiling_tokens and idx < protected_tail_start:
        dropped_tokens += per_message_tokens[idx]
        total_tokens -= per_message_tokens[idx]
        dropped_count += 1
        idx += 1

    if total_tokens + max_tokens > ceiling_tokens:
        protected_tokens = sum(per_message_tokens[:protected_head]) + sum(
            per_message_tokens[protected_tail_start:]
        )
        raise DgxSglangTokenBudgetError(
            "dgx-sglang token budget exceeded even after dropping every "
            "trimmable turn: system prompt + latest turn alone need "
            f"~{protected_tokens:,} prompt tokens + {max_tokens:,} "
            f"max_tokens > ceiling {ceiling_tokens:,} tokens. Reduce "
            "max_tokens, shorten the latest message, or raise the "
            "model's declared context_length."
        )

    if dropped_count == 0:
        return messages, 0, 0

    trimmed = messages[:protected_head] + messages[idx:]
    return trimmed, dropped_count, dropped_tokens


def enforce_dgx_sglang_token_budget(agent: Any, api_kwargs: dict) -> dict:
    """Last-resort hard clamp — mutates/returns ``api_kwargs`` in place.

    No-op for every provider except dgx-sglang (see ``is_dgx_sglang_route``).
    Call this immediately before the request is handed to the provider
    client, after any compression has already run, so it catches whatever
    compression missed.
    """
    provider = getattr(agent, "provider", None)
    base_url = getattr(agent, "base_url", None)
    if not is_dgx_sglang_route(provider, base_url):
        return api_kwargs

    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return api_kwargs

    context_length = _resolve_context_length(agent)
    ceiling_tokens = context_length - DGX_SGLANG_CONTEXT_MARGIN_TOKENS
    max_tokens = _resolve_max_tokens(agent, api_kwargs)

    from agent.model_metadata import estimate_messages_tokens_rough

    trimmed, dropped_count, dropped_tokens = clamp_messages_to_token_budget(
        messages,
        max_tokens=max_tokens,
        ceiling_tokens=ceiling_tokens,
        token_counter=estimate_messages_tokens_rough,
    )

    if dropped_count:
        prompt_tokens_before = estimate_messages_tokens_rough(messages)
        prompt_tokens_after = prompt_tokens_before - dropped_tokens
        logger.warning(
            "dgx-sglang hard token-budget clamp: dropped %d oldest turn(s) "
            "(~%s tokens) to fit under ceiling — prompt %s -> %s tokens, "
            "+ max_tokens %s <= ceiling %s (context_length=%s, margin=%s)",
            dropped_count,
            f"{dropped_tokens:,}",
            f"{prompt_tokens_before:,}",
            f"{prompt_tokens_after:,}",
            f"{max_tokens:,}",
            f"{ceiling_tokens:,}",
            f"{context_length:,}",
            f"{DGX_SGLANG_CONTEXT_MARGIN_TOKENS:,}",
        )
        api_kwargs["messages"] = trimmed

    return api_kwargs


def _resolve_context_length(agent: Any) -> int:
    """Prefer the provider's declared context_length; fall back to default."""
    compressor = getattr(agent, "context_compressor", None)
    context_length = getattr(compressor, "context_length", None) if compressor else None
    try:
        context_length = int(context_length) if context_length else 0
    except (TypeError, ValueError):
        context_length = 0
    return context_length if context_length > 0 else DGX_SGLANG_DEFAULT_CONTEXT_LENGTH


def _resolve_max_tokens(agent: Any, api_kwargs: dict) -> int:
    """Read the output-token cap this request will actually send."""
    for key in ("max_tokens", "max_completion_tokens"):
        value = api_kwargs.get(key)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    agent_max_tokens = getattr(agent, "max_tokens", None)
    if agent_max_tokens:
        try:
            return int(agent_max_tokens)
        except (TypeError, ValueError):
            pass
    return DGX_SGLANG_DEFAULT_MAX_TOKENS
