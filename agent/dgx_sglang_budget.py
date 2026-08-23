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
size, and it does not run on every path that can (re)assemble and send a
request. On 2026-08-23 that gap let a 144,706-token prompt reach the
provider and get rejected with HTTP 400: "The input (144706 tokens) is
longer than the model's context length (131072 tokens)."

Two send paths, not one
------------------------
An adversarial review of the first version of this fix (2026-08-23) found
it only guarded ``agent.chat_completion_helpers.build_api_kwargs`` — the
main conversation-turn send path — and missed two others that dgx-sglang
also uses:

  (a) The iteration-limit "final summary" call in
      ``agent/chat_completion_helpers.py``, which builds ``summary_kwargs``
      as a raw dict and calls ``summary_client.chat.completions.create()``
      directly. This fires exactly when history is largest.
  (b) The auxiliary/fallback router in ``agent/auxiliary_client.py``
      (context-compression's own summarization call, vision, session
      search, etc., plus same-provider retry and cross-provider fallback).
      Every one of those call sites builds its kwargs through the single
      shared ``_build_call_kwargs()`` before sending, so that function is
      this module's chokepoint for the whole file rather than needing to
      patch each of its ~10 call sites individually.

This module therefore exposes two entry points sharing one core:
``enforce_dgx_sglang_token_budget`` (agent-object callers — build_api_kwargs
and the summary call) and ``enforce_dgx_sglang_token_budget_raw``
(primitive-args callers — auxiliary_client._build_call_kwargs, which has no
``agent`` reference, only provider/model/base_url strings).

Tool-call atomicity
--------------------
Dropping oldest messages by raw list index can separate an assistant
message carrying ``tool_calls`` from its paired ``tool`` result message (or
vice versa), which every strict OpenAI-compatible provider — SGLang
included — rejects with "No tool call found for function call output".
Rather than reimplementing hermes's tool-pairing rules here, every drop is
followed by ``agent.agent_runtime_helpers.sanitize_api_messages()`` — the
same unconditional pre-send chokepoint the rest of hermes already relies on
— which drops any orphaned tool result and/or injects a stub result for any
orphaned tool_call. The trim loop re-measures tokens and re-derives the
protected head/tail against the *sanitized* list on every iteration, so it
stays correct even though sanitize can occasionally add a short stub
message back.

Undercounting margin
---------------------
``estimate_messages_tokens_rough`` (reused from hermes's existing Pre-API
compression measurement, not reimplemented here) is a ~4-chars/token
approximation and undercounts BPE-dense JSON/tool-call payloads by roughly
20-25%. A flat margin sized as a small fraction of the context window
(2048 / 131072 = 1.6%) cannot absorb that — a request this module judged to
"fit" could still 400. The margin is therefore sized as a percentage of the
*estimated prompt being sent* (``DGX_SGLANG_CONTEXT_MARGIN_PERCENT``, not of
the context window), with a small flat floor for tiny requests
(``DGX_SGLANG_CONTEXT_MARGIN_TOKENS``) — whichever is larger.
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
# "http://192.168.0.214:30000/v1". Deliberately host+port (not host alone) —
# the same DGX box also serves dgx-ollama on :11434 with a different (much
# smaller) declared context_length, which must NOT be caught by this gate.
DGX_SGLANG_HOST = "192.168.0.214"
DGX_SGLANG_PORT = "30000"

# Fallback context window when the provider's declared context_length isn't
# available at runtime (e.g. context_compressor not yet initialized for this
# model, or no prior probe/config lookup succeeded). Mirrors
# providers.dgx-sglang.models."RadixArk/Qwen3.8-27B-NVFP4".context_length in
# the live profile configs.
DGX_SGLANG_DEFAULT_CONTEXT_LENGTH = 131072

# Flat floor for the safety margin — always subtracted at minimum, even for
# a tiny request where the percentage margin below would round to ~0.
# Covers wire-protocol overhead (message envelope/field JSON) that the rough
# per-message char-count estimator doesn't model.
DGX_SGLANG_CONTEXT_MARGIN_TOKENS = 2048

# Percentage-of-estimated-prompt margin. estimate_messages_tokens_rough is a
# ~4-chars/token approximation that undercounts BPE-dense JSON/tool-call
# payloads by roughly 20-25% (adversarial review finding, 2026-08-23) — a
# flat 2048-token margin (1.6% of the 131072 context window) cannot absorb
# that on a large request. 0.18 (18%) sits inside the reviewed 15-20% band.
# The EFFECTIVE margin is max(flat floor, this percent * prompt_tokens) —
# see ``_effective_margin_tokens`` — so small requests still get the flat
# floor and large requests scale with their own undercount risk.
DGX_SGLANG_CONTEXT_MARGIN_PERCENT = 0.18

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
    other provider/base_url returns False, leaving it untouched. The port
    check specifically excludes the dgx-ollama route on the same host
    (:11434) — same box, different server, different context_length.
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


def _effective_margin_tokens(prompt_tokens: int) -> int:
    """max(flat floor, percent * prompt_tokens) — see module docstring."""
    percent_margin = round(prompt_tokens * DGX_SGLANG_CONTEXT_MARGIN_PERCENT)
    return max(DGX_SGLANG_CONTEXT_MARGIN_TOKENS, percent_margin)


def _budget_error_text(protected_tokens: int, max_tokens: int, ceiling_tokens: int) -> str:
    return (
        "dgx-sglang token budget exceeded even after dropping every "
        "trimmable turn: system prompt + latest turn alone need "
        f"~{protected_tokens:,} prompt tokens + {max_tokens:,} "
        f"max_tokens > ceiling {ceiling_tokens:,} tokens. Reduce "
        "max_tokens, shorten the latest message, or raise the "
        "model's declared context_length."
    )


def _default_sanitize(messages: list) -> list:
    from agent.agent_runtime_helpers import sanitize_api_messages

    return sanitize_api_messages(messages)


def clamp_messages_to_token_budget(
    messages: list,
    *,
    max_tokens: int,
    ceiling_tokens: int,
    token_counter: Callable[[list], int],
    sanitize_fn: Optional[Callable[[list], list]] = None,
) -> tuple[list, int, int]:
    """Drop the oldest trimmable turns until the request fits the ceiling.

    Preserves every leading ``system``-role message and the final message
    (the latest turn the model must respond to) — those are never
    candidates for removal. Turns are dropped oldest-first from the
    remaining middle section, one message at a time, until
    ``prompt_tokens + max_tokens <= ceiling_tokens``.

    After every single-message drop, ``sanitize_fn`` (defaults to hermes's
    own ``agent.agent_runtime_helpers.sanitize_api_messages`` — pass a
    no-op ``lambda m: m`` to disable) runs over the working list so an
    assistant message carrying ``tool_calls`` can never be separated from
    its paired ``tool`` result without repair: an orphaned tool result is
    dropped, an orphaned tool_call gets a stub result injected. Because
    sanitize can occasionally add a short message back, the protected
    head/tail window and the token count are both re-derived from the
    *current* working list on every iteration rather than computed once
    up front.

    Returns ``(trimmed_messages, dropped_count, dropped_tokens)``. When the
    input already fits, returns the input unchanged with
    ``dropped_count == 0``.

    Raises ``DgxSglangTokenBudgetError`` when even the protected head
    (system messages) plus the protected tail (the latest turn) alone
    cannot fit under the ceiling, or when repeated sanitize-driven
    re-inflation prevents convergence within a bounded number of
    iterations — either way, there is nothing left this function is
    permitted to trim, and sending anyway would just reproduce the 400
    this clamp exists to prevent.
    """
    if not messages:
        return messages, 0, 0

    if sanitize_fn is None:
        sanitize_fn = _default_sanitize

    original_tokens = token_counter(messages)
    if original_tokens + max_tokens <= ceiling_tokens:
        return messages, 0, 0

    def _protected_head(msgs: list) -> int:
        head = 0
        while (
            head < len(msgs)
            and isinstance(msgs[head], dict)
            and msgs[head].get("role") == "system"
        ):
            head += 1
        return head

    working = list(messages)
    dropped_count = 0
    # Each iteration removes exactly one message from `working` before
    # sanitize runs, so this can never fire more times than the input is
    # long — a hard termination guarantee independent of how sanitize_fn
    # behaves.
    max_iterations = len(messages)

    for _ in range(max_iterations):
        current_tokens = token_counter(working)
        if current_tokens + max_tokens <= ceiling_tokens:
            break
        head = _protected_head(working)
        tail_start = max(len(working) - 1, head)
        if tail_start <= head:
            raise DgxSglangTokenBudgetError(
                _budget_error_text(current_tokens, max_tokens, ceiling_tokens)
            )
        del working[head]
        dropped_count += 1
        working = sanitize_fn(working)
    else:
        raise DgxSglangTokenBudgetError(
            _budget_error_text(token_counter(working), max_tokens, ceiling_tokens)
        )

    if dropped_count == 0:
        return messages, 0, 0

    dropped_tokens = original_tokens - token_counter(working)
    return working, dropped_count, dropped_tokens


def _log_and_apply(
    api_kwargs: dict,
    messages: list,
    *,
    context_length: int,
    max_tokens: int,
) -> dict:
    from agent.model_metadata import estimate_messages_tokens_rough

    prompt_tokens_before = estimate_messages_tokens_rough(messages)
    margin = _effective_margin_tokens(prompt_tokens_before)
    ceiling_tokens = context_length - margin

    trimmed, dropped_count, dropped_tokens = clamp_messages_to_token_budget(
        messages,
        max_tokens=max_tokens,
        ceiling_tokens=ceiling_tokens,
        token_counter=estimate_messages_tokens_rough,
    )

    if dropped_count:
        prompt_tokens_after = estimate_messages_tokens_rough(trimmed)
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
            f"{margin:,}",
        )
        api_kwargs["messages"] = trimmed

    return api_kwargs


def enforce_dgx_sglang_token_budget(agent: Any, api_kwargs: dict) -> dict:
    """Last-resort hard clamp for agent-object send sites — mutates/returns
    ``api_kwargs`` in place.

    No-op for every provider except dgx-sglang (see ``is_dgx_sglang_route``).
    Call this immediately before the request is handed to the provider
    client, after any compression has already run, so it catches whatever
    compression missed. Used by both
    ``agent.chat_completion_helpers.build_api_kwargs`` (the main
    conversation-turn send path) and the iteration-limit summary call in
    the same module.
    """
    provider = getattr(agent, "provider", None)
    base_url = getattr(agent, "base_url", None)
    if not is_dgx_sglang_route(provider, base_url):
        return api_kwargs

    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return api_kwargs

    context_length = _resolve_context_length_from_agent(agent)
    max_tokens = _resolve_max_tokens(agent, api_kwargs)
    return _log_and_apply(
        api_kwargs, messages, context_length=context_length, max_tokens=max_tokens
    )


def enforce_dgx_sglang_token_budget_raw(
    *,
    provider: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    api_kwargs: dict,
) -> dict:
    """Last-resort hard clamp for primitive-args send sites — mutates/
    returns ``api_kwargs`` in place.

    Used by ``agent.auxiliary_client._build_call_kwargs``, the single
    shared kwargs-builder behind every auxiliary/fallback/retry call in
    that module (context-compression's own summarization call, vision,
    session search, same-provider retry, cross-provider fallback — sync
    and async). None of those call sites carry a live ``agent`` object, so
    context_length is resolved from the read-only local cache
    (``agent.model_metadata.get_cached_context_length`` — no network call)
    and, failing that, from the live hermes config's declared
    ``providers.dgx-sglang.models.<model>.context_length``, before falling
    back to ``DGX_SGLANG_DEFAULT_CONTEXT_LENGTH``.
    """
    if not is_dgx_sglang_route(provider, base_url):
        return api_kwargs

    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return api_kwargs

    context_length = _resolve_context_length_raw(model, base_url, provider)
    max_tokens = _resolve_max_tokens(None, api_kwargs)
    return _log_and_apply(
        api_kwargs, messages, context_length=context_length, max_tokens=max_tokens
    )


def _resolve_context_length_from_agent(agent: Any) -> int:
    """Prefer the provider's declared context_length; fall back to default."""
    compressor = getattr(agent, "context_compressor", None)
    context_length = getattr(compressor, "context_length", None) if compressor else None
    try:
        context_length = int(context_length) if context_length else 0
    except (TypeError, ValueError):
        context_length = 0
    return context_length if context_length > 0 else DGX_SGLANG_DEFAULT_CONTEXT_LENGTH


def _resolve_context_length_raw(
    model: Optional[str], base_url: Optional[str], provider: Optional[str]
) -> int:
    """Read-only context_length resolution with no ``agent`` object and no
    network calls: local probe cache, then live config, then the default.
    """
    try:
        from agent.model_metadata import get_cached_context_length

        cached = get_cached_context_length(model or "", base_url or "")
        if cached:
            cached_int = int(cached)
            if cached_int > 0:
                return cached_int
    except Exception:
        logger.debug("dgx-sglang budget: cached context_length lookup failed", exc_info=True)

    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        providers_cfg = config.get("providers") or {}
        provider_key = (provider or "").strip()
        provider_cfg = (
            providers_cfg.get(provider_key)
            or providers_cfg.get(provider_key.lower())
            or providers_cfg.get(DGX_SGLANG_PROVIDER_NAME)
            or {}
        )
        models_cfg = provider_cfg.get("models") or {}
        model_cfg = models_cfg.get(model or "") or {}
        declared = model_cfg.get("context_length")
        if declared:
            declared_int = int(declared)
            if declared_int > 0:
                return declared_int
    except Exception:
        logger.debug("dgx-sglang budget: config context_length lookup failed", exc_info=True)

    return DGX_SGLANG_DEFAULT_CONTEXT_LENGTH


def _resolve_max_tokens(agent: Optional[Any], api_kwargs: dict) -> int:
    """Read the output-token cap this request will actually send."""
    for key in ("max_tokens", "max_completion_tokens"):
        value = api_kwargs.get(key)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    agent_max_tokens = getattr(agent, "max_tokens", None) if agent is not None else None
    if agent_max_tokens:
        try:
            return int(agent_max_tokens)
        except (TypeError, ValueError):
            pass
    return DGX_SGLANG_DEFAULT_MAX_TOKENS
