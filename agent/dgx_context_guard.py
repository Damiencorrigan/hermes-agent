"""Hermes-side integration with ai-fleet's ops/dgx_context_guard.py.

WHY THIS EXISTS (2026-08-26, HERMES-BOUND mission, Commander DGX-stability wave)

The DGX box (192.168.0.214) serves one model, RadixArk/Qwen3.8-27B-NVFP4, on
SGLang :30000. On 2026-08-26 06:23 the host wedged: dmesg showed a global OOM
(07:15 also) plus NVRM NV_ERR_NO_MEMORY, triggered after an earlier tuning
change lifted server admission (``max_running_requests``) to 16 and roughly
700K input tokens arrived from the fleet inside five minutes. ai-fleet's own
``ops/dgx_context_guard.py`` bounds concurrent long-context requests to
protect the box — but ONLY for callers that go through
``fleet_router.py``'s ``call_lane_api`` chokepoint. Hermes profile gateways
(``~/.hermes/profiles/*/config.yaml``, ~35 of them at the time of writing)
POST straight to ``http://192.168.0.214:30000/v1`` and bypass that guard
entirely — the exact gap ``ops/dgx_context_guard.py``'s own module
docstring names as "a documented, known gap" it does not close.

This module closes it from the Hermes side, without forking the guard: it
imports ``ops.dgx_context_guard`` from the ai-fleet checkout **by path** and
calls its real ``dgx_slot()`` context manager, so a Hermes profile request
and a ``fleet_router`` request draw from the exact same lock-file pools
(same state directory under ``~/ai-fleet/ops/state/dgx_context_guard/``) —
one shared TOTAL pool (``fleet_config.json`` ``dgx.max_concurrent_workers``,
6 fleet-wide as of this mission) and one shared BIG-prompt pool
(``dgx.max_concurrent_big_prompt``, 2), not two independently-capped pools
that could each individually stay "within limit" while jointly overloading
the box.

WHAT THIS GUARDS

Applies only to a request whose ``agent.base_url`` host:port matches the DGX
guard allowlist (default ``192.168.0.214:30000``; override with the
comma-separated env var ``HERMES_DGX_GUARD_HOSTS``). Every other provider's
request is untouched — this module is a complete no-op for OpenRouter,
Anthropic, DeepSeek, Ollama Cloud, and any other base_url.

For an in-allowlist request it:

  1. Estimates the prompt size (``estimate_prompt_tokens`` — chars/3.5 over
     the serialized ``messages`` + ``tools``, a deliberately distinct and
     slightly more conservative ratio than the ai-fleet guard's own
     chars/4, because Hermes requests carry tool schemas the fleet_router
     estimator was never tuned against).
  2. Reserves a TOTAL-pool slot always, waiting up to
     ``WORKER_WAIT_TIMEOUT_S`` (5 min).
  3. When the estimate exceeds the guard's configured
     ``big_prompt_token_threshold`` (shared config, default 80,000),
     additionally reserves a BIG-pool slot, with the larger
     ``BIG_PROMPT_WAIT_TIMEOUT_S`` (10 min) timeout budget covering both
     pools together (mirrors ``ops.dgx_context_guard.dgx_slot``'s own
     single-deadline design).
  4. On timeout, lets ``ops.dgx_context_guard.DGXContextGuardTimeoutError``
     (a ``RuntimeError``) propagate unchanged out of the call. This module
     never proceeds unguarded and never swallows the timeout into a silent
     no-op — Hermes' existing fallback ladder (OpenRouter, etc.) takes over
     exactly as it does for any other dispatch failure, because the
     guarded call sites in ``run_agent.py`` are the same call sites every
     other request-time exception already flows through.
  5. Logs the acquire/release/timeout to the guard's own shared log
     (``~/ai-fleet/ops/state/dgx_context_guard/guard.log``) via
     ``dgx_slot(..., caller_label="hermes:<profile-name>")`` — the profile
     name comes from ``$HERMES_PROFILE`` / ``$HERMES_PROFILE_NAME`` (the
     same env-var pair ``hermes_cli/runtime_provider.py`` already reads for
     diagnostics), so a slot holder is attributable to a specific gateway.

FAILS LOUD, NEVER SILENT (no_silent_failover doctrine)

If ``ops.dgx_context_guard`` cannot be imported from the ai-fleet checkout
(missing checkout, moved path, syntax error introduced upstream), this
module raises :class:`DgxContextGuardUnavailable` rather than falling back
to an unguarded request — a request that matches the DGX allowlist must
either get a real slot from the real shared guard, or fail with a clear
error the caller's fallback ladder can act on. It must never silently
proceed as if no guard existed; that is exactly the 06:23 failure mode this
module exists to close.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Search order for the ai-fleet checkout. HERMES_AI_FLEET_ROOT lets a
# non-standard checkout location (a worktree, a test fixture) override the
# default without editing this file.
_AI_FLEET_ROOT_CANDIDATES = (
    os.environ.get("HERMES_AI_FLEET_ROOT", "").strip(),
    os.path.expanduser("~/ai-fleet"),
)

DEFAULT_DGX_HOSTS = ("192.168.0.214:30000",)

# "e.g. 10 min for a long-context slot, 5 min for a worker slot" (mission
# brief). Both are bounded waits, never unbounded — see module docstring.
WORKER_WAIT_TIMEOUT_S = 300
BIG_PROMPT_WAIT_TIMEOUT_S = 600

# Deliberately distinct from ops.dgx_context_guard.CHARS_PER_TOKEN_ESTIMATE
# (4) -- see module docstring.
CHARS_PER_TOKEN_ESTIMATE = 3.5


class DgxContextGuardUnavailable(RuntimeError):
    """Raised when ops.dgx_context_guard cannot be imported from ai-fleet.

    Hard failure by design: see the module docstring's "FAILS LOUD, NEVER
    SILENT" section. A caller must never treat this as "no guard needed".
    """


_guard_module = None  # cached after first successful import, per process


def _find_ai_fleet_root() -> str:
    for candidate in _AI_FLEET_ROOT_CANDIDATES:
        if candidate and os.path.isfile(
            os.path.join(candidate, "ops", "dgx_context_guard.py")
        ):
            return candidate
    searched = [c for c in _AI_FLEET_ROOT_CANDIDATES if c]
    raise DgxContextGuardUnavailable(
        "dgx_context_guard: could not find ops/dgx_context_guard.py under "
        f"any of {searched}. Set HERMES_AI_FLEET_ROOT to the ai-fleet "
        "checkout path, or restore ~/ai-fleet. Refusing to send an "
        "unguarded request to a DGX allowlisted target — this is a hard "
        "failure by design (no_silent_failover), not an ignorable warning."
    )


def _load_guard_module():
    """Import ops.dgx_context_guard from the ai-fleet checkout, by path.

    Cached module-level after first success (a long-lived gateway process
    should not re-walk the filesystem on every request). Never returns
    None — raises DgxContextGuardUnavailable on any failure.
    """
    global _guard_module
    if _guard_module is not None:
        return _guard_module
    root = _find_ai_fleet_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import ops.dgx_context_guard as guard  # type: ignore
    except Exception as exc:
        raise DgxContextGuardUnavailable(
            f"dgx_context_guard: found ai-fleet at {root} but failed to "
            f"import ops.dgx_context_guard ({exc!r}). Refusing to send an "
            "unguarded request to a DGX allowlisted target."
        ) from exc
    _guard_module = guard
    return guard


def _dgx_hosts() -> tuple:
    raw = os.environ.get("HERMES_DGX_GUARD_HOSTS", "").strip()
    if not raw:
        return DEFAULT_DGX_HOSTS
    hosts = tuple(h.strip() for h in raw.split(",") if h.strip())
    return hosts or DEFAULT_DGX_HOSTS


def _host_port(base_url: str) -> str:
    raw = (base_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return ""
    port = parsed.port
    return f"{host}:{port}" if port else host


def is_dgx_target(base_url: str) -> bool:
    """True when *base_url* points at a host:port on the DGX guard allowlist."""
    hp = _host_port(base_url)
    if not hp:
        return False
    return hp in _dgx_hosts()


def _profile_name() -> str:
    """The active Hermes profile, for guard-log attribution.

    Same env-var pair (and precedence) hermes_cli/runtime_provider.py
    already reads for its own diagnostics — kept consistent so a slot
    holder in the guard log matches what the rest of Hermes calls "the
    profile" elsewhere.
    """
    return (
        os.environ.get("HERMES_PROFILE")
        or os.environ.get("HERMES_PROFILE_NAME")
        or "default"
    )


def estimate_prompt_tokens(api_kwargs: dict) -> int:
    """Rough chars/3.5 token estimate over ``messages`` + ``tools``.

    Labelled ESTIMATE, not a real tokenizer count — good enough to pick a
    pool, not to bill or report as measured usage. See module docstring for
    why this uses a different ratio than ai-fleet's own estimator.
    """
    try:
        messages = api_kwargs.get("messages") or []
        tools = api_kwargs.get("tools") or []
        text = (
            json.dumps(messages, default=str, ensure_ascii=False)
            + json.dumps(tools, default=str, ensure_ascii=False)
        )
    except Exception:
        text = str(api_kwargs)
    return max(0, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


@contextlib.contextmanager
def dgx_request_guard(agent, api_kwargs: dict):
    """Reserve a DGX concurrency slot for one Hermes chat-completion request.

    No-op (yields immediately, no import, no filesystem access) unless
    ``agent.base_url`` matches the DGX guard allowlist — see
    :func:`is_dgx_target`. Every other provider's request pays zero cost.

    When it applies, blocks up to :data:`BIG_PROMPT_WAIT_TIMEOUT_S` (10 min)
    for a big prompt or :data:`WORKER_WAIT_TIMEOUT_S` (5 min) otherwise, then
    either yields with a slot held (released in a ``finally`` no matter how
    the caller's request finishes) or lets
    ``ops.dgx_context_guard.DGXContextGuardTimeoutError`` propagate. See the
    module docstring for the full contract.
    """
    base_url = getattr(agent, "base_url", "") or ""
    if not is_dgx_target(base_url):
        yield
        return

    guard = _load_guard_module()
    cfg = guard.load_config()
    tokens = estimate_prompt_tokens(api_kwargs)
    is_big = tokens > cfg.big_prompt_token_threshold
    timeout_s = BIG_PROMPT_WAIT_TIMEOUT_S if is_big else WORKER_WAIT_TIMEOUT_S
    caller_label = f"hermes:{_profile_name()}"

    with guard.dgx_slot(
        tokens, config=cfg, timeout_s=timeout_s, caller_label=caller_label
    ):
        yield
