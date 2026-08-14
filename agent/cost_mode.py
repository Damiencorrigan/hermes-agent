"""Fleet cost-saving modes, ENFORCED inside hermes (Damien's 2026-07-31 ruling).

Damien can say "cost saving mode 1/2/3" or "cost saving mode off" to any agent
on the machine.  Until now that ruling only bound *Claude Code sessions* — an
agent had to read the rule and choose to obey it.  Hermes cron jobs and live
agent sessions have no such reader in the loop, so a lockdown was advisory
exactly where it mattered most (unattended spend).  This module makes it bind.

Single source of truth: ``~/ai-fleet/cost_mode.json`` —
``{"mode": "off" | "1" | "2" | "3", "set_at": "<ISO>", "set_by": "damien"}``.
Canonical per-mode semantics: ``~/ai-fleet/docs/cost-saving-modes.md``.

What is enforced HERE (deliberately narrow — the rest stays behavioural, in
``~/.hermes/SOUL.md`` §7):

* **mode 3 (lockdown)** — a hard refusal of every metered provider at the two
  chat-completion entry points, plus at the three fallback-SELECTION sites, so
  no rung of a failover chain can quietly become the lane that spends.
* **mode 2 (free-first)** — cron jobs on a metered provider are SKIPPED unless
  the job carries ``"critical": true``.  Live sessions are not gated: a human
  is present to judge "is this critical", which is precisely the judgement
  mode 2 asks for.
* **modes 1 / off** — no gating at all.  Mode 1 is a routing preference, not a
  spend boundary.

FAIL-SAFE, and note this is the OPPOSITE of the spend caps in
``chat_completion_helpers``.  Those fail CLOSED: an unreadable cap is not
permission to spend.  This one fails OPEN (unreadable / missing / malformed
state file ⇒ "off").  The difference is deliberate:

* a spend cap is a *always-on safety backstop* — losing sight of it means
  losing the only thing standing between a runaway loop and a $36 night;
* a cost-saving mode is an *optional, operator-toggled preference* that is
  "off" almost all of the time.  If a missing ``cost_mode.json`` bricked every
  hermes session and cron job, an OPTIONAL feature would have become a
  single-point-of-failure for the whole fleet.  The hard caps are still there
  in every mode, so failing open here loses no money-safety floor — it only
  loses the tighter preference, loudly recoverable by rewriting the file.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Where the state lives ────────────────────────────────────────────────────
_FLEET_ROOT = Path.home() / "ai-fleet"
COST_MODE_FILE = _FLEET_ROOT / "cost_mode.json"
COST_MODE_DOC = _FLEET_ROOT / "docs" / "cost-saving-modes.md"

VALID_MODES = ("off", "1", "2", "3")

# Providers that cost nothing per token because the weights run on this machine
# (or a LAN box the operator owns).  ALLOW-LIST, not a deny-list: a provider
# nobody has classified is treated as metered, which is the safe direction for
# a lockdown.  ``ollama-cloud`` is deliberately absent — it is hosted and
# billed, despite the shared prefix.
#
# Known deliberate over-block: free-TIER cloud lanes (groq's free models, an
# OpenRouter ``:free`` slug) are refused in mode 3 too.  The gate is
# provider-level and cannot verify from a provider name that a given model is
# on a free tier; refusing a free call is recoverable in one sentence, silently
# billing a metered one is not.  ~/ai-fleet/docs/cost-saving-modes.md still
# lists those lanes as mode-3-allowed for the *agent's own routing choices* —
# this hard gate is the floor beneath that, not a restatement of it.
LOCAL_FREE_PROVIDERS = frozenset({
    "ollama",        # bare "ollama" = local server (providers.py alias → custom)
    "ollama-local",
    "local",
    "lmstudio", "lm-studio", "lm_studio",
    "vllm",
    "llamacpp", "llama.cpp", "llama-cpp",
})

# The state file changes only when Damien says a sentence, so re-reading it on
# every API call is pure overhead.  Same 60s TTL as the primary spend-cap
# verdict cache in chat_completion_helpers, for the same reason.
_CACHE_TTL_S = 60.0
_cache: dict[str, Any] = {"expires": 0.0, "mode": "off"}
_cache_lock = threading.Lock()


def read_cost_mode() -> str:
    """Current fleet cost-saving mode, cached ~60s.  Never raises.

    Returns one of ``"off"``, ``"1"``, ``"2"``, ``"3"``.  Anything unreadable,
    missing, malformed, or unrecognised reads as ``"off"`` — see the module
    docstring for why this fails OPEN while the spend caps fail CLOSED.
    """
    now = time.monotonic()
    with _cache_lock:
        if _cache["expires"] > now:
            return _cache["mode"]

    mode = "off"
    try:
        raw = COST_MODE_FILE.read_text(encoding="utf-8")
        value = (json.loads(raw) or {}).get("mode")
        value = str(value).strip().lower()
        if value in VALID_MODES:
            mode = value
        elif value:
            # A typo'd mode must not silently read as a DIFFERENT mode.  "off"
            # is the fail-safe, and the warning is how it gets noticed.
            logger.warning(
                "cost_mode.json has an unrecognised mode %r (expected one of %s)"
                " — treating as 'off'. File: %s",
                value, ", ".join(VALID_MODES), COST_MODE_FILE,
            )
    except FileNotFoundError:
        pass  # the overwhelmingly common case: no mode ever set on this host
    except Exception as exc:
        logger.warning(
            "cost_mode.json unreadable (%s: %s) — treating as 'off' (an OPTIONAL"
            " feature must not break the fleet; the hard spend caps still apply)",
            type(exc).__name__, exc,
        )

    with _cache_lock:
        _cache["expires"] = time.monotonic() + _CACHE_TTL_S
        _cache["mode"] = mode
    return mode


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _cache["expires"] = 0.0
        _cache["mode"] = "off"


def is_metered_provider(provider: Any) -> bool:
    """True when ``provider`` bills per token (i.e. is not a local lane).

    An empty/absent provider is NOT treated as metered: the caller has not
    resolved a lane yet, and refusing on an unknown-yet string would break
    unrelated code paths that pass a placeholder.  The gate runs again once the
    provider is resolved.
    """
    name = (str(provider or "")).strip().lower()
    if not name:
        return False
    return name not in LOCAL_FREE_PROVIDERS


class CostSavingModeBlocked(Exception):
    """A metered call refused because fleet cost-saving mode 3 is active.

    Carries ``status_code = 402`` so ``agent/error_classifier.py`` routes it to
    ``FailoverReason.billing`` (retryable=False, should_fallback=True) with no
    classifier edit — identical to :class:`PrimarySpendCapExceeded`.  The retry
    loop therefore stops hammering the lane instead of burning turns on a
    refusal that will never change.

    ``should_fallback=True`` is safe here BECAUSE the fallback-selection sites
    are gated by :func:`fallback_skip_reason` as well: in mode 3 every rung of
    the chain refuses too, so the chain exhausts and the task surfaces the
    refusal.  There is no rung it can quietly land on.
    """

    status_code = 402


def _refusal(provider: str) -> str:
    return (
        f"REFUSED by FLEET COST-SAVING MODE 3 (lockdown): provider "
        f"'{provider}' is a metered lane and mode 3 blocks every per-token "
        f"lane until the mode is lifted. Only local/$0 providers "
        f"({', '.join(sorted(LOCAL_FREE_PROVIDERS))}) are allowed to run. "
        f"The mode was set by Damien and lives in {COST_MODE_FILE}; the "
        f"canonical per-mode rules are in {COST_MODE_DOC}. This is not a "
        f"credential, quota, or network failure, and no fallback lane will "
        f"answer either — every rung of the chain is gated by the same mode. "
        f"To restore normal operation say 'cost saving mode off' to any agent "
        f"(or set \"mode\": \"off\" in that file). Do not work around this by "
        f"switching provider: state the degraded capability loudly and queue "
        f"or escalate the work."
    )


def enforce_cost_saving_mode(agent: Any) -> None:
    """Refuse a metered chat-completion call while mode 3 is active.

    Called at the top of both chat-completion entry points, mirroring
    ``enforce_primary_spend_cap`` — those two between them cover every
    main-loop turn (streaming delegates to non-streaming for cron/codex, and
    ``direct_api_call`` is only reached from inside the non-streaming one).

    Raises :class:`CostSavingModeBlocked`.  A no-op in modes off/1/2 and for
    local providers.
    """
    if read_cost_mode() != "3":
        return
    provider = (str(getattr(agent, "provider", "") or "")).strip().lower()
    if not is_metered_provider(provider):
        return
    message = _refusal(provider)
    logger.error("%s", message)
    raise CostSavingModeBlocked(message)


def fallback_skip_reason(provider: Any) -> Optional[str]:
    """Skip reason for a FALLBACK chain entry under mode 3, else ``None``.

    The three sites that put a session on a fallback lane
    (``try_activate_fallback``, ``cron/scheduler._select_fallback_runtime``,
    ``agent/agent_init`` init-time fallback) all consult this, so a lockdown
    cannot be routed around by failing over.  Shaped like
    ``_fallback_entry_over_spend_cap``: a short reason string the caller logs
    and adds to its per-session ``unavailable`` set.

    Never raises — a bug in here must not become a new failure mode for the
    ordinary (mode "off") fallback path.
    """
    try:
        if read_cost_mode() != "3":
            return None
        name = (str(provider or "")).strip().lower()
        if not is_metered_provider(name):
            return None
        return f"cost_saving_mode_3_lockdown:{name}"
    except Exception as exc:  # fail OPEN, per the module docstring
        logger.warning("cost-mode fallback gate error (%s) — not skipping",
                       type(exc).__name__)
        return None


def cron_skip_reason(provider: Any, *, critical: bool = False) -> Optional[str]:
    """Reason a scheduled job must be skipped this tick, else ``None``.

    * mode 3 — every metered job is skipped, no exceptions.
    * mode 2 — metered jobs are skipped UNLESS the job is flagged
      ``"critical": true`` in ``~/.hermes/cron/jobs.json``.
    * modes 1 / off — never skipped.

    Skipping is the only correct action: SWAPPING the job onto a cheaper
    provider is exactly the silent provider drift that the #44585 drift guard
    exists to prevent, and an unattended job that quietly answers from a
    different model is worse than one that visibly did not run.

    Never raises (fail-open) for the same reason as the rest of this module.
    """
    try:
        mode = read_cost_mode()
        if mode not in ("2", "3"):
            return None
        name = (str(provider or "")).strip().lower()
        if not is_metered_provider(name):
            return None
        if mode == "2":
            if critical:
                return None
            return (
                f"cost saving mode 2 (free-first): provider '{name}' is metered "
                f"and this job is not flagged critical"
            )
        return f"cost saving mode 3 (lockdown): provider '{name}' is metered"
    except Exception as exc:  # fail OPEN — never let this gate break the cron
        logger.warning("cost-mode cron gate error (%s) — running the job",
                       type(exc).__name__)
        return None


def session_banner() -> Optional[str]:
    """One-line system-prompt notice when a cost-saving mode is active.

    ``None`` in mode "off" (the normal case) so the system prompt is unchanged
    for every ordinary session.  Injected into the VOLATILE system-prompt tier
    so a mode change never invalidates the upstream prompt cache for the
    stable/context tiers.

    Never raises — a broken banner must not break session start-up.
    """
    try:
        mode = read_cost_mode()
        if mode == "off":
            return None
        return (
            f"⚠ FLEET COST-SAVING MODE {mode} ACTIVE — follow {COST_MODE_DOC}"
        )
    except Exception:
        return None


def delegation_dispatch_reason(provider: Any) -> Optional[str]:
    """Reason a ``delegate_task`` subagent dispatch must be refused, else ``None``.

    Mode 3 (lockdown): refuse to spawn a child on a metered lane — the child
    spends on every one of its turns, and the dispatch is the cheapest place to
    stop it (before the subagent exists).  Modes 2 / 1 / off: no hard gate —
    a delegation is a live-session action and the parent agent exercises the
    "is this critical" judgement that mode 2 asks for.

    Never raises (fail-open), for the same reason as the rest of this module.
    """
    try:
        if read_cost_mode() != "3":
            return None
        name = (str(provider or "")).strip().lower()
        if not is_metered_provider(name):
            return None
        return f"cost_saving_mode_3_lockdown:{name}"
    except Exception as exc:  # fail OPEN
        logger.warning("cost-mode delegation gate error (%s) — not refusing",
                       type(exc).__name__)
        return None


def auxiliary_call_block_reason(provider: Any) -> Optional[str]:
    """Reason an auxiliary client call (the context-compression summariser)
    must be downgraded-or-skipped under modes 2 / 3, else ``None``.

    Unlike the live-session main path, the summariser is unattended, so BOTH
    mode 2 (free-first) and mode 3 (lockdown) gate it: a metered aux provider
    must not be called while the fleet is economising.  The caller degrades —
    prefer a local (dgx-ollama) client, else gracefully skip the aux work — and
    must NEVER raise on this, because a broken summariser must not break the
    conversation loop.

    Never raises (fail-open), for the same reason as the rest of this module.
    """
    try:
        mode = read_cost_mode()
        if mode not in ("2", "3"):
            return None
        name = (str(provider or "")).strip().lower()
        if not is_metered_provider(name):
            return None
        return f"cost_saving_mode_{mode}_aux_block:{name}"
    except Exception as exc:  # fail OPEN
        logger.warning("cost-mode aux gate error (%s) — not skipping",
                       type(exc).__name__)
        return None
