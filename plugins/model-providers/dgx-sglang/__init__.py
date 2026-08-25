"""DGX SGLang provider profile — reasoning_effort clamp for this SGLang build.

``provider: dgx-sglang`` names the local SGLang OpenAI-compatible endpoint on
Damien's DGX box (``RadixArk/Qwen3.8-27B-NVFP4`` at ``192.168.0.214:30000``).
Before this profile existed, ``dgx-sglang`` had **no registered provider
profile at all** — ``providers.get_provider_profile("dgx-sglang")`` returned
``None``, so every route in ``agent/transports/chat_completions.py`` that
would normally translate Hermes' ``agent.reasoning_effort`` /
``agent.reasoning_overrides`` config into the wire request never fired for
this provider:

  - the profile-path clamp/mapping hooks never ran (no profile to call them on)
  - the legacy-path ``extra_body["reasoning"]`` block is gated on
    ``agent._supports_reasoning_extra_body()``, which returns False for any
    base_url that isn't nousresearch.com / vercel / github / lmstudio /
    ollama.com / openrouter — the DGX's raw ``192.168.0.214:30000`` matches
    none of those
  - the legacy-path Kimi/TokenHub/LM-Studio ``reasoning_effort`` branches are
    each gated on an exact provider-name/host check that ``dgx-sglang``
    never matches

The only thing that ever reached the wire was a profile's **static**
``providers.dgx-sglang.extra_body`` config block (via
``agent_init._merge_custom_provider_extra_body`` → ``request_overrides`` →
``api_kwargs.update(overrides)``) — whatever a human typed there went
through completely unvalidated. That is the root cause of the 2026-08-26
clamp trap (WS-T bench, ``docs/research/2026-08-26-qwen38-agentic-bench.md``
in the ai-fleet repo): this SGLang build's ``--reasoning-parser qwen3``
only accepts ``reasoning_effort`` values ``none`` (implicit), ``low``,
``medium``, ``xhigh`` — NOT ``high``, even though ``high`` is a valid
value in Hermes' own ``hermes_constants.VALID_REASONING_EFFORTS`` tuple.
A profile with ``reasoning_effort: high`` (or a literal ``high`` typed into
``providers.dgx-sglang.extra_body.reasoning_effort``) reaches this SGLang
build unchanged and gets a hard HTTP 400:
``{"message":"Unexpected reasoning effort high. Supported types are xhigh
(default), medium, and low."}``.

Registering this profile fixes BOTH problems in one place:

1. It makes ``agent.reasoning_effort`` / ``/reasoning <level>`` actually
   take effect for ``dgx-sglang`` for the first time (previously silently
   dropped — the wiring gap above).
2. :meth:`DgxSglangProfile.build_api_kwargs_extras` clamps any effort this
   SGLang build doesn't accept onto the nearest one it does, instead of
   passing it through raw and letting the server 400.

A profile's static ``providers.dgx-sglang.extra_body`` config (if any) is
still applied afterwards by the transport and wins on any key collision —
this profile only fills in the reasoning_effort/enable_thinking keys when a
static override doesn't already set them, so hand-written per-profile
extra_body blocks keep working exactly as before.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# This SGLang build's accepted `reasoning_effort` values, verified by WS-T's
# 2026-08-26 bench (36 direct-API calls across 6 settings) and the RAILS
# vision probe on the same box. `high` is a valid value in Hermes' own
# VALID_REASONING_EFFORTS tuple but is NOT one of these — every other value
# in Hermes' broader vocabulary (`minimal`, `high`, `max`, `ultra`) is
# clamped onto the nearest accepted level below.
_SGLANG_ACCEPTED_EFFORTS = frozenset({"none", "low", "medium", "xhigh"})

# Clamp map for Hermes effort levels this SGLang build rejects outright.
# `minimal` (below `low` in Hermes' vocabulary) clamps down to the lowest
# accepted level; `high`/`max`/`ultra` (at or above the rejected `high`)
# clamp up to `xhigh`, this build's highest accepted level and documented
# default.
_SGLANG_EFFORT_CLAMP = {
    "minimal": "low",
    "high": "xhigh",
    "max": "xhigh",
    "ultra": "xhigh",
}


def clamp_sglang_reasoning_effort(effort: str) -> str:
    """Map a Hermes reasoning-effort string onto a value this SGLang build accepts.

    Passes through unchanged when already accepted (``none``/``low``/
    ``medium``/``xhigh``). Otherwise looks up :data:`_SGLANG_EFFORT_CLAMP`.
    Any level not in either (future Hermes vocabulary additions) falls back
    to ``xhigh`` — this build's own documented default — rather than
    forwarding an unknown string the server has never advertised.
    """
    effort = (effort or "").strip().lower()
    if effort in _SGLANG_ACCEPTED_EFFORTS:
        return effort
    return _SGLANG_EFFORT_CLAMP.get(effort, "xhigh")


class DgxSglangProfile(ProviderProfile):
    """DGX SGLang canary — enable_thinking + clamped reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not isinstance(reasoning_config, dict):
            # Unset → omit both fields, let the server's own default
            # (xhigh, thinking on) apply. Matches the Upstage/Zai precedent
            # of never forcing a preference the user hasn't expressed.
            return extra_body, top_level

        raw_effort = (reasoning_config.get("effort") or "").strip().lower()
        disabled = reasoning_config.get("enabled") is False or raw_effort == "none"

        if disabled:
            # enable_thinking:false is the proven-fast, 100%-reliable setting
            # for agent/tool-use turns (WS-T bench: 2.1s single-call vs
            # 3.8-10.0s for any reasoning-on setting, never lost a call across
            # 30 calls). reasoning_effort:"none" is sent too — this build's
            # own accepted-values list includes it — so either mechanism
            # alone stops the model from reasoning if the other were ever
            # not honoured by a future SGLang build.
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            top_level["reasoning_effort"] = "none"
            return extra_body, top_level

        if raw_effort:
            top_level["reasoning_effort"] = clamp_sglang_reasoning_effort(raw_effort)

        return extra_body, top_level


dgx_sglang = DgxSglangProfile(
    name="dgx-sglang",
    aliases=("sglang", "dgx_sglang"),
    display_name="DGX SGLang",
    description="Local SGLang OpenAI-compatible endpoint (DGX canary)",
    # User-configured per profile via providers.dgx-sglang.api / model.base_url
    # — there is exactly one DGX, but the base_url still comes from config,
    # not a hardcoded default, matching the "custom" provider's convention.
    base_url="",
    env_vars=(),
    auth_type="api_key",
)

register_provider(dgx_sglang)
