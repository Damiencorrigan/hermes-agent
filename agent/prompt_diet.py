"""Per-profile "lean worker" prompt-diet helpers (config-gated, default OFF).

Purpose
-------
Some deployments run long-lived worker agents against an upstream that charges
for the characters it re-processes (e.g. an SGLang server whose context cache
holds the whole conversation). Such workers re-send the full system prompt on
every API call. When a profile's persona SOUL plus its preloaded-skill bodies
are large (~68-95K chars), that repeated cost dominates.

The ``agent.lean_prompt_char_budget`` config key (default ``None`` = OFF) gives
a profile a single number: a per-section character cap applied to the persona
SOUL text and to the combined preloaded-skill bodies. Both injection sites run
once at prompt-build time, so enabling the diet never mutates a live
conversation's cached prefix (prompt caching stays intact).

Every function here is pure — it never touches config, files, or agent state.
It only coerces a budget and trims text. Callers decide whether a budget is in
effect.

Off-by-default guarantee
------------------------
When no positive budget is configured, callers skip these helpers entirely, so
prompt output is byte-identical to a build without this feature.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Clear "trimmed here" delimiter markers. Marker text counts toward the budget
# (room is reserved for it), so the returned text never exceeds the cap even
# after the marker is appended.
PERSONA_TRIM_MARKER = (
    "\n\n[trimmed: profile SOUL exceeds the configured lean-prompt character "
    "budget; the full persona lives in the profile SOUL, not re-sent on every "
    "API call.]"
)
SKILL_TRIM_MARKER = (
    "\n\n[trimmed: preloaded skill bodies exceed the configured lean-prompt "
    "character budget; lower-priority bodies and overlong tails were cut. "
    "Invoke the skill again to load its full body.]"
)


def coerce_char_budget(value: object) -> Optional[int]:
    """Return the effective char budget for *value*, or ``None`` when OFF.

    A positive integer is the only "on" value. ``None``, ``0``, a negative
    number, or a non-numeric value all mean "off" (leave prompts untouched).
    Callers gate on the return value: ``if coerce_char_budget(v): ... trim``.

    Args:
        value: The raw config value (``agent.lean_prompt_char_budget``).

    Returns:
        A positive int when a diet is in effect, else ``None``.
    """
    if isinstance(value, bool):
        # bool is an int subclass; a bare True/False is not a char budget.
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        # Tolerate a numeric string from a hand-edited YAML.
        stripped = value.strip()
        if stripped.isdigit() and int(stripped) > 0:
            return int(stripped)
    return None


def _find_head_boundary(text: str, limit: int) -> int:
    """Largest cut index ``<= limit`` on a paragraph gap (or line boundary).

    A paragraph gap is a blank line (``"\\n\\n"``) run. Cutting there keeps
    whole paragraphs rather than slicing mid-sentence. Returns ``-1`` when no
    such boundary exists before *limit* (caller falls back to a hard cut).
    """
    best = -1
    search_from = 0
    while True:
        gap = text.find("\n\n", search_from)
        if gap == -1 or gap > limit:
            break
        candidate = gap + 2  # consume the trailing blank line
        if candidate <= limit:
            best = candidate
        search_from = candidate
    if best == -1:
        nl = text.rfind("\n", 0, limit)
        if nl != -1:
            best = nl + 1
    return best


def trim_text_to_budget(
    text: str,
    budget: object,
    marker: str = "",
) -> Tuple[str, bool]:
    """Trim *text* so ``len(result) <= budget``, appending *marker* when cut.

    Keeps the head and prefers a coherent blank-line paragraph boundary; falls
    back to a hard line/char cut so the result always fits. The marker is
    appended only when a trim happened and room was reserved for it.

    Args:
        text: The content to cap (never mutated).
        budget: Raw budget value; coerced via :func:`coerce_char_budget`. ``0``
            or falsy means no trimming — *text* is returned unchanged.
        marker: Optional marker string appended when trimmed.

    Returns:
        ``(result_text, was_trimmed)``.
    """
    if not text:
        return text, False
    cap = coerce_char_budget(budget)
    if cap is None:
        return text, False
    if len(text) <= cap:
        return text, False

    marker = marker or ""
    body_budget = cap - len(marker)
    if body_budget <= 0:
        # No room for body with the marker: emit just the marker when it fits.
        return (marker if len(marker) <= cap else ""), True

    body = text
    cut = _find_head_boundary(text, body_budget)
    if cut != -1:
        body = text[:cut].rstrip()
    else:
        body = text[:body_budget].rstrip()

    # Guarantee body + marker never exceeds the cap.
    if len(body) + len(marker) > cap:
        body = body[: max(0, cap - len(marker))].rstrip()
    result = (body + marker) if marker else body
    return result, True


def trim_persona(soul_content: str, budget: object) -> str:
    """Cap *soul_content* (persona SOUL text) to *budget*, keeping the head.

    Apply :func:`trim_text_to_budget` with the persona marker. Callers use this
    right after the sync-block split, so the persona's hand-written text is
    trimmed while the separately-appended FLEET-RULES/DAMIEN-RULINGS/FLEET-STATE
    tail is preserved in full.

    Args:
        soul_content: The persona text (already sync-block-split by caller).
        budget: Raw ``agent.lean_prompt_char_budget`` value.

    Returns:
        The capped text (unchanged when no budget is in effect or it already
        fits).
    """
    if not soul_content or coerce_char_budget(budget) is None:
        return soul_content
    trimmed, _ = trim_text_to_budget(soul_content, budget, PERSONA_TRIM_MARKER)
    return trimmed


def budget_skill_blocks(blocks: List[str], budget: object) -> List[str]:
    """Return a capped prefix of preloaded-skill *blocks* under *budget*.

    Blocks arrive in caller priority order (the order skills were given at
    launch). Higher-priority (earlier) blocks are kept whole for as long as the
    running total fits; once a block would overflow, later (lower-priority)
    blocks are dropped. If even the first block overflows on its own, that lone
    block is coherently truncated via :func:`trim_text_to_budget` with the
    skill marker so at least a bounded lead-in survives.

    When no positive budget is in effect, the original list is returned
    unchanged.

    Args:
        blocks: The already-built skill-body blocks, in priority order.
        budget: Raw ``agent.lean_prompt_char_budget`` value.

    Returns:
        The capped block list (may be shorter than *blocks*).
    """
    if not blocks or coerce_char_budget(budget) is None:
        return blocks
    cap = coerce_char_budget(budget)

    kept: List[str] = []
    total = 0
    for blk in blocks:
        sep = 2 if kept else 0  # "\n\n" join separator between blocks
        if total + sep + len(blk) <= cap:
            kept.append(blk)
            total += sep + len(blk)
            continue
        # Overflow. If nothing kept yet, salvage a bounded lead-in from this
        # single oversized block so the diet never emits an empty prompt.
        if not kept:
            trimmed, _ = trim_text_to_budget(blk, cap, SKILL_TRIM_MARKER)
            if trimmed:
                kept.append(trimmed)
        # All later blocks are lower priority than the overflowing one: drop.
        break
    return kept
