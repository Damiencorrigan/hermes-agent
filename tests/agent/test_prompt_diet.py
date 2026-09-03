"""Tests for the per-profile "lean worker" prompt diet (config-gated, default OFF).

Covers agent/prompt_diet.py's pure helpers plus both prompt-build injection
sites:
  * persona SOUL cap in agent/system_prompt.py (build_system_prompt_parts)
  * preloaded-skill-body cap in agent/skill_commands.py (build_preloaded_skills_prompt)

Default-off invariant: with no positive budget, every path must return output
byte-identical to a build without the feature. We assert that explicitly.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.prompt_diet import (
    budget_skill_blocks,
    coerce_char_budget,
    trim_persona,
)
from agent.skill_commands import build_preloaded_skills_prompt
from agent.system_prompt import build_system_prompt_parts


def _make_skill(skills_dir, name, body="Do the thing."):
    """Create a minimal skill directory with SKILL.md (mirrors harness)."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Description for {name}.
---

# {name}

{body}
"""
    )
    return skill_dir


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        # Diet attr defaults to None (OFF) so SimpleNamespace agents never
        # trim unless a test sets it explicitly.
        _lean_prompt_char_budget=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Pure helpers ────────────────────────────────────────────────────────────


class TestCoerceCharBudget:
    def test_positive_int_is_on(self):
        assert coerce_char_budget(8000) == 8000

    def test_zero_is_off(self):
        assert coerce_char_budget(0) is None

    def test_none_is_off(self):
        assert coerce_char_budget(None) is None

    def test_negative_is_off(self):
        assert coerce_char_budget(-5) is None

    def test_numeric_string_is_on(self):
        assert coerce_char_budget("9000") == 9000

    def test_bool_is_off(self):
        # bool is an int subclass; bare True is not a character budget.
        assert coerce_char_budget(True) is None
        assert coerce_char_budget(False) is None

    def test_garbage_is_off(self):
        assert coerce_char_budget("lots") is None
        assert coerce_char_budget(3.5) is None


class TestTrimPersona:
    def test_no_budget_is_byte_identical(self):
        text = "x" * 5000
        assert trim_persona(text, None) == text
        assert trim_persona(text, 0) == text

    def test_short_text_untouched(self):
        text = "Short persona."
        assert trim_persona(text, 10000) == text

    def test_caps_and_adds_marker_when_over(self):
        body = "word\n\n" * 400  # ~3600 chars with paragraph gaps
        result = trim_persona(body, 500)
        assert len(result) <= 500
        assert "[trimmed:" in result

    def test_removes_overlong_single_paragraph_within_budget(self):
        body = "no gaps here so a hard cut happens " * 200  # one long line
        result = trim_persona(body, 300)
        assert len(result) <= 300
        assert "[trimmed:" in result


class TestBudgetSkillBlocks:
    def test_no_budget_returns_blocks_unchanged(self):
        blocks = ["aaa", "bbbb"]
        assert budget_skill_blocks(blocks, None) == blocks
        assert budget_skill_blocks(blocks, 0) == blocks

    def test_keeps_head_drops_lower_priority_tail(self):
        blocks = ["A" * 100, "B" * 100, "C" * 100]
        capped = budget_skill_blocks(blocks, 250)
        # A+B fit (200 + 2 sep), C would overflow => dropped.
        assert capped == blocks[:2]
        assert sum(len(b) for b in capped) <= 250

    def test_truncates_sole_oversized_block_with_marker(self):
        blocks = ["B" * 5000]
        capped = budget_skill_blocks(blocks, 200)
        assert capped
        joined = "".join(capped)
        assert len(joined) <= 200
        assert "[trimmed:" in joined


# ── Skill preload injection site ────────────────────────────────────────────


class TestPreloadedSkillsDiet:
    def _big_body(self, tag):
        # ~1500 chars each so a modest budget forces a decision.
        return f"{tag} " + ("content_" * 300)

    def test_default_off_output_byte_identical(self, tmp_path):
        """No char_budget passed must equal the pre-feature (off) output."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "alpha", body=self._big_body("alpha"))
            _make_skill(tmp_path, "beta", body=self._big_body("beta"))
            off = build_preloaded_skills_prompt(["alpha", "beta"])
            # char_budget=None is the same as not passing it.
            explicit_off = build_preloaded_skills_prompt(["alpha", "beta"], char_budget=None)

        assert off == explicit_off
        # Both skills present in full (names appear in each activation note).
        assert "alpha" in off[0]
        assert "beta" in off[0]
        assert off[1] == ["alpha", "beta"]

    def test_budget_bounds_and_reduces_output(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "alpha", body=self._big_body("alpha"))
            _make_skill(tmp_path, "beta", body=self._big_body("beta"))
            off = build_preloaded_skills_prompt(["alpha", "beta"])[0]
            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["alpha", "beta"], char_budget=400
            )

        assert missing == []
        # The diet output is bounded and smaller than the off default.
        assert len(off) > 400  # sanity: a budget is actually forcing a trim
        assert len(prompt) <= 400
        assert len(prompt) < len(off)

    def test_oversized_single_skill_is_bounded_not_empty(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "solo", body=self._big_body("solo"))
            off = build_preloaded_skills_prompt(["solo"])[0]
            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["solo"], char_budget=250
            )

        assert missing == []
        assert len(off) > 250  # sanity
        assert len(prompt) <= 250
        assert len(prompt) < len(off)
        assert "[trimmed:" in prompt


# ── Persona injection site (build_system_prompt_parts) ──────────────────────


class TestPersonaDiet:
    _LONG_SOUL_IDENTITY = "## Identity\n" + ("You are a worker agent. " * 120)
    _RULES_BLOCK = (
        "\n\n<!-- FLEET-RULES-BEGIN -->\n"
        "RULING_TAIL_SHOULD_SURVIVE_VERBATIM "
        + ("fleet rule content. " * 60)
        + "\n<!-- FLEET-RULES-END -->\n"
    )

    def _stable_for(self, budget):
        agent = _make_agent(_lean_prompt_char_budget=budget)
        soul = self._LONG_SOUL_IDENTITY + self._RULES_BLOCK
        with (
            patch("run_agent.load_soul_md", return_value=soul),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            return build_system_prompt_parts(agent)["stable"]

    def test_default_off_keeps_full_persona_and_rules(self):
        off = self._stable_for(None)
        # 100 consecutive copies survive (strip only eats the trailing space).
        assert "You are a worker agent. " * 100 in off
        assert "RULING_TAIL_SHOULD_SURVIVE_VERBATIM" in off
        assert "[trimmed:" not in off

    def test_budget_caps_persona_but_keeps_sync_rules_tail(self):
        off = self._stable_for(None)
        diet = self._stable_for(600)
        # Persona was trimmed (marker present, output smaller than off).
        assert "[trimmed:" in diet
        assert len(diet) < len(off)
        # The persona lead is capped to ~the budget, while the sync-managed
        # FLEET-RULES block is preserved verbatim (not part of the cap).
        assert "RULING_TAIL_SHOULD_SURVIVE_VERBATIM" in diet
