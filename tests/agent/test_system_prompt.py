"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt, build_system_prompt_parts


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
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(
        cwd=None, skip_soul=False, context_length=None,
        allow_install_tree_fallback=False,
    ):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _prompt_parts(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)


def _init_code_repo(path):
    """A git repo that actually holds code — the coding posture requires a source
    file (or manifest), not a bare ``.git`` (a prose/notes repo stays general)."""
    import subprocess

    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "main.py").write_text("print('hi')\n")


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        parts = _prompt_parts(agent)
        assert "coding agent" in parts["stable"]
        assert "Workspace" in parts["context"]

    def test_absent_when_off(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)


def test_build_system_prompt_records_stable_prefix():
    agent = _make_agent()
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value="context"),
    ):
        prompt = build_system_prompt(agent)

    assert prompt.startswith(agent._cached_system_prompt_static)
    assert prompt[len(agent._cached_system_prompt_static):].startswith("\n\ncontext")


def test_coding_prompt_preserves_legacy_workspace_order(monkeypatch):
    """The cache split must not reorder the stored coding prompt."""
    import agent.system_prompt as system_prompt

    agent = _make_agent(
        valid_tool_names=["read_file"],
        _parallel_tool_call_guidance=False,
    )
    monkeypatch.setattr(system_prompt, "DEFAULT_AGENT_IDENTITY", "IDENTITY")
    monkeypatch.setattr(system_prompt, "HERMES_AGENT_HELP_GUIDANCE", "HELP")
    monkeypatch.setattr(system_prompt, "STEER_CHANNEL_NOTE", "STEER")
    monkeypatch.setattr(system_prompt, "get_hermes_home", lambda: Path("/hermes"))

    expected_profile = (
        "Active Hermes profile: default. Other profiles (if any) live "
        "under /hermes/profiles/<name>/. Each profile has its own skills/, "
        "plugins/, cron/, and memories/ that affect a different session than "
        "this one. Do not modify another profile's skills/plugins/cron/memories "
        "unless the user explicitly directs you to."
    )
    expected = "\n\n".join((
        "IDENTITY",
        "HELP",
        "STEER",
        "CODING_STABLE",
        "WORKSPACE",
        "Operator instructions (from config):\nOPERATOR",
        expected_profile,
        "SYSTEM_MESSAGE",
        "CONTEXT_FILES",
        "Conversation started: Friday, January 02, 2026",
    ))

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value="CONTEXT_FILES"),
        patch(
            "agent.coding_context.coding_system_prompt_parts",
            return_value=(
                ["CODING_STABLE"],
                ["WORKSPACE"],
                ["Operator instructions (from config):\nOPERATOR"],
            ),
        ),
        patch("agent.file_safety._resolve_active_profile_name", return_value="default"),
        patch("hermes_time.now", return_value=datetime(2026, 1, 2)),
    ):
        prompt = build_system_prompt(agent, system_message="SYSTEM_MESSAGE")

    assert prompt == expected
    assert agent._cached_system_prompt_static == "\n\n".join(expected.split("\n\n")[:4])


class TestTelegramRichMessagesHint:
    """Verify that TELEGRAM_RICH_MESSAGES_HINT is conditionally included."""

    def test_base_hint_without_rich_messages(self, monkeypatch):
        """When rich_messages is False, only the base hint is used."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"rich_messages": False}}}}
            }
            stable = _stable_prompt(agent)
        assert "Standard Markdown is automatically converted" in stable
        assert "lean into it" not in stable
        assert "task lists" not in stable

    def test_rich_hint_with_rich_messages_enabled(self, monkeypatch):
        """When rich_messages is True in gateway.platforms, the extension
        is appended (the canonical/primary location)."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"rich_messages": True}}}}
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable
        assert "task lists" in stable
        assert "math/formulas" in stable

    def test_rich_hint_from_top_level_platforms(self):
        """Top-level ``platforms.telegram.extra.rich_messages`` is merged
        alongside gateway.platforms, so it works on its own."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "platforms": {"telegram": {"extra": {"rich_messages": True}}}
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable
        assert "task lists" in stable

    def test_top_level_overrides_gateway_rich_messages(self):
        """Top-level ``platforms.telegram.extra`` wins over gateway.platforms
        at the leaf, matching the adapter's merge precedence."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"rich_messages": False}}}},
                "platforms": {"telegram": {"extra": {"rich_messages": True}}},
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable

    def test_gateway_extra_other_keys_does_not_block_top_level_rich_messages(self):
        """When gateway.platforms.telegram.extra has other keys but not
        rich_messages, the top-level rich_messages still activates."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": {"disable_link_previews": True}}}},
                "platforms": {"telegram": {"extra": {"rich_messages": True}}},
            }
            stable = _stable_prompt(agent)
        assert "lean into it" in stable

    def test_base_hint_without_config(self, monkeypatch):
        """When config has no telegram section, only base hint is used."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {}
            stable = _stable_prompt(agent)
        assert "Standard Markdown is automatically converted" in stable
        assert "lean into it" not in stable


    def test_gateway_rich_messages_integration_via_real_config(self, tmp_path, monkeypatch):
        """End-to-end through the real config-resolution chain: a config.yaml
        under HERMES_HOME with ``gateway.platforms.telegram.extra.rich_messages``
        must activate the rich hint. ``load_config_readonly`` is NOT mocked here,
        so this guards against the exact path-mismatch bug this PR fixes.
        """
        config_yaml = (
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      extra:\n"
            "        rich_messages: true\n"
        )
        home = tmp_path / "hermes_home"
        home.mkdir()
        (home / "config.yaml").write_text(config_yaml)

        monkeypatch.setenv("HERMES_HOME", str(home))
        # Point config resolution at the temp file without mocking the loader:
        # mirror the pattern used in test_config_env_expansion.py.
        from hermes_cli import config as _cfgmod
        monkeypatch.setattr(_cfgmod, "get_config_path", lambda: home / "config.yaml")

        agent = _make_agent(platform="telegram")
        stable = _stable_prompt(agent)
        assert "lean into it" in stable
        assert "task lists" in stable

    def test_malformed_extra_value_falls_back_to_base_hint(self, tmp_path, monkeypatch):
        """A truthy non-mapping ``extra`` must not crash prompt construction —
        it should fail open to the base hint (Tek's fail-open concern).
        """
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "gateway": {"platforms": {"telegram": {"extra": "not-a-map"}}}
            }
            stable = _stable_prompt(agent)
        assert "Standard Markdown is automatically converted" in stable
        assert "lean into it" not in stable


_SKILLS = "SKILLS_INDEX_SENTINEL"
_CONTEXT = "CONTEXT_FILES_SENTINEL"


def _build(builder, **overrides):
    """Run a build_* function with skills + context files present."""
    agent = _make_agent(valid_tool_names=["skills_list"], **overrides)
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=_CONTEXT),
        patch("run_agent.get_toolset_for_tool", return_value=None),
        patch("run_agent.build_skills_system_prompt", return_value=_SKILLS),
    ):
        return builder(agent)


class TestSkillsInVolatileBand:
    """The skills index is runtime-mutable, so it lives in the volatile band,
    not the stable band, to keep the cached stable prefix reusable when a
    rebuild picks up a skill change."""

    def test_skills_not_in_stable_band(self):
        parts = _build(build_system_prompt_parts)
        assert _SKILLS not in parts["stable"]

    def test_skills_lead_the_volatile_band(self):
        parts = _build(build_system_prompt_parts)
        assert parts["volatile"].startswith(_SKILLS)

    def test_full_order_is_stable_context_then_skills(self):
        # build_system_prompt joins stable + context + volatile, so the skills
        # index renders after the context files and before the per-turn
        # memory/timestamp tail.
        full = _build(build_system_prompt)
        assert full.index(_CONTEXT) < full.index(_SKILLS)
        assert full.index(_SKILLS) < full.index("Conversation started:")


_SOUL_IDENTITY = "## Identity\nYou are Hermes, a helpful assistant."


def _soul_with_rules(ruling_text: str) -> str:
    """A SOUL.md body carrying the sync-managed FLEET-RULES/DAMIEN-RULINGS
    marker blocks that ops/soul_rules_sync.py and ops/rulings_soul_sync.py
    periodically rewrite (see docs/level5-rulings-20260815.md in ai-fleet).
    Only *ruling_text* differs between the two builds under test — this
    models what a sync rewrite actually changes on disk.
    """
    return (
        f"{_SOUL_IDENTITY}\n\n"
        "<!-- FLEET-RULES-BEGIN -->\n"
        f"Consolidated fleet rules. {ruling_text}\n"
        "<!-- FLEET-RULES-END -->\n"
    )


class TestSoulSyncBlocksRenderLast:
    """FLEET-RULES/DAMIEN-RULINGS/FLEET-STATE are sync-managed marker blocks
    inside SOUL.md, rewritten out-of-band by ops/soul_rules_sync.py and
    ops/rulings_soul_sync.py whenever a ruling or fleet-state fact changes —
    not on every run. Left inline, they sit inside the very first
    stable_parts entry, so a sync rewrite bursts the prompt-cache prefix for
    every static guidance block concatenated after them. Extracting them and
    re-appending after that guidance (system_prompt.py's
    ``_split_soul_sync_blocks`` / the ``_soul_sync_tail`` append) keeps the
    guidance ahead of them byte-identical across a sync rewrite."""

    def _stable_for_ruling(self, ruling_text: str) -> str:
        agent = _make_agent(_task_completion_guidance=True, valid_tool_names=["x"])
        with (
            patch("run_agent.load_soul_md", return_value=_soul_with_rules(ruling_text)),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            return build_system_prompt_parts(agent)["stable"]

    def test_static_guidance_prefix_is_byte_identical_across_a_sync_rewrite(self):
        stable_v1 = self._stable_for_ruling("RULE_V1_TEXT")
        stable_v2 = self._stable_for_ruling("RULE_V2_TEXT_DIFFERENT_LENGTH_TOO")

        # Locate each build's marker-block tail and strip it off.
        marker = "<!-- FLEET-RULES-BEGIN -->"
        prefix_v1 = stable_v1[: stable_v1.index(marker)]
        prefix_v2 = stable_v2[: stable_v2.index(marker)]

        # The prefix ahead of the marker block — SOUL identity text plus
        # every static guidance block (TASK_COMPLETION_GUIDANCE etc.) — is
        # byte-identical even though the ruling text inside the marker block
        # changed and changed length.
        assert prefix_v1 == prefix_v2
        # Sanity: static guidance actually rendered into that shared prefix.
        assert "Identity" in prefix_v1

        # Each build still carries its own (volatile) ruling text, at the
        # very end of the stable tier.
        assert stable_v1.endswith(
            "Consolidated fleet rules. RULE_V1_TEXT\n<!-- FLEET-RULES-END -->"
        )
        assert stable_v2.endswith(
            "Consolidated fleet rules. RULE_V2_TEXT_DIFFERENT_LENGTH_TOO\n"
            "<!-- FLEET-RULES-END -->"
        )

    def test_non_marker_soul_text_stays_in_place_ahead_of_static_guidance(self):
        # Hand-written SOUL.md text outside the marker block (a profile's own
        # persona notes) is not sync-managed and must not be relocated.
        stable = self._stable_for_ruling("RULE_V1_TEXT")
        assert stable.index(_SOUL_IDENTITY.split("\n")[1]) < stable.index(
            "<!-- FLEET-RULES-BEGIN -->"
        )


class TestNoCodingWorkspaceStablePrefixIgnoresEnvProbe:
    """PR #10 review (MAJOR): in the no-coding-workspace branch,
    ``post_workspace_parts`` used to alias ``stable_parts`` directly, so the
    environment-probe line (live Python/pip/PEP-668 state, sampled fresh on
    every build — see ``tools.env_probe.get_environment_probe_line``) landed
    inside the emitted "stable" prefix. Because that line can genuinely
    differ between two builds of the same recurring cron agent (a package
    got installed, PEP 668 state changed, ...), the stable prefix was not
    actually byte-identical across runs in this branch — the goal this
    whole PR exists for. It now always renders in the ``context`` tier
    instead (see ``_env_probe_parts`` in ``build_system_prompt_parts``).

    The active-profile and platform hints deliberately stay in the stable
    tier here: unlike the env-probe line, they are deterministic for a given
    agent/profile/platform config (see their own docstrings) — genuinely
    stable across repeat runs of the same recurring agent, not merely
    "usually the same" — so moving them would only widen the diff for no
    prefix-cache benefit, and several tests above pin their presence in the
    stable tier."""

    def _agent_and_soul(self):
        agent = _make_agent(_environment_probe=True)
        soul = (
            f"{_SOUL_IDENTITY}\n\n"
            "<!-- FLEET-RULES-BEGIN -->\n"
            "Consolidated fleet rules. RULE_TEXT\n"
            "<!-- FLEET-RULES-END -->\n"
        )
        return agent, soul

    def _build(self, probe_value: str):
        agent, soul = self._agent_and_soul()
        with (
            patch("run_agent.load_soul_md", return_value=soul),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
            patch("tools.env_probe.get_environment_probe_line", return_value=probe_value),
            patch("agent.file_safety._resolve_active_profile_name", return_value="default"),
        ):
            return build_system_prompt_parts(agent)

    def test_stable_prefix_before_sync_tail_is_byte_identical_across_env_probe_changes(self):
        parts_a = self._build("PEP668_MANAGED_ENV_v1")
        parts_b = self._build("PEP668_UNMANAGED_ENV_v2_LONGER_TEXT")

        marker = "<!-- FLEET-RULES-BEGIN -->"
        prefix_a = parts_a["stable"][: parts_a["stable"].index(marker)]
        prefix_b = parts_b["stable"][: parts_b["stable"].index(marker)]

        # The env-probe line changed between the two builds, yet the stable
        # prefix ahead of the sync tail — including everything the
        # no-coding-workspace branch used to fold the env-probe line into —
        # is byte-identical.
        assert prefix_a == prefix_b
        assert prefix_a  # sanity: not vacuously equal because both are empty

        # The env-probe line itself never appears in either build's stable
        # tier at all — it always renders in context now.
        assert "PEP668_MANAGED_ENV_v1" not in parts_a["stable"]
        assert "PEP668_UNMANAGED_ENV_v2_LONGER_TEXT" not in parts_b["stable"]
        assert "PEP668_MANAGED_ENV_v1" in parts_a["context"]
        assert "PEP668_UNMANAGED_ENV_v2_LONGER_TEXT" in parts_b["context"]

        # The sync tail is still the very last thing in the stable tier.
        assert parts_a["stable"].endswith(
            "Consolidated fleet rules. RULE_TEXT\n<!-- FLEET-RULES-END -->"
        )
        assert parts_b["stable"].endswith(
            "Consolidated fleet rules. RULE_TEXT\n<!-- FLEET-RULES-END -->"
        )

    def test_deterministic_profile_hint_stays_in_stable_tier(self):
        # Deterministic-per-config content (unlike the env-probe line) is
        # not relocated by this change.
        parts = self._build("PEP668_MANAGED_ENV_v1")
        assert "Active Hermes profile: default" in parts["stable"]
