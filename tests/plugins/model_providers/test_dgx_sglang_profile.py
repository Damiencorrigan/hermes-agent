"""Unit tests for the ``dgx-sglang`` provider profile's reasoning-effort clamp.

Regression coverage for the 2026-08-26 clamp trap (WS-T bench,
``docs/research/2026-08-26-qwen38-agentic-bench.md`` in ai-fleet): the SGLang
build running on the DGX canary (``RadixArk/Qwen3.8-27B-NVFP4``,
``--reasoning-parser qwen3``) only accepts ``reasoning_effort`` values
``none``, ``low``, ``medium``, ``xhigh`` — a literal ``high`` (valid in
Hermes' own ``VALID_REASONING_EFFORTS`` tuple) gets a hard HTTP 400 from the
server: "Unexpected reasoning effort high. Supported types are xhigh
(default), medium, and low." Before this profile was registered, no code
path clamped or even forwarded ``agent.reasoning_effort`` for this provider
at all — see the module docstring in
``plugins/model-providers/dgx-sglang/__init__.py`` for the full wiring gap.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def dgx_sglang_profile():
    """Resolve the registered dgx-sglang profile via the global registry.

    Going through ``get_provider_profile`` (rather than importing the class
    directly) keeps the test honest — it fails if the plugin ever stops
    registering under the ``dgx-sglang`` name every profile's
    ``model.provider: dgx-sglang`` actually uses.
    """
    import model_tools  # noqa: F401 — triggers plugin discovery
    import providers

    profile = providers.get_provider_profile("dgx-sglang")
    assert profile is not None, "dgx-sglang provider profile must be registered"
    return profile


class TestSglangEffortClamp:
    """``clamp_sglang_reasoning_effort`` — the mapping table itself.

    Loaded directly from the plugin file (not a normal dotted-package import
    — the plugin directory name contains dashes) via the same
    ``importlib.util`` spec-loading approach ``providers._discover_providers``
    itself uses to load plugin directories.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def clamp_fn():
        import importlib.util
        import pathlib

        plugin_path = (
            pathlib.Path(__file__).resolve().parents[3]
            / "plugins"
            / "model-providers"
            / "dgx-sglang"
            / "__init__.py"
        )
        spec = importlib.util.spec_from_file_location(
            "dgx_sglang_provider_plugin_under_test", plugin_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.clamp_sglang_reasoning_effort

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("none", "none"),
            ("low", "low"),
            ("medium", "medium"),
            ("xhigh", "xhigh"),
            ("minimal", "low"),
            ("high", "xhigh"),  # the trap this fix closes
            ("max", "xhigh"),
            ("ultra", "xhigh"),
            ("HIGH", "xhigh"),  # case-insensitive
            (" high ", "xhigh"),  # whitespace-tolerant
            ("some-future-level", "xhigh"),  # unknown → safe default, not passthrough
        ],
    )
    def test_clamp_table(self, clamp_fn, raw, expected):
        assert clamp_fn(raw) == expected


class TestDgxSglangReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, dgx_sglang_profile):
        """Unset reasoning → omit everything, server default (xhigh, thinking on) applies."""
        eb, tl = dgx_sglang_profile.build_api_kwargs_extras(
            reasoning_config=None, model="RadixArk/Qwen3.8-27B-NVFP4"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_enable_thinking_false_and_reasoning_effort_none(
        self, dgx_sglang_profile
    ):
        """enabled=False → both enable_thinking:false AND reasoning_effort:none.

        WS-T bench: enable_thinking:false is the fastest, 100%-reliable
        setting for agent/tool-use turns (2.1s single-call, never lost a
        call across 30 calls). reasoning_effort:"none" is this build's own
        accepted value for "no reasoning" and is sent too, belt-and-braces.
        """
        eb, tl = dgx_sglang_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="RadixArk/Qwen3.8-27B-NVFP4"
        )
        assert eb == {"chat_template_kwargs": {"enable_thinking": False}}
        assert tl == {"reasoning_effort": "none"}

    def test_effort_none_disables_same_as_enabled_false(self, dgx_sglang_profile):
        eb, tl = dgx_sglang_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="RadixArk/Qwen3.8-27B-NVFP4",
        )
        assert eb == {"chat_template_kwargs": {"enable_thinking": False}}
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "effort,expected_wire_effort",
        [
            ("low", "low"),
            ("medium", "medium"),
            ("xhigh", "xhigh"),
            ("minimal", "low"),
            ("high", "xhigh"),  # THE TRAP: server 400s on literal "high"
            ("max", "xhigh"),
            ("ultra", "xhigh"),
        ],
    )
    def test_enabled_effort_is_clamped_top_level(
        self, dgx_sglang_profile, effort, expected_wire_effort
    ):
        """enabled + effort → TOP-LEVEL reasoning_effort, clamped to this
        SGLang build's accepted set. Never passed through raw when raw would
        400 (the pre-fix behavior for "high")."""
        eb, tl = dgx_sglang_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="RadixArk/Qwen3.8-27B-NVFP4",
        )
        assert tl == {"reasoning_effort": expected_wire_effort}
        assert "chat_template_kwargs" not in eb

    def test_never_emits_the_rejected_high_value(self, dgx_sglang_profile):
        """No input should ever produce a literal 'high' on the wire — that
        is exactly the value this SGLang build 400s on."""
        for effort in ("high", "HIGH", " High "):
            _, tl = dgx_sglang_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                model="RadixArk/Qwen3.8-27B-NVFP4",
            )
            assert tl.get("reasoning_effort") != "high"
