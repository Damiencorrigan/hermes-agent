"""Regression tests for custom_providers per-provider max_output_tokens resolution.

Covers the fix for providers.<key>.max_output_tokens being silently accepted
into config.yaml but never read: the normalizer warned "unknown config keys
ignored" and the value never reached AIAgent.max_tokens, so every custom/
Ollama-backed provider was stuck on the shared "custom" profile's static
default_max_tokens (4096) regardless of what a profile configured.
"""
from __future__ import annotations

import logging

import pytest

from hermes_cli.config import (
    _PROVIDER_NORMALIZE_WARNED,
    _normalize_custom_provider_entry,
    get_custom_provider_max_output_tokens,
)


class TestNormalizeMaxOutputTokens:
    @pytest.fixture(autouse=True)
    def _reset_warn_cache(self):
        _PROVIDER_NORMALIZE_WARNED.clear()
        yield
        _PROVIDER_NORMALIZE_WARNED.clear()

    def test_max_output_tokens_is_known_not_warned(self, caplog):
        entry = {
            "api": "http://192.168.0.214:11434/v1",
            "max_output_tokens": 2048,
        }
        with caplog.at_level(logging.WARNING):
            result = _normalize_custom_provider_entry(entry, provider_key="dgx-ollama")
        unknown_warnings = [
            r for r in caplog.records
            if "unknown config keys" in r.message.lower()
        ]
        assert not unknown_warnings
        assert result["max_output_tokens"] == 2048

    def test_non_positive_or_non_numeric_is_dropped(self):
        for bad in (0, -1, "2048", True, None):
            entry = {"api": "http://x/v1", "max_output_tokens": bad}
            result = _normalize_custom_provider_entry(entry, provider_key="p")
            assert "max_output_tokens" not in result

    def test_float_is_coerced_to_int(self):
        entry = {"api": "http://x/v1", "max_output_tokens": 2048.0}
        result = _normalize_custom_provider_entry(entry, provider_key="p")
        assert result["max_output_tokens"] == 2048
        assert isinstance(result["max_output_tokens"], int)


class TestGetCustomProviderMaxOutputTokens:
    def test_resolves_override_for_matching_route(self):
        custom = [
            {
                "base_url": "http://192.168.0.214:11434/v1",
                "max_output_tokens": 2048,
            }
        ]
        assert get_custom_provider_max_output_tokens(
            "http://192.168.0.214:11434/v1", custom
        ) == 2048

    def test_trailing_slash_insensitive(self):
        custom = [
            {"base_url": "https://example.invalid/v1/", "max_output_tokens": 8192}
        ]
        assert get_custom_provider_max_output_tokens(
            "https://example.invalid/v1", custom
        ) == 8192

    def test_no_override_returns_none(self):
        custom = [{"base_url": "https://example.invalid/v1"}]
        assert get_custom_provider_max_output_tokens(
            "https://example.invalid/v1", custom
        ) is None

    def test_route_isolated(self):
        """A max_output_tokens set for one provider's route must not leak to another."""
        custom = [
            {"base_url": "https://other.invalid/v1", "max_output_tokens": 2048}
        ]
        assert get_custom_provider_max_output_tokens(
            "https://example.invalid/v1", custom
        ) is None

    def test_empty_inputs_return_none(self):
        assert get_custom_provider_max_output_tokens("", [{"base_url": "http://x", "max_output_tokens": 1}]) is None
        assert get_custom_provider_max_output_tokens("http://x", None) is None
        assert get_custom_provider_max_output_tokens("http://x", []) is None
