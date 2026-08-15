from __future__ import annotations

import logging

import pytest
























def test_otlp_attrs_redact_strings_and_never_export_profile():
    from agent.monitoring.otlp_exporter import _span_attrs

    attrs = _span_attrs({
        "event": "gateway_health",
        "name": "gateway.lifecycle",
        "profile": "user@example.com",
        "exit_reason": "Bearer top-secret-token for user@example.com",
    })

    assert "hermes.profile" not in attrs
    assert "top-secret-token" not in str(attrs)
    assert "user@example.com" not in str(attrs)


def test_resource_attributes_are_allowlisted_and_sanitized():
    from agent.monitoring.gateway_health_export import _safe_resource_attributes

    attrs = _safe_resource_attributes({
        "service.name": "hermes-gateway",
        "service.instance.id": "install-1",
        "deployment.environment.name": "staging",
        "user.email": "user@example.com",
        "authorization": "Bearer top-secret-token",
        "custom.request.id": "unbounded",
    })

    assert attrs == {
        "service.name": "hermes-gateway",
        "service.instance.id": attrs["service.instance.id"],
        "deployment.environment.name": "staging",
    }
    assert attrs["service.instance.id"].startswith("sha256:")
    assert "install-1" not in attrs["service.instance.id"]






def test_diagnostic_log_attributes_are_allowlisted_redacted_and_profile_free():
    from agent.monitoring.gateway_health_export import _diagnostic_log_attributes

    attrs = _diagnostic_log_attributes({
        "event": "gateway_diagnostic",
        "name": "platform.fatal",
        "subsystem": "platform.slack",
        "profile": "user@example.com",
        "error_code": "Bearer top-secret-token",
        "custom": "must-not-egress",
    })

    assert "hermes.profile" not in attrs
    assert "hermes.custom" not in attrs
    assert "top-secret-token" not in str(attrs)




















def test_install_id_persists_across_calls(tmp_path, monkeypatch):
    """A minted install id must survive restarts (service.instance.id continuity)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n")

    import hermes_cli.config as cfg_mod
    from agent.monitoring.policy import ensure_install_id

    first = ensure_install_id(cfg_mod.load_config())
    assert first and first != "unknown"
    # Persisted: a fresh load (simulating a new gateway process) returns the same id.
    second = ensure_install_id(cfg_mod.load_config())
    assert second == first
    assert first in (tmp_path / "config.yaml").read_text()


def _enabled_export_config(endpoint: str) -> dict:
    return {
        "monitoring": {
            "gateway_health_export": {"enabled": True},
            "export": {"otlp": {"enabled": True, "endpoint": endpoint}},
        }
    }


def test_start_gateway_health_export_disables_when_collector_unreachable(
    monkeypatch, caplog
):
    """S101: unreachable collector disables the whole exporter with ONE warning.

    The single probe at telemetry init must stop every OTLP plane (metrics,
    spans, diagnostic logs) before any exporter is constructed — each plane
    retries failed exports with backoff, so proceeding would flood the
    gateway log with connection-refused retries.
    """
    from agent.monitoring import otlp_exporter
    from agent.monitoring.gateway_health_export import start_gateway_health_export

    caplog.set_level(logging.WARNING, logger="agent.monitoring.gateway_health_export")
    endpoint = "http://127.0.0.1:1"
    monkeypatch.setattr(otlp_exporter, "probe_collector", lambda endpoint, **kw: False)

    runtime = start_gateway_health_export(_enabled_export_config(endpoint))

    assert runtime.enabled is False
    assert runtime.reason == "collector_unreachable"
    records = [
        r
        for r in caplog.records
        if r.name == "agent.monitoring.gateway_health_export"
        and r.levelno == logging.WARNING
    ]
    assert len(records) == 1, "exactly one warning at exporter init"
    assert endpoint in records[0].getMessage()


def test_start_gateway_health_export_proceeds_when_collector_reachable(
    monkeypatch,
):
    """S101: a reachable collector leaves the init path unchanged."""
    from agent.monitoring import otlp_exporter
    import agent.monitoring.gateway_health_export as GHE

    endpoint = "http://127.0.0.1:1"
    monkeypatch.setattr(otlp_exporter, "probe_collector", lambda endpoint, **kw: True)

    def _sdk_unavailable(*args, **kwargs):
        raise RuntimeError("sdk unavailable")

    monkeypatch.setattr(GHE, "_require_metrics_sdk", _sdk_unavailable)

    runtime = GHE.start_gateway_health_export(_enabled_export_config(endpoint))

    # Passed the probe and took the normal path: the SDK availability check
    # was reached (and failed only because the test stubbed it).
    assert runtime.enabled is False
    assert runtime.reason == "otlp_unavailable"


