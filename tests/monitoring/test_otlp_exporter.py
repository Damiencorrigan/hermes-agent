"""OTLP exporter tests: config resolution, span mapping, streaming subscriber.

No SQLite involved — monitoring is an egress path, so the exporter consumes
emitter batches directly. Uses the in-memory OTel span exporter; skipped when
the optional otlp extra is not installed.
"""

from __future__ import annotations

import logging

import pytest

otel = pytest.importorskip("opentelemetry.sdk.trace", reason="otlp extra not installed")

import agent.monitoring.otlp_exporter as OE
from agent.monitoring.emitter import MonitoringEmitter


def _mem_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_gateway_health_event_maps_to_span_with_attrs():
    provider, mem = _mem_provider()
    n = OE.export_batch(provider, [{
        "event": "gateway_health", "name": "gateway.lifecycle",
        "old_state": "starting", "new_state": "running",
        "active_agents": 2, "pid": 4242,
    }])
    assert n == 1
    spans = mem.get_finished_spans()
    assert spans[0].name == "hermes.gateway_health"
    attrs = dict(spans[0].attributes or {})
    assert attrs["hermes.old_state"] == "starting"
    assert attrs["hermes.new_state"] == "running"
    assert attrs["hermes.active_agents"] == 2






def test_headers_resolve_from_env_not_value(monkeypatch):
    monkeypatch.setenv("DD_KEY_ENV", "secret-value")
    resolved = OE._resolve_headers({"DD-API-KEY": "DD_KEY_ENV", "X-Missing": "NOPE_ENV"})
    assert resolved == {"DD-API-KEY": "secret-value"}




def test_trace_resource_includes_stable_hashed_instance():
    attrs = OE._resource_attributes(
        {"monitoring": {"install_id": "private-install-id"}}
    )

    assert attrs["service.name"] == "hermes-gateway"
    assert attrs["service.instance.id"].startswith("sha256:")
    assert len(attrs["service.instance.id"]) == len("sha256:") + 24
    assert "private-install-id" not in str(attrs)
    assert attrs["telemetry.scope"] == "gateway_monitoring"


def test_trace_resource_includes_configured_deployment_environment():
    attrs = OE._resource_attributes({
        "monitoring": {
            "install_id": "private-install-id",
            "gateway_health_export": {
                "resource_attributes": {"deployment.environment.name": "production"},
            },
        },
    })

    assert attrs["deployment.environment.name"] == "production"
    assert attrs["service.name"] == "hermes-gateway"




def test_streamer_receives_events_and_respects_filter(monkeypatch):
    provider, mem = _mem_provider()
    monkeypatch.setattr(OE, "_make_provider", lambda cfg: (provider, None))
    streamer = OE.OTLPStreamer(
        {}, event_filter=lambda ev: ev.get("event") == "gateway_health")

    em = MonitoringEmitter()
    em.subscribe(streamer)
    em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    em.emit({"event": "model_call", "provider": "anthropic"})  # filtered out
    em.flush()
    em.close()

    spans = mem.get_finished_spans()
    assert [s.name for s in spans] == ["hermes.gateway_health"]
    assert streamer.exported == 1


def test_failing_streamer_never_breaks_emitter(monkeypatch):
    def boom(cfg):
        raise RuntimeError("no provider")

    em = MonitoringEmitter()

    def bad_subscriber(batch):
        raise RuntimeError("export down")

    seen: list = []
    em.subscribe(bad_subscriber)
    em.subscribe(lambda batch: seen.extend(batch))
    em.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
    em.flush()
    em.close()
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# S101: collector reachability gate at telemetry init.
# ---------------------------------------------------------------------------


def _otlp_enabled_config(endpoint: str) -> dict:
    return {
        "monitoring": {
            "export": {"otlp": {"enabled": True, "endpoint": endpoint}},
        }
    }


def test_start_streaming_disables_exporter_when_collector_unreachable(caplog):
    """A configured-but-down collector disables the exporter with ONE warning.

    Regression (S101): the OTLP HTTP exporter retries failed exports with
    backoff, so an unset collector at localhost:3000 produced 27,591
    connection-refused retries in the gateway log.  Port 1 refuses the TCP
    probe instantly; the exporter must not be constructed.
    """
    caplog.set_level(logging.WARNING, logger="agent.monitoring.otlp_exporter")
    endpoint = "http://127.0.0.1:1"
    assert OE.start_streaming(_otlp_enabled_config(endpoint)) is None

    records = [
        r
        for r in caplog.records
        if r.name == "agent.monitoring.otlp_exporter"
        and r.levelno == logging.WARNING
    ]
    assert len(records) == 1, "exactly one warning at exporter init"
    assert endpoint in records[0].getMessage()


def test_start_streaming_unchanged_when_collector_reachable(
    monkeypatch, caplog
):
    """A reachable collector leaves exporter init exactly as before."""
    caplog.set_level(logging.WARNING, logger="agent.monitoring.otlp_exporter")
    endpoint = "http://127.0.0.1:1"
    monkeypatch.setattr(OE, "probe_collector", lambda endpoint, **kw: True)

    class _FakeStreamer:
        def __init__(self, cfg, *, event_filter=None):
            self.cfg = cfg
            self.event_filter = event_filter

        def shutdown(self):
            pass

    monkeypatch.setattr(OE, "OTLPStreamer", _FakeStreamer)
    streamer = OE.start_streaming(
        _otlp_enabled_config(endpoint), event_filter=lambda ev: True
    )
    assert isinstance(streamer, _FakeStreamer), "streamer configured as before"
    assert streamer.event_filter is not None

    records = [
        r
        for r in caplog.records
        if r.name == "agent.monitoring.otlp_exporter"
        and r.levelno == logging.WARNING
    ]
    assert records == [], "no warning on the healthy path"


def test_probe_collector_true_when_endpoint_listening():
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    port = srv.getsockname()[1]
    try:
        assert OE.probe_collector(f"http://127.0.0.1:{port}") is True
        # Drain the accept queue so the next connect is not dropped.
        conn, _ = srv.accept()
        conn.close()
        # Scheme-less endpoints (as configured for self-hosted langfuse) probe too.
        assert OE.probe_collector(f"127.0.0.1:{port}") is True
    finally:
        srv.close()


def test_probe_collector_false_on_refused_and_malformed():
    assert OE.probe_collector("http://127.0.0.1:1") is False  # refused
    assert OE.probe_collector("") is False
    assert OE.probe_collector("file:///tmp/x") is False
    assert OE.probe_collector("not a url") is False
