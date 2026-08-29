"""Durable cron execution-ledger behavior."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_execution_transitions_are_durable(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    claimed = executions.create_execution("job-1", source="builtin")
    assert claimed["status"] == "claimed"
    assert claimed["claimed_at"]
    assert claimed["started_at"] is None
    assert claimed["finished_at"] is None

    running = executions.mark_execution_running(claimed["id"])
    assert running["status"] == "running"
    assert running["started_at"]

    completed = executions.finish_execution(claimed["id"], success=True)
    assert completed["status"] == "completed"
    assert completed["finished_at"]
    assert completed["error"] is None

    persisted = executions.list_executions(job_id="job-1")
    assert persisted == [completed]


def test_terminal_execution_cannot_be_rewritten(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)

    assert executions.finish_execution(
        record["id"], success=False, error="late writer"
    ) is None
    assert executions.latest_execution("immutable")["status"] == "completed"


def test_retention_bounds_terminal_history_but_preserves_inflight(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 3)
    inflight = executions.create_execution("live", source="builtin")
    executions.mark_execution_running(inflight["id"])
    for index in range(8):
        row = executions.create_execution(f"done-{index}", source="builtin")
        executions.finish_execution(row["id"], success=True)

    records = executions.list_executions(limit=100)
    assert len([row for row in records if row["status"] == "completed"]) == 3
    assert executions.latest_execution("live")["status"] == "running"


def test_corrupt_store_fails_closed_without_overwrite(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    executions.EXECUTIONS_FILE.write_bytes(b"not a sqlite database")

    with __import__("pytest").raises(sqlite3.DatabaseError):
        executions.create_execution("new", source="builtin")
    assert executions.EXECUTIONS_FILE.read_bytes() == b"not a sqlite database"


def test_cron_runs_cli_prints_execution_history(monkeypatch, tmp_path, capsys):
    executions = _point_ledger(monkeypatch, tmp_path)
    row = executions.create_execution("cli-job", source="builtin")
    executions.finish_execution(row["id"], success=False, error="boom")
    from hermes_cli.cron import cron_runs

    cron_runs("cli-job", limit=10)

    output = capsys.readouterr().out
    assert row["id"] in output
    assert "failed" in output
    assert "boom" in output


def test_quick_backup_includes_execution_ledger():
    from hermes_cli.backup import _QUICK_STATE_FILES

    assert "cron/executions.db" in _QUICK_STATE_FILES


def test_failed_execution_keeps_error(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    record = executions.create_execution("job-2", source="external")
    failed = executions.finish_execution(record["id"], success=False, error="provider exploded")

    assert failed["status"] == "failed"
    assert failed["error"] == "provider exploded"


def test_recovery_does_not_mark_live_process_execution_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("still-live", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution("still-live")["status"] == "running"


def test_restart_marks_interrupted_execution_unknown_without_requeue(tmp_path):
    """Real temp-HERMES_HOME subprocess restart: in-flight is audit-only unknown."""
    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    create = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cron.executions import create_execution, mark_execution_running; "
            "r=create_execution('restart-job', source='builtin'); "
            "mark_execution_running(r['id']); print(r['id'])",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    execution_id = create.stdout.strip()

    recover = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from cron.executions import recover_interrupted_executions, list_executions; "
            "print(recover_interrupted_executions()); "
            "print(json.dumps(list_executions(job_id='restart-job'))) ",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = recover.stdout.strip().splitlines()
    assert lines[0] == "1"
    records = json.loads(lines[1])
    assert len(records) == 1
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "unknown"
    assert records[0]["finished_at"]
    assert "restart" in records[0]["error"].lower()
    # Recovery only classifies the old attempt. It must not manufacture a new
    # claimed record (which would imply an automatic retry).
    assert [r["status"] for r in records] == ["unknown"]


def test_generic_submit_failure_finishes_attempt_and_releases_guard(monkeypatch):
    import cron.scheduler as scheduler

    class BrokenPool:
        def submit(self, _callable):
            raise ValueError("executor rejected")

    finished = []
    monkeypatch.setattr(
        scheduler, "create_execution",
        lambda *_args, **_kwargs: {"id": "exec-submit-fail"},
    )
    monkeypatch.setattr(
        scheduler, "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [{"id": "submit-fail"}])
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: 0)
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: BrokenPool())

    assert scheduler.tick(verbose=False, sync=False) == 0
    assert finished == [
        ("exec-submit-fail", {
            "success": False,
            "error": "Executor dispatch failed: executor rejected",
        })
    ]
    assert "submit-fail" not in scheduler.get_running_job_ids()


def test_run_one_job_records_running_then_terminal(monkeypatch):
    import cron.scheduler as scheduler

    events = []
    monkeypatch.setattr(
        scheduler,
        "mark_execution_running",
        lambda execution_id: events.append(("running", execution_id)),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: events.append(("finish", execution_id, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda job, *, defer_agent_teardown=None, **_kw: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job({"id": "job-3", "execution_id": "exec-3"}) is True
    assert events[0] == ("running", "exec-3")
    assert events[-1][0:2] == ("finish", "exec-3")
    assert events[-1][2]["success"] is True


def test_provider_start_recovers_interrupted_records_before_tick(monkeypatch):
    import cron.scheduler_provider as provider

    events = []
    stop = __import__("threading").Event()
    stop.set()
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
        raising=False,
    )
    monkeypatch.setattr("cron.jobs.record_ticker_heartbeat", lambda **_kwargs: events.append("heartbeat"))

    provider.InProcessCronScheduler().start(stop, interval=1)

    assert events[:2] == ["recover", "heartbeat"]


def test_external_provider_start_recovers_interrupted_records(monkeypatch):
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    provider._client = type("Client", (), {"arm": lambda self, **kwargs: None})()
    events = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
    )
    monkeypatch.setattr(provider, "reconcile", lambda: events.append("reconcile"))

    provider.start(__import__("threading").Event())

    assert events == ["recover", "reconcile"]


class _TrackingConnection:
    """Delegates to a real sqlite3.Connection while recording close() calls.

    sqlite3.Connection is a static C type: it has no per-instance __dict__
    and its class methods can't be monkeypatched, so open/close tracking is
    done via a delegating wrapper returned in place of the real connection.
    """

    def __init__(self, real, closed_ids):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed_ids", closed_ids)

    def close(self):
        self._closed_ids.append(id(self._real))
        self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _count_open_connections(executions, monkeypatch):
    """Wrap sqlite3.connect to track open/close balance for the ledger module."""
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _TrackingConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)
    return opened_ids, closed_ids


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Regression for #69567: every ledger call must close its connection
    deterministically instead of relying on garbage collection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    record = executions.create_execution("leak-check", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)
    executions.list_executions(job_id="leak-check")
    executions.latest_executions(["leak-check"])
    executions.recover_interrupted_executions()

    assert len(opened) == 6
    assert len(closed) == 6
    assert set(opened) == set(closed)


def test_early_return_still_closes_connection(monkeypatch, tmp_path):
    """mark_execution_running returns None mid-block on a bad transition;
    the connection must still be closed rather than leaked."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    assert executions.mark_execution_running("does-not-exist") is None

    assert len(opened) == 1
    assert len(closed) == 1


def test_exception_during_operation_still_closes_connection(monkeypatch, tmp_path):
    """A failing statement inside the transaction must roll back and close,
    not leak the connection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    with __import__("pytest").raises(sqlite3.IntegrityError):
        with executions._transaction() as conn:
            conn.execute(
                "INSERT INTO executions (id, job_id, source, process_id, pid, "
                "status, claimed_at) VALUES ('x', 'x', 'x', 'x', 1, 'bogus-status', 'now')"
            )

    assert len(opened) == 1
    assert len(closed) == 1


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """If PRAGMA/DDL setup in _connect() fails after sqlite3.connect()
    succeeds, the partially-initialized connection must still be closed."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _FailingSchemaConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)

    with __import__("pytest").raises(sqlite3.OperationalError):
        executions.create_execution("init-fail", source="builtin")

    assert len(opened_ids) == 1
    assert len(closed_ids) == 1


def test_job_listing_exposes_latest_execution(monkeypatch, tmp_path):
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    executions = _point_ledger(monkeypatch, tmp_path)

    job = jobs.create_job(prompt="audit me", schedule="every 1h", name="audit")
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])

    listed = jobs.list_jobs(include_disabled=True)
    assert listed[0]["latest_execution"]["id"] == record["id"]
    assert listed[0]["latest_execution"]["status"] == "running"


class TestExecutionsFileMultiplexScoping:
    """Root-caused 2026-08-24 investigating a live ticker freeze: EXECUTIONS_FILE
    was a plain module-level constant computed ONCE at import time from
    get_hermes_home(), with no use_cron_store()/set_hermes_home_override()
    awareness at all -- unlike jobs.json (cron.jobs._current_cron_store()).

    Effect under a multiplexed gateway (scheduler_provider._start_multiplex
    scopes each profile's tick via use_cron_store(profile_home)): every
    create_execution()/recover_interrupted_executions() call during a
    profile-scoped tick silently wrote to the DEFAULT/root profile's
    executions.db instead of the calling profile's own -- so a profile's
    dead-owner claimed/running rows could never be seen or reconciled by the
    per-tick recovery pass (#60432 follow-up), even though that pass ran
    without error every tick. Found via commander/trove/tva each carrying
    real dead-pid rows in their OWN executions.db that recover_interrupted()
    never touched.
    """

    def test_create_execution_scoped_to_use_cron_store_home(self, tmp_path):
        import cron.executions as executions
        import cron.jobs as jobs

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "commander"
        default_home.mkdir(parents=True)
        profile_home.mkdir(parents=True)

        with jobs.use_cron_store(profile_home):
            record = executions.create_execution("job-scoped", source="builtin")

        # The profile's OWN ledger carries the row.
        with jobs.use_cron_store(profile_home):
            rows = executions.list_executions(job_id="job-scoped")
        assert len(rows) == 1
        assert rows[0]["id"] == record["id"]

        # The default store was never touched -- no orphaned phantom entry,
        # exactly the corruption the pre-fix behavior produced.
        with jobs.use_cron_store(default_home):
            assert executions.list_executions(job_id="job-scoped") == []

    def test_recover_interrupted_executions_scoped_to_use_cron_store_home(
        self, tmp_path, monkeypatch
    ):
        """The per-tick dead-pid reconciliation (InProcessCronScheduler.
        recover_interrupted -> recover_interrupted_executions) must clean the
        CALLING profile's own ledger, not silently no-op against an unrelated
        (already-clean) default store."""
        import cron.executions as executions
        import cron.jobs as jobs

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "trove"
        default_home.mkdir(parents=True)
        profile_home.mkdir(parents=True)

        with jobs.use_cron_store(profile_home):
            record = executions.create_execution("job-dead", source="builtin")
            # Simulate a dead owner from a DIFFERENT process generation: a
            # different process_id (recover_interrupted_executions() skips
            # rows matching this module instance's own _PROCESS_ID as "not
            # abandoned, that's me") and a pid/start-time pair that can never
            # match anything alive (pid 1 with a start time no live pid=1
            # process on this host will ever report).
            with executions._transaction() as conn:
                conn.execute(
                    "UPDATE executions SET process_id=?, pid=1, "
                    "process_started_at=1 WHERE id=?",
                    ("a-different-dead-process-id", record["id"]),
                )

            recovered = executions.recover_interrupted_executions()
            assert recovered == 1

            updated = executions.list_executions(job_id="job-dead")[0]
            assert updated["status"] == "unknown"

        # Never touched the default store.
        with jobs.use_cron_store(default_home):
            assert executions.list_executions(job_id="job-dead") == []

    def test_reverts_to_default_after_use_cron_store_exits(self, tmp_path):
        import cron.executions as executions
        import cron.jobs as jobs

        profile_home = tmp_path / "profiles" / "scoped"
        profile_home.mkdir(parents=True)

        before = executions._resolve_executions_file()
        with jobs.use_cron_store(profile_home):
            assert executions._resolve_executions_file() == (
                profile_home.resolve() / "cron" / "executions.db"
            )
        after = executions._resolve_executions_file()
        assert after == before

    def test_monkeypatched_executions_file_still_honored_with_no_override(
        self, monkeypatch, tmp_path
    ):
        """Backward compatibility: the documented test/embedder surface of
        directly re-pointing executions.EXECUTIONS_FILE (no use_cron_store()
        active) must keep working exactly as before this fix."""
        import cron.executions as executions

        custom = tmp_path / "custom" / "executions.db"
        monkeypatch.setattr(executions, "EXECUTIONS_FILE", custom)

        assert executions._resolve_executions_file() == custom
        record = executions.create_execution("job-compat", source="builtin")
        assert custom.exists()
        assert executions.list_executions(job_id="job-compat")[0]["id"] == record["id"]


def test_1001st_execution_is_retained_and_oldest_is_evicted(monkeypatch, tmp_path):
    """The retention cap is a rotating window, not a hard stop.

    Regression for the 2026-08-26 incident: executions.db pinned at exactly
    1000 terminal rows and recording stopped. After this test, the 1001st
    execution is recorded and the OLDEST terminal row is evicted instead of
    the ledger refusing new records.
    """
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1000)

    oldest = executions.create_execution("oldest-job", source="builtin")
    executions.finish_execution(oldest["id"], success=True)

    for index in range(999):
        row = executions.create_execution(f"filler-{index}", source="builtin")
        executions.finish_execution(row["id"], success=True)

    # The 1001st execution: full claimed -> running -> completed lifecycle.
    claim = executions.create_execution("number-1001", source="builtin")
    executions.mark_execution_running(claim["id"])
    done = executions.finish_execution(claim["id"], success=True)
    assert done["status"] == "completed"

    conn = sqlite3.connect(executions.EXECUTIONS_FILE)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, status FROM executions").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1000
    ids = {row["id"] for row in rows}
    assert claim["id"] in ids  # 1001st retained
    assert oldest["id"] not in ids  # oldest evicted


def test_create_path_self_heals_over_cap_store(monkeypatch, tmp_path):
    """A store left over the cap by an older writer trims on the next write;
    create_execution never refuses to record at the cap."""
    executions = _point_ledger(monkeypatch, tmp_path)

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 10**9)
    for index in range(1050):
        row = executions.create_execution(f"seed-{index}", source="builtin")
        executions.finish_execution(row["id"], success=True)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1000)

    claim = executions.create_execution("resumed", source="builtin")
    assert claim["status"] == "claimed"

    conn = sqlite3.connect(executions.EXECUTIONS_FILE)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, job_id, status FROM executions"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1001  # 1000 terminal + the new claimed attempt
    ids = {row["id"] for row in rows}
    job_ids = {row["job_id"] for row in rows}
    assert claim["id"] in ids
    assert "resumed" in job_ids
    terminal = [
        row for row in rows if row["status"] in ("completed", "failed", "unknown")
    ]
    assert len(terminal) == 1000
    # The 50 oldest seeded rows were evicted on the resume write.
    assert all(f"seed-{index}" not in job_ids for index in range(50))
