"""Tests for #60432: cron jobs must not be silently invisible to gateway
shutdown, and a job whose tool subprocess got killed by shutdown must
never be reported as a successful run.

Covers the cron/scheduler.py primitives directly:
  - get_running_job_ids() -- thread-safe snapshot the gateway drain reads
  - mark_running_jobs_interrupted() -- called by the gateway right after
    it force-kills tool subprocesses
  - the interrupted-flag race guard in run_one_job(), which must win over
    the job's own thread finishing normally with a plausible-looking
    result AFTER its tool was already killed out from under it
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Every test starts from a clean slate and leaves one behind, since
    these sets are module-level globals shared across the test process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_job_homes.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_job_homes.clear()
    sched._interrupted_job_ids.clear()


class TestGetRunningJobIds:
    def test_empty_when_nothing_running(self):
        import cron.scheduler as sched

        assert sched.get_running_job_ids() == frozenset()

    def test_reflects_in_flight_jobs(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        sched._running_job_ids.add("job-2")

        result = sched.get_running_job_ids()

        assert result == frozenset({"job-1", "job-2"})

    def test_snapshot_is_immutable_and_independent(self):
        """Mutating _running_job_ids after the call must not change the
        already-returned snapshot -- callers (the gateway drain loop) rely
        on this to safely count in a tight polling loop."""
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        snapshot = sched.get_running_job_ids()
        sched._running_job_ids.add("job-2")

        assert snapshot == frozenset({"job-1"})


class TestMarkRunningJobsInterrupted:
    def test_no_op_when_nothing_running(self):
        import cron.scheduler as sched

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == []
        mock_mark.assert_not_called()

    def test_marks_every_in_flight_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("gateway shutdown (final-cleanup)")

        assert sorted(marked) == ["job-1", "job-2"]
        assert mock_mark.call_count == 2
        called_ids = {c.args[0] for c in mock_mark.call_args_list}
        assert called_ids == {"job-1", "job-2"}
        for c in mock_mark.call_args_list:
            # success must be False -- an interrupted run is never "ok".
            assert c.args[1] is False
            assert "gateway shutdown" in c.args[2]

    def test_sets_interrupted_flag_for_consumption_by_run_one_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")

        with patch("cron.scheduler.mark_job_run"):
            sched.mark_running_jobs_interrupted("shutdown")

        assert "job-1" in sched._interrupted_job_ids

    def test_one_job_marking_failure_does_not_block_the_others(self):
        """mark_job_run raising for one job (e.g. a jobs.json write race)
        must not prevent the rest from being marked -- this runs during
        shutdown, there's no retry window."""
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        def _side_effect(job_id, success, reason, **kwargs):
            if job_id == "job-1":
                raise OSError("disk full")

        with patch("cron.scheduler.mark_job_run", side_effect=_side_effect):
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == ["job-2"]


class TestIsInterrupted:
    """Peek-only check used at the delivery gate -- must NOT clear the
    flag, unlike _consume_interrupted_flag."""

    def test_false_when_not_marked(self):
        import cron.scheduler as sched

        assert sched._is_interrupted("job-1") is False

    def test_true_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._is_interrupted("job-1") is True

    def test_does_not_clear_the_flag(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        sched._is_interrupted("job-1")

        # Still set -- the later, authoritative check before mark_job_run
        # must still see it.
        assert "job-1" in sched._interrupted_job_ids
        assert sched._is_interrupted("job-1") is True


class TestConsumeInterruptedFlag:

    def test_true_and_clears_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._consume_interrupted_flag("job-1") is True
        # Consumed -- a second check (e.g. a later, unrelated fire of the
        # same recurring job ID) must not still read as interrupted.
        assert sched._consume_interrupted_flag("job-1") is False


class TestRunOneJobHonoursInterruptedFlag:
    """run_one_job() must not let a job's own completion overwrite a
    status the shutdown path already wrote for the same run."""

    def _make_job(self, job_id="job-1"):
        return {"id": job_id, "name": "test job", "prompt": "do work"}

    def test_success_path_skipped_when_interrupted(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        # The would-be "success" write must NOT happen -- the shutdown
        # path already wrote the authoritative interrupted status.
        mock_mark.assert_not_called()
        # Flag is consumed so a later, unrelated fire of the same job ID
        # isn't permanently silenced.
        assert job["id"] not in sched._interrupted_job_ids

    def test_interrupted_job_delivers_failure_summary_not_raw_response(self):
        """The status-write guard alone isn't enough: delivery happens
        BEFORE mark_job_run in run_one_job's own flow, so a job that kept
        running post-kill and produced a plausible-looking final_response
        must not have that response sent to the user just because the
        eventual status write gets suppressed. Interrupted jobs must route
        through the same failure-summary delivery path a real failure
        would."""
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "a plausible final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch(
                 "cron.scheduler._summarize_cron_failure_for_delivery",
                 return_value="This run was interrupted.",
             ) as mock_summarize, \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None) as mock_deliver, \
             patch("cron.scheduler.mark_job_run"):
            result = sched.run_one_job(job)

        assert result is True
        mock_summarize.assert_called_once()
        # The summarizer's error argument must mention the interruption,
        # not be silently None / the agent's own (possibly absent) error.
        assert "interrupt" in mock_summarize.call_args.args[1].lower()
        delivered_content = mock_deliver.call_args.args[1]
        assert delivered_content == "This run was interrupted."
        assert "plausible final response" not in delivered_content


    def test_exception_path_also_honours_interrupted_flag(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("cron.scheduler.run_job", side_effect=RuntimeError("boom")), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is False
        mock_mark.assert_not_called()


class TestRunningJobHomeTracking:
    """#60432 follow-up (2026-08-24): try_register_running_job snapshots the
    ACTIVE profile home (the same one a normal tick scopes itself to via
    set_hermes_home_override()/use_cron_store() -- see
    scheduler_provider._start_multiplex) so a later, separate-thread
    interrupt-mark can route back to the right store."""

    def test_captures_active_home_on_register(self, tmp_path):
        import cron.scheduler as sched
        import hermes_constants

        home = tmp_path / "profiles" / "trove"
        home.mkdir(parents=True)

        token = hermes_constants.set_hermes_home_override(str(home))
        try:
            assert sched.try_register_running_job("job-x") is True
        finally:
            hermes_constants.reset_hermes_home_override(token)

        assert sched._running_job_homes["job-x"] == str(home.resolve())

    def test_captures_ambient_home_when_no_override_active(self, tmp_path, monkeypatch):
        """Single-profile gateway (no multiplex override): still captures
        SOMETHING sane (the ambient HERMES_HOME) rather than leaving the
        entry unset."""
        import cron.scheduler as sched
        import hermes_constants

        assert hermes_constants.get_hermes_home_override() is None
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        assert sched.try_register_running_job("job-y") is True

        assert sched._running_job_homes["job-y"] == str(tmp_path.resolve())

    def test_release_clears_home(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-x")
        sched._running_job_homes["job-x"] = "/some/profile/home"

        sched.release_running_job("job-x")

        assert "job-x" not in sched._running_job_homes
        assert "job-x" not in sched._running_job_ids

    def test_home_capture_failure_falls_back_to_none_without_blocking_registration(self):
        """A home-capture error must never stop the job from being
        registered as running -- worst case the later interrupt-mark falls
        back to the default store for just this one job, matching pre-fix
        behavior; it must not lose in-flight tracking entirely."""
        import cron.scheduler as sched

        with patch("cron.scheduler.get_hermes_home", side_effect=RuntimeError("boom")):
            assert sched.try_register_running_job("job-z") is True

        assert sched._running_job_homes["job-z"] is None
        assert "job-z" in sched._running_job_ids


class TestMarkRunningJobsInterruptedScoping:
    """#60432 follow-up (2026-08-24): root-caused incident -- the shutdown
    path called mark_job_run() with no profile scope active, so every mark
    landed in the DEFAULT cron store regardless of which profile actually
    owned the job. Job d999d6f70e16 (owned by profile `trove`) was marked in
    the default store; `cron.jobs.mark_job_run` logged "job_id ... not
    found, skipping save" for the real (trove) record, which stayed wedged
    'running' and froze the multiplex tick pass for 7.5h.

    These tests exercise the real cron.jobs store (via use_cron_store()),
    not a mocked mark_job_run, so they prove the fix at the file-scoping
    level the incident actually broke at.
    """

    def test_scoped_to_owning_profile_not_default_store(self, tmp_path, monkeypatch):
        import cron.jobs as jobs
        import cron.scheduler as sched
        import hermes_constants

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "trove"
        default_home.mkdir(parents=True)
        profile_home.mkdir(parents=True)

        # The shutdown thread's ambient home -- must NOT receive the write.
        monkeypatch.setenv("HERMES_HOME", str(default_home))

        # Create the job directly in the profile's own store.
        with jobs.use_cron_store(profile_home):
            job = jobs.create_job(prompt="do work", schedule="every 1h")
        job_id = job["id"]

        # Simulate a normal tick's dispatch: try_register_running_job runs
        # while the profile's scope is active on this thread, exactly as
        # scheduler_provider._start_multiplex sets up around cron_tick().
        home_token = hermes_constants.set_hermes_home_override(str(profile_home))
        try:
            with jobs.use_cron_store(profile_home):
                assert sched.try_register_running_job(job_id) is True
        finally:
            hermes_constants.reset_hermes_home_override(home_token)

        # Now simulate the shutdown thread: no profile scope active at all
        # (the bug this fixes ran mark_job_run exactly like this).
        assert hermes_constants.get_hermes_home_override() is None
        marked = sched.mark_running_jobs_interrupted("gateway shutdown (test)")

        assert marked == [job_id]

        # The profile's OWN store carries the interrupted mark.
        with jobs.use_cron_store(profile_home):
            updated = jobs.get_job(job_id)
        assert updated is not None
        assert updated["last_status"] == "error"
        assert "gateway shutdown (test)" in updated["last_error"]

        # The default store was never touched -- no orphaned phantom entry,
        # exactly the corruption the incident produced.
        with jobs.use_cron_store(default_home):
            assert jobs.load_jobs() == []

    def test_not_found_in_owning_store_logs_error_and_is_excluded(self, tmp_path, monkeypatch, caplog):
        """A job whose captured home doesn't actually contain it (scoping
        still wrong somehow, or removed mid-run) must be LOUD, not a silent
        WARNING indistinguishable from routine noise -- and must not be
        reported to the caller as successfully marked."""
        import logging

        import cron.scheduler as sched
        import hermes_constants

        default_home = tmp_path / "default"
        ghost_home = tmp_path / "profiles" / "ghost"
        default_home.mkdir(parents=True)
        ghost_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(default_home))

        sched._running_job_ids.add("phantom-job")
        sched._running_job_homes["phantom-job"] = str(ghost_home.resolve())

        with caplog.at_level(logging.ERROR, logger="cron.scheduler"):
            marked = sched.mark_running_jobs_interrupted("gateway shutdown (test)")

        assert marked == []
        assert any(
            "not found in its owning store" in r.message for r in caplog.records
        )

    def test_no_captured_home_falls_back_to_default_store(self, tmp_path, monkeypatch):
        """A job registered with no home captured (single-profile gateway,
        or the pre-fix code path for anything that predates registration
        tracking) must still get marked -- in the default store, same as
        before this fix existed."""
        import cron.jobs as jobs
        import cron.scheduler as sched

        default_home = tmp_path / "default"
        default_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(default_home))

        job = jobs.create_job(prompt="do work", schedule="every 1h")
        sched._running_job_ids.add(job["id"])
        sched._running_job_homes[job["id"]] = None

        marked = sched.mark_running_jobs_interrupted("gateway shutdown (test)")

        assert marked == [job["id"]]
        updated = jobs.get_job(job["id"])
        assert updated["last_status"] == "error"
