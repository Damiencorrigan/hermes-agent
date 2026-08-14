"""Tests for the cronjob() 'critical' flag (S50 — cost-saving mode gate).

``critical: true`` marks a job worth paid tokens even under cost-saving mode 2.
The flag must be settable via ``create_job`` (not hand-edit-only), read back
correctly, and ABSENT (not False) when never set — so existing jobs stay
byte-identical.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with a temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


def test_critical_true_is_persisted_and_read_back(cron_env):
    from cron.jobs import create_job, get_job

    job = create_job(prompt="x", schedule="every 1h", critical=True)
    assert job["critical"] is True
    assert get_job(job["id"])["critical"] is True


def test_critical_false_is_persisted(cron_env):
    from cron.jobs import create_job

    job = create_job(prompt="x", schedule="every 1h", critical=False)
    assert job["critical"] is False


def test_critical_absent_when_not_set(cron_env):
    from cron.jobs import create_job

    job = create_job(prompt="x", schedule="every 1h")
    assert "critical" not in job
