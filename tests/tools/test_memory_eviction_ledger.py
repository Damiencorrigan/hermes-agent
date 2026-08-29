"""Tests for the no-silent-eviction rule in tools/memory_tool.py.

Fleet constitution Article 5.1: a fact may never be evicted from MEMORY.md /
USER.md without a visible record. These tests pin the enforcement contract:

  - remove / replace / apply_batch append a JSON row to eviction_ledger.jsonl
    BEFORE the mutation lands on disk
  - the evicted text is archived (MEMORY.archive.md / USER.archive.md)
  - target=user rows are flagged protected (Damien corrections)
  - a rejected batch writes NO ledger rows
  - the record-keeping can be disabled via config keys
"""

import json

import pytest

import tools.memory_tool as mt
from tools.memory_tool import MemoryStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A MemoryStore with temp storage (mirrors test_memory_tool.py)."""
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=2000, user_char_limit=2000)
    s.load_from_disk()
    return s


def _ledger_rows(tmp_path):
    ledger = tmp_path / "eviction_ledger.jsonl"
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def test_remove_logs_ledger_and_archive_before_disk_write(store, tmp_path):
    store.add("memory", "Fact one: deploy to main on Fridays")
    result = store.remove("memory", "deploy to main")

    assert result["success"] is True
    rows = _ledger_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["action"] == "remove"
    assert rows[0]["target"] == "memory"
    assert "deploy to main" in rows[0]["content"]
    assert rows[0]["protected"] is False

    archive = (tmp_path / "MEMORY.archive.md").read_text()
    assert "deploy to main" in archive
    # The fact is gone from the live store.
    assert "deploy to main" not in (tmp_path / "MEMORY.md").read_text()


def test_replace_logs_old_content(store, tmp_path):
    store.add("memory", "Old entry to be superseded")
    result = store.replace("memory", "Old entry", "New replacement entry")

    assert result["success"] is True
    rows = _ledger_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["action"] == "replace"
    assert rows[0]["content"] == "Old entry to be superseded"
    # New content is on disk, old content is not.
    disk = (tmp_path / "MEMORY.md").read_text()
    assert "New replacement entry" in disk
    assert "Old entry" not in disk


def test_user_target_eviction_is_flagged_protected(store, tmp_path):
    store.add("user", "Damien prefers short reports")
    result = store.remove("user", "short reports")

    assert result["success"] is True
    rows = _ledger_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["target"] == "user"
    assert rows[0]["protected"] is True
    archive = (tmp_path / "USER.archive.md").read_text()
    assert "short reports" in archive


def test_batch_logs_each_evicted_entry(store, tmp_path):
    store.add("memory", "Entry A stale")
    store.add("memory", "Entry B stale")
    store.add("memory", "Entry C keep")

    result = store.apply_batch("memory", [
        {"action": "remove", "old_text": "Entry A"},
        {"action": "replace", "old_text": "Entry B", "content": "Entry B fresh"},
    ])

    assert result["success"] is True
    rows = _ledger_rows(tmp_path)
    assert len(rows) == 2
    by_action = {r["action"]: r["content"] for r in rows}
    assert by_action["batch-remove"] == "Entry A stale"
    assert by_action["batch-replace"] == "Entry B stale"


def test_rejected_batch_writes_no_ledger_rows(store, tmp_path):
    store.add("memory", "Entry A stale")
    result = store.apply_batch("memory", [
        {"action": "remove", "old_text": "Entry A"},
        {"action": "remove", "old_text": "no such entry"},
    ])

    assert result["success"] is False
    assert _ledger_rows(tmp_path) == []
    # All-or-nothing: the first remove was not applied either.
    assert "Entry A stale" in (tmp_path / "MEMORY.md").read_text()


def test_settings_can_disable_record_keeping(store, tmp_path, monkeypatch):
    monkeypatch.setattr(
        mt,
        "_load_eviction_settings",
        lambda: {
            "ledger_enabled": False,
            "archive_enabled": False,
            "ledger_path": None,
            "archive_path": None,
        },
    )
    store.add("memory", "Fact to evict")
    result = store.remove("memory", "Fact to evict")

    assert result["success"] is True
    assert _ledger_rows(tmp_path) == []
    assert not (tmp_path / "MEMORY.archive.md").exists()


def test_archive_path_override_is_honored(store, tmp_path, monkeypatch):
    override = tmp_path / "custom_archive.md"
    monkeypatch.setattr(
        mt,
        "_load_eviction_settings",
        lambda: {
            "ledger_enabled": True,
            "archive_enabled": True,
            "ledger_path": str(tmp_path / "custom_ledger.jsonl"),
            "archive_path": str(override),
        },
    )
    store.add("memory", "Fact routed to custom archive")
    store.remove("memory", "custom archive")

    assert "custom archive" in override.read_text()
    assert "custom archive" in (tmp_path / "custom_ledger.jsonl").read_text()
