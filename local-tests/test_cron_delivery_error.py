"""Tests of the per-execution cron delivery-failure reason (port of probepark cron-delivery-error-reason).

The patch adds a nullable ``delivery_error`` TEXT column to ``cron/executions.db`` (an additive
migration through ``add_column_if_missing``) and a ``delivery_error`` keyword on
``finish_execution`` next to ``cancelled``. ``_bounded_delivery_error`` scrubs the text and caps it
at ``MAX_DELIVERY_ERROR_CHARS`` (4000). The scheduler passes the reason in the completed,
interrupted and crash finish paths, and ``hermes cron runs`` prints it for each attempt. The
jobs store keeps only the latest ``last_delivery_error`` per job; the ledger row is now the
per-attempt record. The tests show:
 - a completed run whose delivery failed stores the reason on its execution row;
 - a 10000-character reason is cut to 4000 characters (plus the truncation marker);
 - a ledger with the base schema gets the ``delivery_error`` column on the next open;
 - ``cron runs`` output shows the reason of the attempt;
 - a successful delivery stores NULL.
"""
import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-delivery-error-")
os.environ["HERMES_HOME"] = _HOME

import cron.executions as executions  # noqa: E402
import cron.jobs as jobs  # noqa: E402
import cron.scheduler as scheduler  # noqa: E402
from cron.scheduler_bookkeeping import finish_completed_run  # noqa: E402
from hermes_cli.cron import cron_runs  # noqa: E402

REASON = "telegram: chat not found (HTTP 400)"

_BASE_SCHEMA = """CREATE TABLE executions (
     id TEXT PRIMARY KEY,
     job_id TEXT NOT NULL,
     source TEXT NOT NULL,
     process_id TEXT NOT NULL,
     pid INTEGER NOT NULL,
     process_started_at INTEGER,
     status TEXT NOT NULL CHECK(status IN
       ('claimed','running','completed','failed','unknown','cancelled','missed')),
     handoff_pending INTEGER NOT NULL DEFAULT 0,
     handoff_started_at REAL,
     claimed_at TEXT NOT NULL,
     started_at TEXT,
     finished_at TEXT,
     error TEXT,
     delivery_outcome TEXT,
     scheduled_instant TEXT,
     progress_at TEXT
   )"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _LedgerCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="home-", dir=_HOME)
        home = Path(self._tmp.name)
        env = patch.dict(os.environ, {"HERMES_HOME": str(home)})
        env.start()
        self.addCleanup(env.stop)
        self.db_path = home / "cron" / "executions.db"
        ledger = patch.object(executions, "EXECUTIONS_FILE", self.db_path)
        ledger.start()
        self.addCleanup(ledger.stop)
        store = jobs.use_cron_store(home)
        store.__enter__()
        self.addCleanup(store.__exit__, None, None, None)
        self.addCleanup(self._tmp.cleanup)

    def _running(self, job_id="job1") -> str:
        execution_id = executions.create_execution(job_id, source="direct")["id"]
        self.assertIsNotNone(executions.mark_execution_running(execution_id))
        return execution_id

    def _row(self, execution_id) -> dict:
        record = executions.get_execution(execution_id)
        self.assertIsNotNone(record)
        return record


class FailedDeliveryReasonTest(_LedgerCase):
    def _finish_completed(self, delivery_error):
        job = {"id": "job1", "name": "job1", "prompt": "noop", "enabled": True,
               "state": "scheduled", "schedule": {"kind": "interval", "minutes": 60},
               "deliver": "telegram", "no_agent": True}
        jobs.save_jobs([job])
        execution_id = self._running("job1")
        d = scheduler._RunDelivery(
            job=dict(job, execution_id=execution_id), success=True, error=None,
            delivery_attempted=True, delivery_error=delivery_error, should_deliver=True)
        self.assertTrue(finish_completed_run(d, None, execution_id))
        return self._row(execution_id)

    def test_failed_delivery_stores_reason_on_execution_row(self):
        row = self._finish_completed(REASON)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["delivery_outcome"], "failed")
        self.assertEqual(row.get("delivery_error"), REASON)
        self.assertIsNone(row["error"])

    def test_successful_delivery_stores_null(self):
        row = self._finish_completed(None)
        self.assertEqual(row["delivery_outcome"], "delivered")
        self.assertIn("delivery_error", row)
        self.assertIsNone(row["delivery_error"])


class BoundedReasonTest(_LedgerCase):
    def test_long_reason_is_cut_to_4000_chars(self):
        execution_id = self._running()
        executions.finish_execution(
            execution_id, success=True, delivery_outcome="failed", delivery_error="x" * 10000)
        stored = self._row(execution_id).get("delivery_error")
        self.assertIsNotNone(stored)
        self.assertTrue(stored.startswith("x" * 4000))
        self.assertNotIn("x" * 4001, stored)
        self.assertLess(len(stored), 4100)

    def test_cancelled_finish_keeps_the_reason(self):
        execution_id = self._running()
        executions.finish_execution(
            execution_id, success=False, error="stopped", cancelled=True, delivery_error=REASON)
        row = self._row(execution_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row.get("delivery_error"), REASON)


class MigrationTest(_LedgerCase):
    def test_base_schema_ledger_gets_the_column(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(_BASE_SCHEMA)
        conn.execute(
            "INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at) "
            "VALUES ('old1', 'job1', 'direct', 'p', 1, 'completed', ?)", (_now(),))
        conn.commit()
        conn.close()

        records = executions.list_executions(job_id="job1")
        self.assertEqual([r["id"] for r in records], ["old1"])
        self.assertIn("delivery_error", records[0])
        self.assertIsNone(records[0]["delivery_error"])
        conn = sqlite3.connect(self.db_path)
        try:
            columns = [r[1] for r in conn.execute("PRAGMA table_info(executions)")]
        finally:
            conn.close()
        self.assertIn("delivery_error", columns)


class CronRunsOutputTest(_LedgerCase):
    def test_cron_runs_shows_the_reason(self):
        execution_id = self._running("job1")
        executions.finish_execution(
            execution_id, success=True, delivery_outcome="failed", delivery_error=REASON)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cron_runs(job_id="job1")
        text = out.getvalue()
        self.assertIn(execution_id, text)
        self.assertIn("delivery:", text)
        self.assertIn(REASON, text)


if __name__ == "__main__":
    unittest.main()
