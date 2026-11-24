"""Tests of the kanban parent-result gate port (fork finding #108), run in the image build.

The patch adds a nullable ``task_links.required_parent_result`` column (with a migration for old
boards), ``create_task(required_parent_results={parent: result})`` and
``link_tasks(required_parent_result=...)``. A result-gated edge releases the child only when the
parent is ``done`` with exactly that ``tasks.result``; a NULL edge keeps the done/archived gate.
An idempotent retry with the same key and a different dependency contract is rejected. The tests show:
 - a ``done`` gate promotes the child only when the parent result matches;
 - a different parent result keeps the child in ``todo``;
 - a NULL (plain) link promotes the child on any ``done`` result, as before;
 - the migration adds the column to a board with the old ``task_links`` shape and keeps old links NULL;
 - a retry with the same idempotency key and a different contract raises ``ValueError``;
 - the CLI ``--require-parent-result PARENT=RESULT`` value reaches the link row.
"""
import argparse
import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


class _BoardCase(unittest.TestCase):
    """Each test gets its own HERMES_HOME and kanban DB in a temp dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        home = root / ".hermes"
        home.mkdir()
        self.db_path = home / "kanban.db"
        env = {
            "HERMES_HOME": str(home),
            "HERMES_KANBAN_HOME": str(home),
            "HERMES_KANBAN_DB": str(self.db_path),
        }
        self._env = patch.dict(os.environ, env)
        self._env.start()
        self._home = patch.object(Path, "home", lambda: root)
        self._home.start()
        os.environ.pop("HERMES_KANBAN_TASK", None)
        kb.init_db(self.db_path)
        self.conn = kbc.connect(self.db_path)

    def tearDown(self):
        self.conn.close()
        self._home.stop()
        self._env.stop()
        self._tmp.cleanup()

    def status(self, task_id):
        return self.conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"]

    def link_result(self, parent_id, child_id):
        return self.conn.execute(
            "SELECT required_parent_result FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        ).fetchone()["required_parent_result"]

    def finish(self, task_id, result):
        self.assertTrue(kb.complete_task(self.conn, task_id, result=result, force=True))
        kb.recompute_ready(self.conn)


class ResultGateTests(_BoardCase):
    def test_matching_parent_result_promotes_child(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        child = kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                               required_parent_results={parent: "done"})
        self.assertEqual(self.status(child), "todo")
        self.assertEqual(self.link_result(parent, child), "done")
        self.finish(parent, "done")
        self.assertEqual(self.status(child), "ready")

    def test_different_parent_result_keeps_child_in_todo(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        child = kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                               required_parent_results={parent: "done"})
        self.finish(parent, "rejected")
        self.assertEqual(self.status(parent), "done")
        self.assertEqual(self.status(child), "todo")
        self.assertFalse(kb._parents_satisfied(self.conn, child))
        self.assertIsNone(kb.claim_task(self.conn, child))

    def test_link_tasks_gate_on_existing_child(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        child = kb.create_task(self.conn, title="deploy", assignee="ops")
        kb.link_tasks(self.conn, parent, child, required_parent_result="ready_to_deploy")
        self.assertEqual(self.status(child), "todo")
        self.finish(parent, "needs_work")
        self.assertEqual(self.status(child), "todo")

    def test_null_link_keeps_done_behavior(self):
        parent = kb.create_task(self.conn, title="research", assignee="r")
        child = kb.create_task(self.conn, title="write", assignee="w", parents=[parent])
        self.assertIsNone(self.link_result(parent, child))
        self.assertEqual(self.status(child), "todo")
        self.finish(parent, "anything at all")
        self.assertEqual(self.status(child), "ready")

    def test_create_under_done_parent_with_wrong_result_is_todo(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        self.finish(parent, "rejected")
        plain = kb.create_task(self.conn, title="plain", assignee="w", parents=[parent])
        gated = kb.create_task(self.conn, title="gated", assignee="w", parents=[parent],
                               required_parent_results={parent: "done"})
        self.assertEqual(self.status(plain), "ready")
        self.assertEqual(self.status(gated), "todo")

    def test_required_result_for_unknown_parent_is_rejected(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        with self.assertRaises(ValueError):
            kb.create_task(self.conn, title="x", assignee="w", parents=[parent],
                           required_parent_results={"t_missing": "done"})


class IdempotencyContractTests(_BoardCase):
    def test_same_key_same_contract_returns_existing(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        first = kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                               required_parent_results={parent: "done"}, idempotency_key="k1")
        again = kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                               required_parent_results={parent: "done"}, idempotency_key="k1")
        self.assertEqual(first, again)

    def test_same_key_different_contract_is_rejected(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                       required_parent_results={parent: "done"}, idempotency_key="k2")
        with self.assertRaises(ValueError):
            kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                           required_parent_results={parent: "other"}, idempotency_key="k2")
        with self.assertRaises(ValueError):
            kb.create_task(self.conn, title="deploy", assignee="ops", parents=[parent],
                           idempotency_key="k2")


class MigrationTests(_BoardCase):
    def test_migration_adds_column_to_old_task_links(self):
        parent = kb.create_task(self.conn, title="p", assignee="a")
        child = kb.create_task(self.conn, title="c", assignee="a", parents=[parent])
        self.conn.close()
        # Rebuild task_links in the base-schema shape (no required_parent_result column).
        with contextlib.closing(sqlite3.connect(self.db_path)) as raw:
            raw.executescript(
                "CREATE TABLE task_links_old (parent_id TEXT NOT NULL, child_id TEXT NOT NULL,"
                " PRIMARY KEY (parent_id, child_id));"
                "INSERT INTO task_links_old SELECT parent_id, child_id FROM task_links;"
                "DROP TABLE task_links;"
                "ALTER TABLE task_links_old RENAME TO task_links;"
            )
            cols = {r[1] for r in raw.execute("PRAGMA table_info(task_links)")}
            self.assertNotIn("required_parent_result", cols)
        kb.init_db(self.db_path)
        self.conn = kbc.connect(self.db_path)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(task_links)")}
        self.assertIn("required_parent_result", cols)
        self.assertIsNone(self.link_result(parent, child))
        self.finish(parent, "whatever")
        self.assertEqual(self.status(child), "ready")


class CliTests(_BoardCase):
    def test_cli_parent_result_values(self):
        self.assertEqual(kcli._parse_required_parent_results(["t_a=ok", " t_b = ship "]),
                         {"t_a": "ok", "t_b": "ship"})
        for bad in ("t_a", "=ok", "t_a="):
            with self.assertRaises(ValueError):
                kcli._parse_required_parent_results([bad])

    def test_cli_create_and_link_write_the_gate(self):
        parent = kb.create_task(self.conn, title="qa", assignee="qa")
        parser = argparse.ArgumentParser()
        kcli.build_parser(parser.add_subparsers(dest="cmd"))
        args = parser.parse_args(["kanban", "create", "deploy", "--assignee", "ops",
                                  "--parent", parent, "--require-parent-result", f"{parent}=done",
                                  "--json"])
        self.assertEqual(args.required_parent_results, [f"{parent}=done"])
        with contextlib.redirect_stdout(io.StringIO()):
            rc = kcli._cmd_create(args)
        self.assertIn(rc, (0, None))
        child = self.conn.execute("SELECT child_id FROM task_links WHERE parent_id = ?",
                                  (parent,)).fetchone()["child_id"]
        self.assertEqual(self.link_result(parent, child), "done")

        other = kb.create_task(self.conn, title="other", assignee="ops")
        args = parser.parse_args(["kanban", "link", parent, other, "--required-parent-result", "ok"])
        with contextlib.redirect_stdout(io.StringIO()):
            kcli._cmd_link(args)
        self.assertEqual(self.link_result(parent, other), "ok")
        self.assertEqual(self.status(other), "todo")


if __name__ == "__main__":
    unittest.main()
