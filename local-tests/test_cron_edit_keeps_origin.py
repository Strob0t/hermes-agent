"""Tests of the cron edit-keeps-origin port (tools/cronjob_tools.py), run in the image build.

The patch adds ``_resolve_update_deliver``: when the edited job already owns an origin (a captured
job ``origin``, or ``origin`` in its ``deliver`` or ``failure_deliver``), a literal ``origin`` in an
update stays literal. Otherwise it resolves as on create (inside a cron run, to the running job's
concrete target). ``_update_core_fields`` uses it for ``deliver`` and ``failure_deliver``.
The tests show:
 - a prompt-only edit keeps ``deliver`` and ``failure_deliver`` = ``origin``;
 - an edit from inside a cron run that re-sends ``origin`` keeps both literal ``origin``
   (it does not borrow the running job's target);
 - an explicit new ``deliver`` still changes the job;
 - a job without its own origin still resolves ``origin`` to the running job's target.
"""
import json
import os
import tempfile
import unittest

from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})


def setUpModule():
    # cron.jobs reads HERMES_HOME at import time, so the imports run after the patch starts.
    global cron_jobs, _VAR_MAP, clear_session_vars, set_session_vars, cronjob
    _ENV.start()
    from cron import jobs as cron_jobs
    from gateway.session_context import _VAR_MAP, clear_session_vars, set_session_vars
    from tools.cronjob_tools import cronjob


def tearDownModule():
    _ENV.stop()


def _call(**kw):
    return json.loads(cronjob(**kw))


class CronRun:
    """Context of a cron run that auto-delivers to ``platform:chat_id``."""

    def __init__(self, platform, chat_id):
        self.platform, self.chat_id = platform, chat_id

    def __enter__(self):
        self.tokens = set_session_vars(platform="", chat_id="", chat_name="", cron_session="1")
        self.extra = [
            (_VAR_MAP[k], _VAR_MAP[k].set(v)) for k, v in (
                ("HERMES_CRON_AUTO_DELIVER_PLATFORM", self.platform),
                ("HERMES_CRON_AUTO_DELIVER_CHAT_ID", self.chat_id))]
        return self

    def __exit__(self, *exc):
        for var, token in reversed(self.extra):
            var.reset(token)
        clear_session_vars(self.tokens)


class CronStoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store = cron_jobs.use_cron_store(self._tmp.name)
        self._store.__enter__()
        cron_jobs.ensure_dirs()

    def tearDown(self):
        self._store.__exit__(None, None, None)
        self._tmp.cleanup()

    def _create(self, **kw):
        result = _call(action="create", schedule="every 4h", prompt="check the thing", **kw)
        self.assertTrue(result["success"], result)
        return result["job_id"]

    def _delivery(self, job_id):
        job = cron_jobs.get_job(job_id)
        return job.get("deliver"), job.get("failure_deliver")


class TestEditKeepsOwnOrigin(CronStoreCase):
    def test_prompt_only_edit_keeps_origin_delivery(self):
        job_id = self._create(deliver="origin", failure_deliver="origin")
        with CronRun("telegram", "-100999"):
            result = _call(action="update", job_id=job_id, prompt="check the other thing")
        self.assertTrue(result["success"], result)
        self.assertEqual(self._delivery(job_id), ("origin", "origin"))
        self.assertEqual(cron_jobs.get_job(job_id)["prompt"], "check the other thing")

    def test_resent_origin_from_cron_run_keeps_origin(self):
        job_id = self._create(deliver="origin", failure_deliver="origin")
        with CronRun("telegram", "-100999"):
            result = _call(action="update", job_id=job_id, prompt="new prompt",
                           deliver="origin", failure_deliver="origin")
        self.assertTrue(result["success"], result)
        self.assertEqual(self._delivery(job_id), ("origin", "origin"))


class TestEditChangesDelivery(CronStoreCase):
    def test_explicit_deliver_changes_it(self):
        job_id = self._create(deliver="origin", failure_deliver="origin")
        with CronRun("telegram", "-100999"):
            result = _call(action="update", job_id=job_id, deliver="local")
        self.assertTrue(result["success"], result)
        self.assertEqual(self._delivery(job_id), ("local", "origin"))

    def test_job_without_own_origin_resolves_to_running_target(self):
        job_id = self._create(deliver="local")
        with CronRun("telegram", "-100999"):
            result = _call(action="update", job_id=job_id, deliver="origin")
        self.assertTrue(result["success"], result)
        self.assertEqual(self._delivery(job_id)[0], "telegram:-100999")


if __name__ == "__main__":
    unittest.main()
