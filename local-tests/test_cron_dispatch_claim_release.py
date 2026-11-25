"""Tests of the cron rejected-dispatch claim release (port of djagya fix/cron-dispatch-reject-claim).

The patch adds ``cron.jobs.release_fire_claim``, which clears a job's ``fire_claim`` only when
its owner matches ``expected_owner`` (under the per-job fire fence). The scheduler run body calls
it when ``claim_dispatch`` rejects a fire, because no run follows and ``mark_job_run`` never
clears the claim. The tests show:
 - a fire that ``claim_dispatch`` rejects leaves the job with no ``fire_claim``;
 - after that, the one-shot can be re-armed and claimed for a new fire at once;
 - ``release_fire_claim`` does not clear a claim of a different owner;
 - a normal dispatch keeps its own claim while the job runs, and the claim is cleared at the end.
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from unittest.mock import patch

_HOME = tempfile.mkdtemp(prefix="hermes-claim-release-")
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})


def setUpModule():
    # cron.jobs reads HERMES_HOME at import time, so the import runs after the patch starts.
    global jobs, scheduler
    _ENV.start()
    import cron.jobs as jobs
    import cron.scheduler as scheduler


def tearDownModule():
    _ENV.stop()


OWNER = "test-host:owner-1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _oneshot(job_id: str, *, completed: int, last_run_at=None) -> dict:
    return {
        "id": job_id,
        "name": job_id,
        "prompt": "noop",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "once", "run_at": _now(), "display": "once"},
        "next_run_at": _now(),
        "repeat": {"times": 1, "completed": completed},
        "last_run_at": last_run_at,
        "deliver": "local",
        "fire_claim": {"at": _now(), "by": OWNER},
    }


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=_HOME)
        self._store = jobs.use_cron_store(self._tmp.name)
        self._store.__enter__()

    def tearDown(self):
        self._store.__exit__(None, None, None)
        self._tmp.cleanup()

    def _stored(self, job_id: str) -> dict:
        return next(j for j in jobs.load_jobs() if j["id"] == job_id)


class RejectedDispatchReleasesClaimTest(_StoreCase):
    def _fire_rejected(self, job_id: str) -> None:
        # completed == times with a prior run: claim_dispatch rejects and retains the record.
        jobs.save_jobs([_oneshot(job_id, completed=1, last_run_at=_now())])
        result = scheduler._run_one_job_body(dict(self._stored(job_id)))
        self.assertTrue(result)

    def test_rejected_dispatch_clears_fire_claim(self):
        self._fire_rejected("rej1")
        job = self._stored("rej1")
        self.assertEqual(job.get("state"), "completed")
        self.assertIsNone(job.get("fire_claim"))

    def test_rejected_job_can_be_rearmed_and_fired_again(self):
        self._fire_rejected("rej2")
        run_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        rearmed = jobs.rearm_oneshot("rej2", run_at)
        self.assertIsNotNone(rearmed)
        self.assertTrue(jobs.claim_job_for_fire("rej2"))
        self.assertIsInstance(self._stored("rej2").get("fire_claim"), dict)


class ReleaseFireClaimOwnerFenceTest(_StoreCase):
    def test_other_owner_claim_is_kept(self):
        jobs.save_jobs([_oneshot("own1", completed=0)])
        self.assertFalse(jobs.release_fire_claim("own1", expected_owner="someone-else"))
        self.assertEqual(self._stored("own1")["fire_claim"]["by"], OWNER)
        self.assertTrue(jobs.release_fire_claim("own1", expected_owner=OWNER))
        self.assertIsNone(self._stored("own1").get("fire_claim"))


class NormalDispatchKeepsClaimTest(_StoreCase):
    def test_claim_held_during_run_and_cleared_at_end(self):
        jobs.save_jobs([_oneshot("ok1", completed=0)])
        seen = []
        real_run_job = scheduler.run_job

        # A stub stands in for run_job: the real one starts an LLM agent.
        def fake_run_job(job, **_kwargs):
            seen.append(self._stored("ok1").get("fire_claim"))
            return True, "output", "final response", None

        scheduler.run_job = fake_run_job
        try:
            self.assertTrue(scheduler._run_one_job_body(dict(self._stored("ok1"))))
        finally:
            scheduler.run_job = real_run_job
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0] or {}).get("by"), OWNER)
        self.assertIsNone(self._stored("ok1").get("fire_claim"))


if __name__ == "__main__":
    unittest.main()
