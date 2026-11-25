"""Tests of the SSE-disconnect port (upstream fork fix for #145750), run in the image build.

With a session authority, a streaming chat completion is only an observer of a durable admission.
Before the patch, a client disconnect cancelled the observer task only, and the canonical turn ran
on. The patch gives the stream an admission future (resolved by ``run_api_turn`` once the row is
committed and bound to the stream id), and on disconnect ``abandon_authority_agent_task`` waits for
it and stops that exact admission through ``request_stop_run`` before it detaches the observer.
``stop_run`` keeps its contract on top of ``request_stop_run``. The tests show:
 - a disconnect while chunks are written interrupts the running canonical turn;
 - a disconnect while the SSE response is prepared also interrupts the turn and re-raises;
 - a disconnect before the admission commits waits for it and cancels only that queued row;
 - an explicit ``stop_run`` still interrupts a streamed turn and returns ``stopping``;
 - a complete stream is unchanged: all frames, ``[DONE]``, outcome ``completed``, no interrupt.
"""
import asyncio
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ControlledAgent:
    """The canonical turn's agent: blocks until released or interrupted."""

    def __init__(self):
        self.started = asyncio.Event()
        self.continue_work = asyncio.Event()
        self.interrupted = False
        self.steps = []
        self.loop = None

    def hard_interrupt(self, _message=None, **_kwargs):
        self.interrupted = True
        self.loop.call_soon_threadsafe(self.continue_work.set)


class FakeSSE:
    """The prepared SSE response: records frames; ``fail_on`` raises a reset on that write."""

    def __init__(self, fail_on=None, before_fail=None):
        self.frames = []
        self.fail_on = fail_on
        self.before_fail = before_fail
        self.failed = asyncio.Event()

    async def write(self, data):
        if self.fail_on is not None and len(self.frames) + 1 == self.fail_on:
            if self.before_fail is not None:
                await self.before_fail()
            self.failed.set()
            raise ConnectionResetError("client disconnected")
        self.frames.append(data)


async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


class SSEDisconnectBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from gateway.config import GatewayConfig, PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.session import SessionStore
        from gateway.session_authority import SessionAuthority
        from persistence_assembly import open_persistence, profile_scope

        self.home = tempfile.mkdtemp(prefix="hermes-sse-")
        env = patch.dict(os.environ, {"HERMES_HOME": self.home})
        env.start()
        self.addCleanup(env.stop)
        self.persistence = open_persistence(profile_scope(home=self.home))
        store = SessionStore(config=GatewayConfig(), sessions_dir=Path(self.home) / "sessions")
        store._db, store._persistence_beside = self.persistence, lambda pinned: pinned
        runner = SimpleNamespace(_draining=False, session_store=store)
        self.owner = SessionAuthority(
            runner, profile_id="default", instance_id="first",
            epoch=self.persistence.runtime.begin_epoch(instance_id="first"), persistence=self.persistence)
        runner.session_authority = self.owner
        self.api = APIServerAdapter(PlatformConfig(enabled=True))
        self.api.gateway_runner = runner
        self.api._session_db = self.persistence
        runner._adapter_for_source = lambda source: self.api
        self.agents = {}

    async def asyncTearDown(self):
        for agent in self.agents.values():
            agent.continue_work.set()
        from gateway import run_runtime
        await wait_until(lambda: all(t.done() for t in run_runtime._authority_tasks(self.owner)), 10)
        self.api._response_store.close()
        self.api._run_idempotency_store.close()
        self.persistence.close()
        shutil.rmtree(self.home, ignore_errors=True)

    def install_execute(self):
        """Every admission runs a ControlledAgent keyed by its text; ``block`` texts wait."""
        from gateway import session_finite
        loop = asyncio.get_running_loop()

        async def execute(authority, ref, row):
            text = row["payload"]["text"]
            agent = self.agents.setdefault(text, ControlledAgent())
            agent.loop = loop
            authority.adopt_agent(ref.session_id, row["generation"], agent)
            agent.steps.append("first")
            agent.started.set()
            if not agent.interrupted:
                await agent.continue_work.wait()
            if not agent.interrupted:
                agent.steps.append("second")
            result = {"final_response": "interrupted" if agent.interrupted else text,
                      "messages": [], "interrupted": agent.interrupted}
            authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
            return result["final_response"]

        p = patch.object(session_finite, "execute_finite_admission", execute)
        p.start()
        self.addCleanup(p.stop)

    def spawn_stream(self, completion_id, *, session_id, text):
        """What ``_handle_chat_completions`` does for ``stream: true`` under an authority."""
        from gateway.session_api_turn import api_admission_future
        from gateway.platforms.api_server import ThreadSafeAsyncQueue
        stream_q = ThreadSafeAsyncQueue()
        future = asyncio.get_running_loop().create_future()
        notify, on_done = self.api._register_stream_approval(
            SimpleNamespace(headers={}), completion_id, stream_q, session_id)
        token = api_admission_future.set(future)
        try:
            task, ref = self.api._spawn_stream_agent(
                stream_q, on_done=on_done, approval_notify_callback=notify,
                approval_session_key=completion_id, user_message=text,
                conversation_history=[], session_id=session_id)
        finally:
            api_admission_future.reset(token)
        return stream_q, future, task, ref

    def write_stream(self, completion_id, stream_q, future, task, ref, response=None, prepare_error=None):
        async def prepare(_request, _session_id, _key):
            if prepare_error is not None:
                raise prepare_error
            return response
        self.api._prepare_sse_response = prepare
        return asyncio.ensure_future(self.api._write_sse_chat_completion(
            SimpleNamespace(headers={}), completion_id, "test-model", 1, stream_q, task, ref,
            admission_future=future))

    def rows(self, session_id):
        from gateway.platforms.api_server_store import authority_store
        return {r.payload["text"]: r for r in authority_store(self.owner).admissions.list(
            session_id=session_id, pending_only=False)}

    async def settle(self, session_id):
        live = self.owner.sessions.get(session_id)
        if live is not None and live.task is not None:
            await asyncio.wait_for(asyncio.shield(live.task), 5)
        await wait_until(lambda: not getattr(self.owner, "api_observers", None))
        await wait_until(lambda: all(r.status == "terminal" for r in self.rows(session_id).values()))


class StreamWriteDisconnectTest(SSEDisconnectBase):
    async def test_disconnect_mid_stream_interrupts_canonical_turn(self):
        self.install_execute()
        cid = "chatcmpl-mid"
        q, fut, task, ref = self.spawn_stream(cid, session_id="mid", text="work")
        writer = self.write_stream(cid, q, fut, task, ref, response=FakeSSE(fail_on=2))
        await wait_until(lambda: "work" in self.agents)
        await asyncio.wait_for(self.agents["work"].started.wait(), 5)
        q.put_nowait("first delta")
        await asyncio.wait_for(writer, 5)
        await self.settle("mid")
        self.assertTrue(self.agents["work"].interrupted)
        self.assertEqual(self.agents["work"].steps, ["first"])
        self.assertNotIn(cid, self.api._run_admission_aliases)
        self.assertNotIn(cid, self.api._stopping_run_ids)

    async def test_complete_stream_is_unchanged(self):
        self.install_execute()
        cid = "chatcmpl-done"
        q, fut, task, ref = self.spawn_stream(cid, session_id="done", text="answer")
        response = FakeSSE()
        writer = self.write_stream(cid, q, fut, task, ref, response=response)
        await wait_until(lambda: "answer" in self.agents)
        await asyncio.wait_for(self.agents["answer"].started.wait(), 5)
        q.put_nowait("hello")
        self.agents["answer"].continue_work.set()
        self.assertIs(await asyncio.wait_for(writer, 5), response)
        await self.settle("done")
        self.assertFalse(self.agents["answer"].interrupted)
        self.assertEqual(self.agents["answer"].steps, ["first", "second"])
        self.assertEqual(response.frames[-1], b"data: [DONE]\n\n")
        self.assertTrue(any(b'"hello"' in f for f in response.frames))
        self.assertEqual(self.rows("done")["answer"].outcome, "completed")
        self.assertNotIn(cid, self.api._stopping_run_ids)


class PrepareDisconnectTest(SSEDisconnectBase):
    async def test_disconnect_while_preparing_sse_interrupts_turn(self):
        self.install_execute()
        cid = "chatcmpl-prep"
        q, fut, task, ref = self.spawn_stream(cid, session_id="prep", text="early")
        writer = self.write_stream(cid, q, fut, task, ref, prepare_error=ConnectionResetError("gone"))
        with self.assertRaises(ConnectionResetError):
            await asyncio.wait_for(writer, 5)
        self.assertTrue(task.done())
        await self.settle("prep")
        row = self.rows("prep")["early"]
        agent = self.agents.get("early")
        stopped = row.outcome == "cancelled" or (agent is not None and agent.interrupted)
        self.assertTrue(stopped, f"turn ran on after the disconnect: {row.outcome}")
        self.assertNotIn("second", agent.steps if agent else [])
        self.assertNotIn(cid, self.api._stopping_run_ids)


class StopRunTest(SSEDisconnectBase):
    async def test_disconnect_during_admission_cancels_only_its_queued_row(self):
        self.install_execute()
        blocker = asyncio.ensure_future(self.api._run_agent(
            user_message="blocker", conversation_history=[], session_id="queued"))
        await wait_until(lambda: "blocker" in self.agents)
        await asyncio.wait_for(self.agents["blocker"].started.wait(), 5)
        lock = self.owner._api_admissions = asyncio.Lock()
        await lock.acquire()
        cid = "chatcmpl-queued"
        q, fut, task, ref = self.spawn_stream(cid, session_id="queued", text="queued")
        response = FakeSSE(fail_on=1)
        writer = self.write_stream(cid, q, fut, task, ref, response=response)
        await asyncio.wait_for(response.failed.wait(), 5)
        await asyncio.sleep(0.05)
        self.assertFalse(writer.done(), "the disconnect did not wait for the in-flight admission")
        lock.release()
        await asyncio.wait_for(writer, 5)
        self.agents["blocker"].continue_work.set()
        await asyncio.wait_for(blocker, 5)
        await self.settle("queued")
        rows = self.rows("queued")
        self.assertNotIn("queued", self.agents)
        self.assertFalse(self.agents["blocker"].interrupted)
        self.assertEqual(rows["blocker"].outcome, "completed")
        self.assertEqual(rows["queued"].outcome, "cancelled")
        self.assertNotIn(cid, self.api._stopping_run_ids)

    async def test_explicit_stop_run_still_interrupts_streamed_turn(self):
        from gateway.platforms.api_server_authority_runs import stop_run
        self.install_execute()
        cid = "chatcmpl-stop"
        _q, _fut, task, _ref = self.spawn_stream(cid, session_id="stop", text="stop me")
        await wait_until(lambda: "stop me" in self.agents)
        await asyncio.wait_for(self.agents["stop me"].started.wait(), 5)
        projection = await stop_run(self.api, cid)
        await asyncio.wait_for(task, 5)
        await self.settle("stop")
        self.assertEqual(projection["status"], "stopping")
        self.assertTrue(self.agents["stop me"].interrupted)
        self.assertNotIn(cid, self.api._stopping_run_ids)


if __name__ == "__main__":
    unittest.main()
