"""Tests of the busy-queue user-priority port (gateway/run_busy.py), run in the image build.

The patch adds ``_enqueue_user_ahead_of_internal``: while a session is busy, real user input is
queued ahead of internal events (goal continuations, webhook wakes, terminal/process notifications,
all ``MessageEvent(internal=True)``), and the order within each class is kept. When the queue is at
its cap, the oldest internal event is dropped instead of the user input. The tests show:
 - a full queue of internal events + 1 user message: 1 internal event is dropped, the user message
   is next, and the remaining internal events keep their order;
 - a user message under the cap goes ahead of queued internal events, nothing is dropped;
 - a full queue of user messages keeps the old behavior: the new user message is dropped;
 - user messages without queued internal events stay plain FIFO.
"""
import os
import tempfile
import unittest

from unittest.mock import MagicMock, patch

_HOME = tempfile.mkdtemp(prefix="hermes-home-")
_ENV = patch.dict(os.environ, {"HERMES_HOME": _HOME})

SK = "telegram:user:1"


def setUpModule():
    # The gateway imports touch HERMES_HOME at import time, so they run after the patch starts.
    global BasePlatformAdapter, Platform, PlatformConfig, MessageEvent, MessageType, GatewayRunner
    global StubAdapter
    _ENV.start()
    from gateway.platforms.base import BasePlatformAdapter, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner

    class StubAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            self._mark_disconnected()

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            raise AssertionError("queueing must not send")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id, "type": "dm"}


def tearDownModule():
    _ENV.stop()


def _source():
    return MagicMock(chat_id="123", platform=Platform.TELEGRAM, profile=None)


def user(text):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(), message_id=text)


def goal_continuation():
    return MessageEvent(text="[Continuing toward your standing goal]\nGoal: ship it",
                        message_type=MessageType.TEXT, source=_source(), internal=True)


def webhook_wake():
    return MessageEvent(text="webhook wake", message_type=MessageType.TEXT, source=_source(),
                        internal=True, metadata={"notification_category": "wake"})


def terminal_notify():
    return MessageEvent(text="process finished", message_type=MessageType.TEXT, source=_source(),
                        internal=True, metadata={"notification_category": "result"})


class QueueCase(unittest.TestCase):
    CAP = 3

    def setUp(self):
        self.runner = GatewayRunner.__new__(GatewayRunner)
        self.runner._BUSY_QUEUE_MAX_PENDING = self.CAP
        self.adapter = StubAdapter()
        self.runner._delivery_adapter_for = lambda source: self.adapter

    def fill(self, *events):
        for event in events:
            self.runner._enqueue_fifo(SK, event, self.adapter)

    def queue_texts(self):
        slot = self.adapter._pending_messages.get(SK)
        head = [slot.text] if slot is not None else []
        return head + [e.text for e in self.runner._session_state(SK).conversation.queued_events]


class FullInternalQueueTest(QueueCase):
    def test_user_message_drops_one_internal_event_and_runs_next(self):
        goal, wake, notify = goal_continuation(), webhook_wake(), terminal_notify()
        self.fill(goal, wake, notify)
        msg = user("real question")

        self.runner._queue_or_replace_pending_event(SK, msg)

        self.assertIs(self.adapter._pending_messages[SK], msg)
        self.assertEqual(self.queue_texts(), ["real question", "webhook wake", "process finished"])
        self.assertEqual(self.runner._queue_depth(SK, adapter=self.adapter), self.CAP)

    def test_at_cap_user_lands_after_queued_users_and_oldest_internal_is_dropped(self):
        self.fill(user("u1"), webhook_wake(), terminal_notify())

        self.runner._queue_or_replace_pending_event(SK, user("u2"))

        self.assertEqual(self.queue_texts(), ["u1", "u2", "process finished"])


class UnderCapPriorityTest(QueueCase):
    def test_user_message_goes_ahead_of_internal_events_without_drop(self):
        self.runner._BUSY_QUEUE_MAX_PENDING = 10
        self.fill(goal_continuation(), webhook_wake())

        self.runner._queue_or_replace_pending_event(SK, user("u1"))
        self.runner._queue_or_replace_pending_event(SK, user("u2"))

        self.assertEqual(self.queue_texts(), ["u1", "u2", goal_continuation().text, "webhook wake"])


class FullUserQueueTest(QueueCase):
    def test_full_user_queue_drops_the_new_user_message(self):
        self.fill(user("u1"), user("u2"), user("u3"))
        msg = user("u4")

        self.runner._queue_or_replace_pending_event(SK, msg)

        self.assertEqual(self.queue_texts(), ["u1", "u2", "u3"])
        self.assertFalse(getattr(msg, "_gateway_accepted", False))

    def test_user_messages_without_internal_events_stay_fifo(self):
        self.runner._BUSY_QUEUE_MAX_PENDING = 10
        self.fill(user("u1"))

        self.runner._queue_or_replace_pending_event(SK, user("u2"))
        self.runner._queue_or_replace_pending_event(SK, user("u3"))

        self.assertEqual(self.queue_texts(), ["u1", "u2", "u3"])


if __name__ == "__main__":
    unittest.main()
