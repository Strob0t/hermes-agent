"""Tests of the opt-in API-server vision routing port (fork finding #112).

With ``gateway.api_server.image_vision_routing: true``, ``APIServerAdapter._run_agent`` resolves the
image input mode of the main model (``agent.image_routing.decide_image_input_mode``). In text mode it
replaces the ``image_url`` parts of the user message and of the conversation history with the
description notes of ``GatewayInboundMixin._enrich_message_with_vision`` (auxiliary vision via
``vision_analyze_tool``). A ``data:image`` URL is decoded to a file first. The tests show:
 - with the default config, image parts reach the agent unchanged and vision is not called;
 - enabled with a text-only main model, image parts become descriptions, and a data URL is given
   to vision as a local file with the decoded bytes;
 - the same image in many history turns is described once per request;
 - enabled with a vision-capable main model (native mode), the content is unchanged.
Vision is a mock (no network); the agent run is captured at the ``run_api_turn`` hand-off.
Each test uses its own temporary HERMES_HOME.
"""
import asyncio
import base64
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(64))
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()
HTTP_URL = "https://example.invalid/cat.png"


def _image_turn(text, url):
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]


class _Case(unittest.TestCase):
    config_text = ""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-home-")
        self.addCleanup(shutil.rmtree, self.home, True)
        env = patch.dict(os.environ, {"HERMES_HOME": self.home, "TERMINAL_ENV": "local"})
        env.start()
        self.addCleanup(env.stop)
        if self.config_text:
            Path(self.home, "config.yaml").write_text(self.config_text)
        self.vision_calls = []
        self.seen_files = []

        async def fake_vision(image_url, user_prompt, **_kw):
            self.vision_calls.append(image_url)
            if os.path.isfile(image_url):
                self.seen_files.append(Path(image_url).read_bytes())
            return json.dumps({"success": True, "analysis": f"DESC#{len(self.vision_calls)}"})

        vp = patch("tools.vision_tools.vision_analyze_tool", fake_vision)
        vp.start()
        self.addCleanup(vp.stop)

    def run_agent(self, user_message, history):
        captured = {}

        async def fake_run_api_turn(adapter, **kwargs):
            captured.update(kwargs)
            return {"final_response": "ok"}, {}

        api = APIServerAdapter(PlatformConfig(enabled=True))
        api.gateway_runner = SimpleNamespace(session_authority=object())
        self.addCleanup(api._response_store.close)
        self.addCleanup(api._run_idempotency_store.close)
        with patch("gateway.session_api_turn.run_api_turn", fake_run_api_turn):
            asyncio.run(api._run_agent(user_message=user_message, conversation_history=history,
                                       session_id="s1"))
        return captured["user_message"], captured["conversation_history"]


class DefaultConfigTest(_Case):
    config_text = "model:\n  default: text-only-model\n  provider: custom\n  supports_vision: false\n"

    def test_image_parts_reach_agent_unchanged(self):
        msg = _image_turn("What is this?", DATA_URL)
        hist = [{"role": "user", "content": _image_turn("before", HTTP_URL)},
                {"role": "assistant", "content": "a cat"}]
        out_msg, out_hist = self.run_agent(msg, hist)
        self.assertEqual(out_msg, _image_turn("What is this?", DATA_URL))
        self.assertEqual(out_hist, [{"role": "user", "content": _image_turn("before", HTTP_URL)},
                                    {"role": "assistant", "content": "a cat"}])
        self.assertEqual(self.vision_calls, [])


class TextModeTest(_Case):
    config_text = ("model:\n  default: text-only-model\n  provider: custom\n  supports_vision: false\n"
                   "gateway:\n  api_server:\n    image_vision_routing: true\n")

    def test_image_parts_become_descriptions_and_data_url_is_a_file(self):
        out_msg, _ = self.run_agent(_image_turn("What is this?", DATA_URL), [])
        self.assertIsInstance(out_msg, str)
        self.assertIn("DESC#1", out_msg)
        self.assertTrue(out_msg.endswith("What is this?"))
        self.assertNotIn("data:image", out_msg)
        self.assertEqual(len(self.vision_calls), 1)
        self.assertFalse(self.vision_calls[0].startswith("data:"))
        self.assertEqual(self.seen_files, [PNG])

    def test_history_image_is_described_once(self):
        hist = [{"role": "user", "content": _image_turn("first", HTTP_URL)},
                {"role": "assistant", "content": "a cat"},
                {"role": "user", "content": _image_turn("again", HTTP_URL)},
                {"role": "assistant", "content": "still a cat"}]
        out_msg, out_hist = self.run_agent(_image_turn("and now?", HTTP_URL), hist)
        self.assertEqual(self.vision_calls, [HTTP_URL])
        self.assertIn("DESC#1", out_hist[0]["content"])
        self.assertIn("DESC#1", out_hist[2]["content"])
        self.assertIn("DESC#1", out_msg)
        self.assertEqual(out_hist[1], {"role": "assistant", "content": "a cat"})
        self.assertEqual(len(out_hist), 4)


class NativeModeTest(_Case):
    config_text = ("model:\n  default: vision-model\n  provider: custom\n  supports_vision: true\n"
                   "gateway:\n  api_server:\n    image_vision_routing: true\n")

    def test_native_mode_keeps_image_parts(self):
        hist = [{"role": "user", "content": _image_turn("before", HTTP_URL)}]
        out_msg, out_hist = self.run_agent(_image_turn("What is this?", DATA_URL), hist)
        self.assertEqual(out_msg, _image_turn("What is this?", DATA_URL))
        self.assertEqual(out_hist, [{"role": "user", "content": _image_turn("before", HTTP_URL)}])
        self.assertEqual(self.vision_calls, [])


if __name__ == "__main__":
    unittest.main()
