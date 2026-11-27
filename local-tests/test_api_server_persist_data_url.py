"""Tests of the opt-in data-URL image persistence port (fork finding #113) in the API server.

With ``gateway.api_server.persist_data_url_images: true``, ``_normalize_multimodal_content`` saves
each inbound ``data:image/...;base64,...`` image in the gateway image cache as
``api_inbound_<sha256[:16]><ext>`` and adds one text note ``[image '<name>' saved at: <path>]``
next to the unchanged image part. The tests show:
 - with the default config, no file is written and the content is the same as before the patch;
 - when enabled, a PNG data URL is saved once under its sha256 name, the image part stays, and
   exactly 1 text note with the saved path is added;
 - a replay of the normalized content (conversation_history / previous_response_id) adds no
   second note and no second file.
Each test uses its own temporary HERMES_HOME.
"""
import base64
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.platforms.api_server import _normalize_multimodal_content
from gateway.platforms.base import get_image_cache_dir

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(64))
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()
STABLE_NAME = f"api_inbound_{hashlib.sha256(PNG).hexdigest()[:16]}.png"


def _content():
    return [{"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": DATA_URL}}]


class _HomeCase(unittest.TestCase):
    config_text = ""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="hermes-home-")
        self.addCleanup(shutil.rmtree, self.home, True)
        env = patch.dict(os.environ, {"HERMES_HOME": self.home, "TERMINAL_ENV": "local"})
        env.start()
        self.addCleanup(env.stop)
        if self.config_text:
            Path(self.home, "config.yaml").write_text(self.config_text)

    def home_files(self):
        """Files in the media caches (config load itself writes SOUL.md and config backups)."""
        roots = [Path(self.home, "cache"), Path(self.home, "image_cache")]
        return sorted(p for r in roots if r.exists() for p in r.rglob("*") if p.is_file())


class DefaultConfigTest(_HomeCase):
    def test_default_writes_no_file_and_keeps_content(self):
        out = _normalize_multimodal_content(_content())
        self.assertEqual(out, _content())
        self.assertEqual(self.home_files(), [])

    def test_explicit_false_writes_no_file(self):
        Path(self.home, "config.yaml").write_text(
            "gateway:\n  api_server:\n    persist_data_url_images: false\n")
        self.assertEqual(_normalize_multimodal_content(_content()), _content())
        self.assertEqual(self.home_files(), [])


class EnabledTest(_HomeCase):
    config_text = "gateway:\n  api_server:\n    persist_data_url_images: true\n"

    def _notes(self, parts):
        return [p for p in parts if p.get("type") == "text" and "saved at:" in p.get("text", "")]

    def test_png_saved_once_under_sha256_name_with_one_note(self):
        out = _normalize_multimodal_content(_content())
        stable = (get_image_cache_dir() / STABLE_NAME).resolve()
        self.assertTrue(stable.is_file())
        self.assertEqual(stable.read_bytes(), PNG)
        self.assertEqual([p.resolve() for p in self.home_files()], [stable])
        self.assertIn({"type": "image_url", "image_url": {"url": DATA_URL}}, out)
        notes = self._notes(out)
        self.assertEqual(len(notes), 1)
        self.assertIn(str(get_image_cache_dir() / STABLE_NAME), notes[0]["text"])

    def test_replay_adds_no_second_note_or_file(self):
        first = _normalize_multimodal_content(_content())
        replay = _normalize_multimodal_content(first)
        self.assertEqual(replay, first)
        self.assertEqual(len(self._notes(replay)), 1)
        self.assertEqual(len(self.home_files()), 1)

    def test_invalid_image_bytes_write_nothing(self):
        bad = "data:image/png;base64," + base64.b64encode(b"<html>not an image</html>").decode()
        content = [{"type": "image_url", "image_url": {"url": bad}}]
        self.assertEqual(_normalize_multimodal_content(content), content)
        self.assertEqual(self.home_files(), [])


if __name__ == "__main__":
    unittest.main()
