"""Tests of the local runs-multimodal patch (patch_runs_multimodal.py), run in the image build.

POST /v1/runs takes the content of the last input item; a list content is normalized like the chat route,
a bad part gives 400 with the validation message, and a str stays the same str.
"""
import json
import unittest

from gateway.platforms.api_server import _openai_error
from gateway.platforms.api_server_runs import _run_user_message
from gateway.session_api_media import _image_urls
from session_rules.api_payloads import data_url_bytes

# 1x1 PNG
PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
       "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def run_input(content):
    return [{"role": "user", "content": content}]


def error_body(response):
    return json.loads(response.text)["error"]


class TextTest(unittest.TestCase):
    def test_str_input_stays_the_same_str(self):
        message, err = _run_user_message("check the service", _openai_error)
        self.assertIsNone(err)
        self.assertEqual(message, "check the service")

    def test_str_content_stays_the_same_str(self):
        message, err = _run_user_message(run_input("check the service"), _openai_error)
        self.assertIsNone(err)
        self.assertEqual(message, "check the service")

    def test_text_parts_collapse_to_a_str(self):
        parts = [{"type": "input_text", "text": "first line"}, {"type": "text", "text": "second line"}]
        message, err = _run_user_message(run_input(parts), _openai_error)
        self.assertIsNone(err)
        self.assertEqual(message, "first line\nsecond line")


class ImageTest(unittest.TestCase):
    def test_text_and_png_become_canonical_parts(self):
        parts = [{"type": "input_text", "text": "what is on the image?"},
                 {"type": "input_image", "image_url": PNG}]
        message, err = _run_user_message(run_input(parts), _openai_error)
        self.assertIsNone(err)
        self.assertEqual(message, [{"type": "text", "text": "what is on the image?"},
                                   {"type": "image_url", "image_url": {"url": PNG}}])

    def test_chat_completions_shape_keeps_detail(self):
        parts = [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": PNG, "detail": "low"}}]
        message, err = _run_user_message(run_input(parts), _openai_error)
        self.assertIsNone(err)
        self.assertEqual(message[1], {"type": "image_url", "image_url": {"url": PNG, "detail": "low"}})

    def test_admission_media_reads_the_image(self):
        parts = [{"type": "input_text", "text": "describe"}, {"type": "input_image", "image_url": PNG}]
        message, _ = _run_user_message(run_input(parts), _openai_error)
        decoded = [data_url_bytes(url) for url in _image_urls(message)]
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0][0], "image/png")
        self.assertTrue(decoded[0][1].startswith(b"\x89PNG"))


class BadPartTest(unittest.TestCase):
    def assert_refused(self, parts, code):
        message, err = _run_user_message([{"role": "user", "content": "earlier"}, *run_input(parts)], _openai_error)
        self.assertIsNone(message)
        self.assertEqual(err.status, 400)
        body = error_body(err)
        self.assertEqual(body["code"], code)
        self.assertEqual(body["param"], "input[1].content")
        self.assertTrue(body["message"])

    def test_non_image_data_url_is_refused(self):
        self.assert_refused([{"type": "input_image", "image_url": "data:text/plain;base64,aGk="}],
                            "unsupported_content_type")

    def test_bad_image_url_is_refused(self):
        self.assert_refused([{"type": "input_image", "image_url": "file:///etc/passwd"}], "invalid_image_url")

    def test_file_part_is_refused(self):
        self.assert_refused([{"type": "input_file", "file_id": "f1"}], "unsupported_content_type")

    def test_unknown_part_type_is_refused(self):
        self.assert_refused([{"type": "audio", "data": "x"}], "unsupported_content_type")

    def test_empty_parts_are_refused_as_missing(self):
        message, err = _run_user_message(run_input([{"type": "text", "text": ""}]), _openai_error)
        self.assertIsNone(message)
        self.assertEqual(err.status, 400)
        self.assertEqual(error_body(err)["message"], "No user message found in input")


if __name__ == "__main__":
    unittest.main()
