import time

import pytest

from agent.redact import redact_sensitive_text

_INPUTS = [
    "1|" + " " * 40000 + "token:",
    " " * 40000 + "token:",
    "   12\t" + " " * 40000 + "password",
    " " * 40000 + "x",
]


@pytest.mark.parametrize("secret_file", [True, False])
@pytest.mark.parametrize("text", _INPUTS, ids=["gutter-token", "token", "tab-gutter-password", "plain"])
def test_long_indented_line_is_untouched_in_linear_time(text, secret_file):
    started = time.perf_counter()
    result = redact_sensitive_text(text, force=True, file_read=True, secret_file=secret_file)
    elapsed = time.perf_counter() - started
    assert result == text
    assert elapsed < 1.0
