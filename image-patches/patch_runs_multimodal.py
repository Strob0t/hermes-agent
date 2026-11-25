"""Local patch of Hermes v0.23.1: POST /v1/runs accepts image parts in the last input item.

_run_user_message passed a list content through without validation: no image part was checked and a bad
part reached the agent. The chat, Responses and session chat routes normalize content with
_normalize_multimodal_content (api_server.py); the patch does the same for a list content of /v1/runs and
answers a bad part with 400 and the validation message. A str input stays the same str.
The old text must occur exactly once, else the build fails.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/hermes"

PATCHES = [
    ("gateway/platforms/api_server_runs.py",
     """        user_message = raw_input[-1].get("content", "") if isinstance(raw_input, list) else ""
    if not user_message:
""",
     """        user_message = raw_input[-1].get("content", "") if isinstance(raw_input, list) else ""
    if isinstance(user_message, list):
        # local patch: validate parts like the chat route; text-only parts collapse to a str
        from gateway.platforms.api_server import _multimodal_validation_error, _normalize_multimodal_content
        try:
            user_message = _normalize_multimodal_content(user_message)
        except ValueError as exc:
            return None, _multimodal_validation_error(exc, param=f"input[{len(raw_input) - 1}].content")
    if not user_message:
"""),
]


def main():
    texts = {}
    for rel, old, new in PATCHES:
        if rel not in texts:
            with open(f"{ROOT}/{rel}", encoding="utf-8") as f:
                texts[rel] = f.read()
            if "local patch:" in texts[rel]:
                sys.exit(f"patch does not apply: {rel} is already patched")
        count = texts[rel].count(old)
        if count != 1:
            sys.exit(f"patch does not apply: {rel}: old text found {count} times")
        texts[rel] = texts[rel].replace(old, new)
    # nothing is written unless every replacement matched
    for rel, text in texts.items():
        with open(f"{ROOT}/{rel}", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
