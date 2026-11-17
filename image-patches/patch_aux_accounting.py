"""Local patch of Hermes: the aux usage-cost lookup stays offline (no base_url).

``record_aux_usage`` (agent/aux_accounting.py) prices every auxiliary call through
``estimate_usage_cost(..., base_url=base_url)``. With a base URL the pricing code does a live
``GET /v1/models`` without the provider key; the local Bifrost answers 401 per provider, so the
lookup is a failed network call and the aux cost estimate stays empty anyway. Passing ``base_url=None``
keeps the lookup offline and removes the dead request. The build fails when the old text is gone.

The old text must occur exactly once, else the build fails.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/hermes"

PATCHES = [
    ("agent/aux_accounting.py",
     "            cost = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)\n",
     "            cost = estimate_usage_cost(model, usage, provider=provider, base_url=None)\n"),
]


def main():
    texts = {}
    for rel, old, new in PATCHES:
        if rel not in texts:
            with open(f"{ROOT}/{rel}", encoding="utf-8") as f:
                texts[rel] = f.read()
            if "base_url=None)" in texts[rel] and "estimate_usage_cost(model, usage, provider=provider, base_url=None)" in texts[rel]:
                sys.exit(f"patch does not apply: {rel} is already patched")
        count = texts[rel].count(old)
        if count != 1:
            sys.exit(f"patch does not apply: {rel}: old text found {count} times")
        texts[rel] = texts[rel].replace(old, new)
    for rel, text in texts.items():
        with open(f"{ROOT}/{rel}", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
