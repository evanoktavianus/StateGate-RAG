"""The runs of E7: the 218 prompts under `no_access_control` and `filtering_guidance`, GLM.

    runs = build(prompts, alt_ids, model)        # 218 x 2 = 436 runs

A planted prompt is asked with the corpus variant that holds its passage (`main`, or `alt` when another prompt shares its carrier passage). Runs of the same prompt are put
together; the order is by a checksum of the id.
"""

from __future__ import annotations

import zlib

MODEL = "z-ai/glm-5.3-flash"
CONDITIONS = ("no_access_control", "filtering_guidance")


def build(prompts: list[dict], alt_ids: set[str], model: str = MODEL, conditions=CONDITIONS) -> list[dict]:
    runs = []
    for p in sorted(prompts, key=lambda p: zlib.crc32(("e7\0" + p["id"]).encode())):
        for cond in conditions:
            base = {"prompt_id": p["id"], "type": p["type"], "placement": p["placement"], "condition": cond, "workflow": p["workflow"], "user": p["person"], "model": model}
            corpus = "alt" if p["id"] in alt_ids else "main"
            runs.append({**base, "key": f"{p['id']}|{cond}|target", "role": "target", "question": p["question"], "corpus": corpus})
    return runs
