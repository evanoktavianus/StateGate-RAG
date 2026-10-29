"""Reads a run log (`python -m evalkit.run ... --log`) and prints its tables as Markdown.

    python -m evalkit.runlog work/smoke-x.json
"""

from __future__ import annotations

import json
import sys


def table(summary: dict) -> str:
    rows = ["| channel | requests (n) | wrongful allow | wrongful denial | wrong version | existence leaks | median ms | p95 ms |", "|---|---|---|---|---|---|---|---|"]
    for c, t in summary["by_channel"].items():
        lat = summary["latency_ms"][c]
        rows.append(f"| {c} | {t['n']} | {t['wrongful_allow']} | {t['wrongful_denial']} | {t['wrong_version']} | {t['existence_leaks']}/{t['existence_checked']} | {lat['median']} | {lat['p95']} |")
    a = summary["all"]
    rows.append(f"| **all** | {a['n']} | {a['wrongful_allow']} | {a['wrongful_denial']} | {a['wrong_version']} | {a['existence_leaks']}/{a['existence_checked']} | | |")
    return "\n".join(rows)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        s = json.load(open(path))["summary"]
        print(f"### {s['workflow']} ({s['repo']}{', baseline' if s['baseline'] else ''}), {s['requests']} requests, {len(s['by_class'])} person classes, {s['seconds']} s\n")
        print(table(s), "\n")
