"""Report of the random-path baseline: for each workflow, seeds 1 to 10 (the files of `evalkit.random_paths`, merged), the mutation score with and without the survivors judged
equivalent, against the generated suite and against the request-pool baseline (random requests from the same pool), on the same mutant samples.

    python -m evalkit.random_paths_report eval/results/runs/random_paths --md eval/results/runs/random_paths/report.md
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

from .report import ITERATION_1, ITERATION_2, adjusted

SEEDS = range(1, 11)
WFS = [ITERATION_1, *ITERATION_2, "knowledge_asset", "card7_official_document", "regulation_single"]
SHORT = {"card2_policy_analysis_individual": "card2", "regulation_single": "regulation"}


def merged(folder: Path, wf: str) -> dict:
    out = None
    for f in sorted(glob.glob(str(folder / f"random-paths-{wf}.json")) + glob.glob(str(folder / f"seed-*.json"))):
        d = json.loads(Path(f).read_text())
        if d["workflow"] != wf:
            continue
        if out is None:
            out = {"workflow": wf, "judged": d["judged"], "equivalent": d["equivalent"], "seeds": {}}
        out["seeds"].update(d["seeds"])
    return out


def main(argv: list[str]) -> None:
    folder = Path(argv[0])
    args = dict(zip(argv[1::2], argv[2::2]))
    root = Path(__file__).resolve().parents[2]
    rows, lines = [], []
    head = ["workflow", "mutants judged", "judged equivalent", "seeds", "generated suite", "generated suite without equivalent", "random paths: mean (range)", "random paths without equivalent: mean (range)",
            "random requests from the same pool (30 repetitions): mean", "steps run / steps of the generated set (mean over seeds)"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for wf in WFS:
        m = merged(folder, wf)
        d = "e1" if wf == ITERATION_1 else "e2"
        mut = json.loads((root / "eval" / "results" / "runs" / d / f"mut-{wf}.json").read_text())
        adj = adjusted(mut, wf)
        have = [s for s in SEEDS if str(s) in m["seeds"]]
        n, eq = len(m["judged"]), set(m["equivalent"])
        scores = [len(m["seeds"][str(s)]["killed"]) / n for s in have]
        scores_ne = [len(m["seeds"][str(s)]["killed"]) / (n - len(eq)) for s in have]
        steps = statistics.mean(m["seeds"][str(s)]["steps"] / max(1, m["seeds"][str(s)]["steps_wanted"]) for s in have)
        f = lambda v: f"{statistics.mean(v):.3f} ({min(v):.3f} to {max(v):.3f})"
        lines.append(f"| {wf} | {n} | {len(eq)} | {len(have)} (1 to 10: {'all' if len(have) == 10 else 'MISSING'}) | {adj['killed']}/{n} = {adj['score']} | {adj['killed']}/{adj['judged_without_equivalent']} = {adj['score_without_equivalent']} | {f(scores)} | {f(scores_ne)} | {adj['baseline_mean']} ({adj['baseline_mean_without_equivalent']} without equivalent) | {steps:.0%} |")
        rows.append({"workflow": wf, "seeds": have, "random_paths": scores, "random_paths_without_equivalent": scores_ne, "suite": adj["score"], "suite_without_equivalent": adj["score_without_equivalent"]})
    text = "# Random-path baseline (seeds 1 to 10)\n\nRandom walks from the initial state, as many paths as the generated set and, path by path, as many steps (a walk stops early only when no action is accepted from where it is), attribute set drawn at random per path; requests as usual (one person per class, every channel); the same mutant samples as the generated suite (60 per workflow, all 33 for card1). A mutant judged equivalent was not run again and counts as not killed. \"Without equivalent\" leaves those out of the denominator.\n\n" + "\n".join(lines) + "\n\nThe request-pool baseline draws requests at random from the pool of all cast members, with the size of the suite; it is labelled \"random requests from the same pool\".\n"
    if "--md" in args:
        Path(args["--md"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--md"]).write_text(text)
    if "--json" in args:
        Path(args["--json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--json"]).write_text(json.dumps(rows, indent=1))
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
