"""System mutants: a small, separate set of faults put into a COPY of `src/` in a temporary directory, never into the repository.

Not the Pretschner method (that mutates the policy); a check that the grid and the probes notice a faulty engine. Each patch must match exactly once.

    python -m evalkit.system_mutants [workflow ...]
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MUTANTS = [
    ("stale_decision_cache", "src/docflow/stores/world.py",
     "        doc = self.documents[doc_id]\n        return readable_version(doc.timeline, self.person(user, doc.workspace), self.context(doc), for_search=for_search)\n",
     "        cache = self.__dict__.setdefault('_stale', {})\n        key = (doc_id, user, for_search)\n        if key not in cache:\n            doc = self.documents[doc_id]\n            cache[key] = readable_version(doc.timeline, self.person(user, doc.workspace), self.context(doc), for_search=for_search)\n        return cache[key]\n"),
    ("skip_who_and_requirement_check", "src/docflow/access_control/access_policy_engine/machine.py",
     "        self.check(transition, ctx or Context())\n        return transition.to\n",
     "        return transition.to\n"),
    ("accept_action_from_any_state", "src/docflow/access_control/access_policy_engine/machine.py",
     "        transition = self.states[state].get(action)\n        if transition is None:\n            raise Refused(\"not_accepted\", action=action, state=state)\n        self.check(",
     "        transition = self.states[state].get(action) or next((s[action] for s in self.states.values() if action in s), None)\n        if transition is None:\n            raise Refused(\"not_accepted\", action=action, state=state)\n        self.check("),
    ("serve_latest_not_version_as_of_state", "src/docflow/access_control/access_policy_engine/access.py",
     "        case (\"from\", state):\n            return timeline.version_at(state)\n",
     "        case (\"from\", state):\n            return timeline.version\n"),
    ("pooled_keyword_statistics", "src/docflow/stores/search_index.py",
     "        n, k1, b = len(rows), 1.5, 0.75\n        df = {t: sum(1 for i in rows if t in self._tf[i]) for t in terms}\n        avg = sum(self._len[i] for i in rows) / max(n, 1)\n",
     "        n, k1, b = len(self._len), 1.5, 0.75\n        df = {t: self._df.get(t, 0) for t in terms}\n        avg = self._avg\n"),
]


def apply(name: str, dest: Path) -> None:
    for n, rel, old, new in MUTANTS:
        if n == name:
            f = dest / rel
            text = f.read_text(encoding="utf-8")
            if text.count(old) != 1:
                raise SystemExit(f"{name}: the patch matches {text.count(old)} times in {rel}, not once")
            f.write_text(text.replace(old, new), encoding="utf-8")
            return
    raise KeyError(name)


def run_grid(workflow: str, name: str | None) -> tuple[int, str]:
    """(disagreements, last line) of the grid run on a copy of src with the mutant `name` (or on the clean copy)."""
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(ROOT / "src", Path(d) / "src", ignore=shutil.ignore_patterns("__pycache__"))
        if name:
            apply(name, Path(d))
        env = {**os.environ, "PYTHONPATH": f"{d}/src:{ROOT}/.vendor:{ROOT}/eval:" + os.environ.get("PYTHONPATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run([sys.executable, "-m", "evalkit.run", workflow, "--brief", "--no-http"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        m = re.search(r"(\d+) disagreements", out.stdout)
        return (int(m.group(1)) if m else -1), (out.stdout.strip().splitlines() or [""])[-1] if m else out.stderr.strip().splitlines()[-1][:200]


def run_probe7(workflow: str, name: str | None) -> str:
    """The status of probe P7 (a hidden document must not change a ranking) on a copy of src with the mutant."""
    code = "import sys; from evalkit import probes; r = probes.p7_hidden_changes_ranking(sys.argv[1]); r2 = probes.p7b_cross_unit_ranking(); print('P7', r.status, r.counts, '| P7b', r2.status, r2.counts)"
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(ROOT / "src", Path(d) / "src", ignore=shutil.ignore_patterns("__pycache__"))
        if name:
            apply(name, Path(d))
        env = {**os.environ, "PYTHONPATH": f"{d}/src:{ROOT}/.vendor:{ROOT}/eval:" + os.environ.get("PYTHONPATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run([sys.executable, "-W", "ignore", "-c", code, workflow], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        return (out.stdout.strip().splitlines() or ["P7 crashed " + out.stderr.strip()[-100:]])[-1]


def main(argv: list[str]) -> None:
    workflows = argv or ["card1_policy_analysis_simple"]
    for wf in workflows:
        n, line = run_grid(wf, None)
        print(f"{wf}: clean copy -> {n} disagreements")
        for name, *_ in MUTANTS:
            n, line = run_grid(wf, name)
            verdict = "survived" if n == 0 else ("CRASHED (the run itself fails: " + line + ")" if n < 0 else "KILLED")
            print(f"{wf}: {name} -> {verdict}" + (f" ({n} disagreements)" if n >= 0 else "") + f"; probe {run_probe7(wf, name)}")


if __name__ == "__main__":
    main(sys.argv[1:])
