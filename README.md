# StateGate

Workflow-state-based access control for agentic retrieval-augmented generation. Every document in the data is synthetic.

Article: *StateGate: Workflow-State-Based Access Control for Agentic Retrieval-Augmented Generation in Public Organizations*, IEEE Access. Authors: Evan Oktavianus, Dana Indra Sensuse, Imairi Eitiveni (Universitas Indonesia).

## Where things are

| Folder | What it holds |
|---|---|
| `src/docflow/` | the StateGate engine (Python package) |
| `workflows/` | the eight workflow files (YAML), and `language_reference.txt`, every word the workflow language allows |
| `scenarios/` | the tables the independent oracle is checked against |
| `eval/evalkit/` | the evaluation code |
| `eval/data/documents/` | 1,024 synthetic documents (2,660 versions), their passages and the vectors of the passages in parts |
| `eval/data/same_topic_memos/` | 240 same-topic memos |
| `eval/data/regulations/` | 300 synthetic regulations and their records, their passages and the vectors in parts |
| `eval/data/questions/` | 1,076 questions with three paraphrases each |
| `eval/data/injections/` | 218 prompt-injection prompts |
| `eval/data/SYNTHETIC_DATA.md` | how the data was generated: models, acceptance criteria, prompts |
| `eval/results/runs/e1`, `e2` | request grids with and without the access check, mutation results |
| `eval/results/runs/e4` | property checks and planted engine bugs |
| `eval/results/runs/e5` | retrieval with and without the filter |
| `eval/results/runs/e6` | agent runs, context-leak check, guidance answers and judge verdicts, human check |
| `eval/results/runs/e7` | prompt injection |
| `eval/results/runs/bench`, `static_baseline`, `exhaustive`, `random_paths` | overhead, static policy baseline, surviving-mutant check, random-walk baselines |
| `tests/` | the test suite |

## Setup

Python 3.11 or later. Run everything from the repository root. PostgreSQL 17 or 18 with pgvector is optional (`DATABASE_URL`; `DOCFLOW_STORE=memory` forces the in-memory store). The stored E5 results use the in-memory store. With PostgreSQL, search ranks per unit and merges the lists by reciprocal rank fusion, so E5 recall differs there; the access decisions do not depend on the store.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[eval]'
export PYTHONPATH=src:eval

# join the parts of the two vector files (510 MB), checked by sha256
python3 scripts/join_vectors.py
gunzip -k eval/data/documents/chunks.jsonl.gz

python3 -m pytest -q
```

## Results, commands and result files

No row needs a model call or a key. Output of the commands goes to `work/`.

| Paper | Command | Stored result |
|---|---|---|
| E1–E3, mutation (Tables IV–VI) | `python -m evalkit.report --e2 1 --logs eval/results/runs/e2 --md work/grid.md --json work/grid.json` | `runs/e1/`, `runs/e2/`, `survivor_verdicts.json` |
| same, run again (about 50 minutes on 8 cores) | `python -m evalkit.reproduce --only grid,mutation` | the same |
| Coverage of each workflow (Tables IV, V) | `python -m evalkit.paths --json work/paths.json` | `eval/results/paths.json` |
| Random-walk baselines (Table V) | `python -m evalkit.random_paths_report eval/results/runs/random_paths --md work/random.md` | `runs/random_paths/` |
| same, run again | `python -m evalkit.random_paths <workflow> --mut eval/results/runs/e2/mut-<workflow>.json --out work/random-paths-<workflow>.json --seeds 10` | `runs/random_paths/` |
| Surviving mutants: 106 equivalent, 5 beyond the test cast, 4 gaps | `python -m evalkit.exhaustive <workflow> --verdicts eval/results/survivor_verdicts.json --mut eval/results/runs/e2/mut-<workflow>.json --out work/exhaustive-<workflow>.json` | `runs/exhaustive/`, `survivor_verdicts.json` |
| E4, property checks and planted bugs (Table VII; rank identity 67 = P7 63 + P7b 4) | `python -m evalkit.reproduce --show`; run again: `python -m evalkit.reproduce --only probes` and `python -m evalkit.system_mutants <workflow> ...` (all eight workflows) | `runs/e4/` |
| E5 with the filter (Table VIII) | `python -m evalkit.retrieval_utility --show eval/results/runs/e5/utility.json`; run again (about 20 minutes on 11 cores): `python -m evalkit.retrieval_utility --json work/utility.json` | `runs/e5/utility.json` |
| E5 without the filter (Table VIII) | `python -m evalkit.retrieval_utility --show eval/results/runs/e5/utility-nofilter.json`; run again: `python -m evalkit.retrieval_utility --no-filter --json work/utility-nofilter.json` | `runs/e5/utility-nofilter.json` |
| Static policy baseline (Table V) | `python -m evalkit.static_baseline --out work/static.json` | `runs/static_baseline/static.json` |
| Overhead of the access decision (timings: Intel Core i7-9750H, 16 GB, in-memory store; machine-dependent) | `python -m evalkit.bench --out work/bench.json` | `runs/bench/bench.json` |
| E6, context leak | `python -m evalkit.e6.analysis` | `runs/e6/context_leaks.json.gz`, `runs/e6/results.jsonl`, `runs/e6/plan.json` |
| E6, stored tool calls run again on the engine | `python -m evalkit.reproduce --only e6-tools`; `python -m evalkit.e6.context_leaks --results eval/results/runs/e6/results.jsonl --out work/e6` | the same |
| E6, guidance | `python -m evalkit.e6.guidance report --answers eval/results/runs/e6/guidance/answers.jsonl.gz --verdicts eval/results/runs/e6/guidance/verdicts_glm.jsonl.gz --verdicts2 eval/results/runs/e6/guidance/verdicts_muse.jsonl.gz` | `runs/e6/guidance/` |
| E6, guidance judges against the author's labels | `python -m evalkit.e6.guidance_human score eval/results/runs/e6/guidance/human_check/labels.csv --sample eval/results/runs/e6/guidance/human_check/sample.json` | `runs/e6/guidance/human_check/` |
| E7, prompt injection (Table IX) | `python -m evalkit.e7.score eval/results/runs/e7/results.jsonl.gz work/e7` | `runs/e7/`, `eval/data/injections/prompts.json` |

## Runs with a model

These need `OPENROUTER_API_KEY` in the environment. Agent model `z-ai/glm-5.3-flash` (and `qwen/qwen3.7-flash` on a subset), embeddings `openai/text-embedding-3-small`, both through OpenRouter. Judges: `z-ai/glm-5.3-flash` and `meta/muse-spark-1.3-contributor`.

```
export OPENROUTER_API_KEY=...

python -m evalkit.e6.runner --plan eval/results/runs/e6/plan.json --out work/e6 --cap 10      # about USD 9 per 1,900 runs; add --limit 30 for a few
python -m evalkit.e7.run --out work/e7 --cap 5                                                # 436 runs, about USD 0.4
python -m evalkit.e7.score work/e7/results.jsonl work/e7
python -m evalkit.e6.guidance judge --answers eval/results/runs/e6/guidance/answers.jsonl.gz --out work/verdicts.jsonl.gz --model <id> --workers 100 --max-tokens 4000   # about USD 0.75 for both judges (6000 for Muse)
python -m evalkit.reproduce --only e6,e7 --scripted                                           # fake model: no key, no cost
```

## Tests

```
python3 -m pytest -q
```

About 12 minutes with the data files, about a minute without them. Tests that need the data or PostgreSQL skip with a message. `scripts/pg_password_verifier.py` makes the password verifier for the PostgreSQL role (see `src/docflow/stores/setup_postgres.sql`).

## Licence

Software (`src/`, `eval/evalkit/`, `scripts/`, `tests/`, `workflows/`, `scenarios/`): Apache License 2.0 (`LICENSE`). Data and results (`eval/data/`, `eval/results/`): Creative Commons Attribution 4.0 International (`LICENSE-DATA.txt`); credit the authors (`NOTICE`). All documents in the data are synthetic; institution names in the text do not describe any real document, rule or position.
