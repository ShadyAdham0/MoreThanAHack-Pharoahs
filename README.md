# More than a Hack 2027 — Phase 1 submission

Per-frame football possession for the provided Barcelona–Athletic Club tactical match: one label
per frame (`H` home in possession, `A` away in possession, `D` dead ball, `X` no answer / ungraded),
199,319 rows, ranked on the full-match possession-spell F1.

## What is here

| path | what it is |
|---|---|
| `SOLUTION.md` | the approach, the full-match/public results, the run commands, and the publication checklist |
| `docs/technical_report.md` | the technical document deliverable |
| `solution/` | the algorithm: metadata + ATD parsing, ball reconstruction, dead-ball evidence, sticky HMM/Viterbi, cleanup, submission writer |
| `tests/synthetic_match.py` | end-to-end correctness test; fabricates the three input formats, so it needs no match data |
| `tests/benchmark_full.py` | full-size (199,319-frame) throughput and memory smoke test |
| `evaluation/` | the official scorer, the public five-minute ground truth, and the repository's baseline files |
| `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, … | the upstream challenge documents, included for context |

## Reproduce the submission

```bash
python -m pip install -r requirements-solution.txt

# the licensed tactical bundle is not distributed here; get it through enrolment (DATA.md)
# and unpack it under data/tactical/ (BAR-ATH_tactical_atd.txt, _atd.xml,
# _smart_tagging.xml, _homography.csv)

python solution/run.py --data-dir data --out evaluation/results/Pharoahs.csv \
                       --cache analysis/atd_cache.npz \
                       --diagnostics analysis/real_diag.npz

python evaluation/validate_submission.py evaluation/results/Pharoahs.csv
python evaluation/run_evaluation.py
```

The first run parses the 187 MB ATD file (about 16 s on the reference machine) and caches the
arrays under `analysis/`; later runs take a few seconds. The pipeline is deterministic: no
randomness, no learned weights, no network access, and only `numpy` and `pandas` as dependencies.
The delivered CSV is not committed because `evaluation/results/*.csv` is gitignored; the command
above regenerates it byte-for-byte on the pinned versions.

## Compliance notes

- The labels are produced by the algorithm in `solution/`; there is no hand annotation.
- The public five-minute ground truth is used only to check format, frame alignment and the
  official score. It is not a training set, and no constant was fitted to it. Every constant in
  `solution/config.py` carries its football rationale next to it.
- The private full-match ground truth is not used, and nothing is derived from it.
- The licensed match bundle is **not** redistributed. `data/` is gitignored except for its README.

## Team

**Pharoahs.** The technical document is submitted as `Pharoahs.pdf`; the possession file is
`Pharoahs.csv`. The competition rules require a team of 3–5 members for the Innovation Challenge
and Final Pitch, and the Phase 1 deadline is **17 November 2026**.
