# Solution — More than a Hack 2027, Phase 1

Two things live here:

- **`solution/`** — the real submission pipeline. It reads the tactical ATD, reconstructs the
  ball track, separates *possession* from *ball-in-play*, and decodes a persistent H/A/D sequence.
- **`evaluation/results/fallback_hmm.csv`** — a valid full-length stand-in built from the
  `baseline_raw.csv` that ships with the repository, so the code still runs on a machine without
  the licensed tracking bundle. It is not the competition entry.

The technical document deliverable is `docs/technical_report.md`.

---

## Status: the real pipeline has been run

The full tactical bundle is now extracted under `data/tactical/`:

```
data/tactical/BAR-ATH_tactical_atd.xml             59 KB    metadata
data/tactical/BAR-ATH_tactical_atd.txt            187 MB    tracking (199,319 frames)
data/tactical/BAR-ATH_tactical_smart_tagging.xml  255 KB    restarts, goals, phases
data/tactical/BAR-ATH_tactical_homography.csv      22 MB    read, not needed
```

It is licensed for the competition only and is **not** committed — `data/` is gitignored. The
generated submission is `evaluation/results/my_team.csv`:

- **346 possessions, mean 13.3 s**
- `D` on 27.6 % of graded frames; home/away 55.0/45.0 % of live frames
- 199,319 rows, validator `OK`

---

## Run it

```bash
python -m pip install -r requirements-solution.txt    # numpy/pandas pins used for this build

# real pipeline (needs data/tactical/*_atd.txt)
python solution/run.py --data-dir data --out evaluation/results/my_team.csv \
                       --cache analysis/atd_cache.npz

# stand-in from the shipped baseline (no match bundle needed)
python solution/run.py --fallback evaluation/results/baseline_raw.csv \
                       --out evaluation/results/fallback_hmm.csv

# check and score
python evaluation/validate_submission.py evaluation/results/my_team.csv
python evaluation/run_evaluation.py
```

Tested with CPython 3.11.9, `numpy==1.26.4`, `pandas==2.1.4`. The pipeline is deterministic: no
random sampling, no GPU, no learned weights, so the same inputs produce the same CSV.

Dependencies: `numpy` and `pandas` only, the same as the official scorer. On a full-size synthetic
file (199,319 frames, 46 slots) parsing takes ~16 s, features + Viterbi ~5 s, and a cached run
re-reads the arrays in under a second.

---

## What the model does

1. **Ingest** — metadata XML gives the home/away slot split; the raw ATD is streamed once into
   float32 arrays (positions and speeds for 46 slots + the ball).
2. **Ball reconstruction** — linear interpolation over occlusions up to 0.6 s; longer gaps stay
   missing rather than being invented.
3. **Possession evidence** — nearest-player geometry in metres, `P(home | ball visible)`,
   damped when nobody is within playing distance and when the ball is a pass in flight.
4. **Play-state evidence** — `P(ball dead)` from four independent signals: ball out of the pitch,
   ball stationary (speed *and* position spread), collective player speed against the match’s own
   tempo, and Smart Tagging set pieces/goals as restart markers. A restart tag that names a side
   also gives a short team prior for the first seconds of the next possession; throw-ins are `N/A`
   and give none.
5. **Sequence model** — sticky three-state HMM (`H`/`A`/`D`) decoded with Viterbi. The transition
   matrix encodes a 14 s mean possession, so a one-second tracking flicker cannot create a new
   possession. Evidence is averaged over 2 s first, centred so transitions stay sharp. Each block
   is decoded independently; a block opening at a kick-off starts on the tagged restart side, and
   the second half’s pre-kickoff frames start on `D`.
6. **Cleanup** — sub-0.25 s team runs and sub-0.4 s dead runs are absorbed into their neighbours;
   dead-ball plugs between the same team are bridged.
7. **Contract** — frames outside the two graded halves are `X`. The second half’s pre-kickoff
   block (published boundary, not a label) is forced `D`.

The key distinction from the baseline: **a missing ball is not a dead ball**. A gap with no ball
track is evidence about the *team* only weakly (hold the previous possession), while dead-ball
evidence comes from the players and the tags. `X` could be used inside an occlusion, but the
decoder’s held label is at least as good on `macroF1` and identical on `spellF1` after the
scorer’s forward fill.

---

## Public 5-minute result (sanity check only)

`my_team` (data-driven) and `fallback_hmm` vs the published baselines, on the 17-possession public
window:

| submission | macroF1 | spellF1 | possessions | mean dur | D F1 |
|---|---|---|---|---|---|
| `baseline_smooth125` | 0.5812 | **0.4242** | 16 | 18.5 s | — |
| **`my_team`** | **0.6658** | 0.4103 | 22 | 12.9 s | 0.445 |
| `fallback_hmm` | 0.6438 | 0.3429 | 25 | 10.6 s | 0.424 |
| `baseline_smooth50` | 0.6207 | 0.3256 | 26 | 10.8 s | — |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 | 7.6 s | — |
| `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0 s | — |

The real pipeline has the best `macroF1` and the best alive/dead F1 (0.920) on this window, and
beats every baseline except `smooth125` on `spellF1`. It does **not** beat `smooth125` on the
public five minutes — and that is expected, not a defect: `baseline_smooth125` is the documented
example of a configuration that wins this window (0.4242) and loses the full match (0.2242 vs
0.2433). The full-match structure is the better signal: 346 possessions at 13.3 s is close to the
public GT cadence of 17 at 14.2 s, while `smooth125` reports 398 possessions. No constant was
selected to maximise the public score; the dead-ball tempo reference was moved from the median
(which includes the stoppages being detected) to the live-play shoulder, and the dead-ball bias is
the football base rate.

The remaining weakness is short untagged stoppages: public dead-ball precision is 1.0 but recall
is 0.286. A five-second throw-in with the ball still tracked looks like a slow live phase; long
goal celebrations, tagged restarts and clear out-of-play spells are caught. The real pipeline has
the player-speed and ball-position evidence the stand-in lacks, but the dead-ball cues need more
work before the short spells stop costing `spellF1`.

---

## Reproduce / test

```bash
# the delivered submission, on the provided bundle
python solution/run.py --data-dir data --out evaluation/results/my_team.csv \
                       --cache analysis/atd_cache.npz --diagnostics analysis/real_diag.npz

# formats -> pipeline -> CSV, no match data needed
python tests/synthetic_match.py
python tests/benchmark_full.py
```

`tests/synthetic_match.py` fabricates the three provided file formats with a known possession
sequence, runs the full data-driven pipeline (metadata → ATD parser → features → HMM → CSV) and
asserts the recovered accuracy and the submission contract. A local development check,
`analysis/replay_gt.py` (not shipped in the public repository, because it reads the public ground
truth), drives the pipeline with the public window's own possession *cadence* (17 possessions,
real short spells and long stoppages) but synthetic player and ball positions: it recovers macroF1
0.963 / spellF1 0.743, which is a check of the decoder, not a validation on the match.

---

## File map

```
solution/
  config.py     every constant, with the football reason next to it
  data.py       metadata XML, raw EPTS text (streaming + cache), Smart Tagging, homography
  features.py   ball filling, possession geometry, dead-ball evidence
  sequence.py   transition matrix, Viterbi, run cleanup
  pipeline.py   end-to-end build + submission writer
  fallback.py   the stand-in from baseline_raw.csv
  run.py        CLI
tests/synthetic_match.py
docs/technical_report.md
requirements-solution.txt
```

---

## Running it on another match

1. Drop that match's files into `data/tactical/` (and optionally `data/broadcast/`).
2. `python solution/run.py --data-dir data --out evaluation/results/my_team.csv --diagnostics analysis/real_diag.npz`
3. Score the public window and inspect per-class F1 — use it to check alignment, not to pick
   variants.
4. `python evaluation/validate_submission.py evaluation/results/my_team.csv`

The constants most worth a second look on a new match are all in `solution/config.py` and are
marked: `tempo_percentile`, `live_mean_speed_mps`, `still_speed_mps`, the four `dead_*_weight`s,
the restart weights, and the two HMM mean durations. They are priors, not fitted parameters.

---

## What to publish (and what not to)

The submission repository is public and the match data is not redistributable.

Publish:

```
solution/                 the pipeline
tests/synthetic_match.py  correctness test, no match data needed
tests/benchmark_full.py   full-size throughput test
docs/technical_report.md  the technical document deliverable
SOLUTION.md
requirements-solution.txt
games.py, evaluation/     upstream MIT files the pipeline imports and the scorer
```

Do **not** publish:

```
data/                     licensed match assets (already gitignored)
analysis/                 development scripts; replay_gt.py reads the public ground truth
                          and is not needed to regenerate the submission
```

`analysis/replay_gt.py` is a local correctness check, not part of the delivered pipeline. Keeping
it out of the public repository avoids any appearance that the public five minutes were used to
validate or tune the model.

## Submission checklist

- [ ] `frame,label`, 199,319 rows, frames `0…199318`, labels `H/A/D/X`
- [ ] `python evaluation/validate_submission.py <file>.csv` → `OK`
- [ ] file named `<team_name>.csv`; technical document `<team_name>.pdf` or `.md`
- [ ] public repository contains no `data/` and reproduces the exact CSV with the command above
- [ ] team has 3–5 members (RULES.md eligibility; fewer than 3 is not eligible for the Innovation
      Challenge and Final Pitch)
- [ ] deadline **17 November 2026**; one Google Form response per team, last response counts
