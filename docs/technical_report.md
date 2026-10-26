# Ball possession from automatic tracking — technical report

**Team Pharoahs** · More than a Hack 2027 · Phase 1 technical document

**Task.** One label per tactical-video frame: `H` (home in possession, ball alive), `A` (away in
possession, ball alive), `D` (dead ball), `X` (no answer / ungraded). 199,319 rows. Ranked by
`spellF1` over possession spells; `macroF1` is reported and breaks ties.

**Summary.** The system estimates two things independently — *which team is near the ball* and
*whether the ball is in play* — and fuses them with a sticky three-state HMM decoded by Viterbi.
The dead-ball signal is the part the reference baseline lacks, and the sequence model is what turns
a per-frame nearest-player signal into possessions rather than flicker.

---

## 1. Assets used

| asset | use |
|---|---|
| `BAR-ATH_tactical_atd.xml` | home/away assignment of the 46 track slots |
| `BAR-ATH_tactical_atd.txt` | ball and player positions + speeds, one record per video frame |
| `BAR-ATH_tactical_smart_tagging.xml` | restart markers: set pieces and goals |
| `BAR-ATH_tactical_homography.csv` | read, but not needed: the ATD is already in pitch coordinates and the tactical matrix is valid on 100 % of frames |
| tactical video | not read |
| broadcast video / ATD / homography | not used |

**Dropped, and why.** The broadcast feed is not synchronised to the tactical timeline and a third
of its frames have no usable homography; aligning it would need a dynamic-time-warping pass over
two noisy trajectories, and the expected gain in ball coverage did not justify the risk without a
full-match validation signal. The homography is redundant for possession because the ATD is
already normalised to the pitch; it would matter for a video-based ball detector, which was out of
scope for the same reason. Smart Tagging's phase-level `Team` tags are parsed but weighted zero:
phases are not possessions, and every throw-in — the restart where the side would help most — is
tagged `N/A`. The `Team` field on a kick-off or set-piece tag is different: it names the side that
*restarts*, which is where the next possession begins, so those tags supply a short team prior (2 s
for set pieces, 3 s for kick-offs, log-odds 0.8 and 1.8).

**Frame alignment.** Submission row = ATD `frame_id − 1`; Smart Tagging seconds × 25; the two
published graded blocks `[11185, 90651]` and `[110808, 190197]`; second-half kickoff at 113742.
The 2,934 frames between the second block's start and the kickoff are graded and the ball is not
in play, so they are forced `D`.

---

## 2. Method

### 2.1 Ball reconstruction

The ball is absent on 41 % of graded frames. Gaps up to 15 frames (0.6 s) are linearly
interpolated; longer gaps stay missing. The rule is deliberate: a one-second-old ball position
still describes the same ball, a five-second-old one has been kicked.

Ball speed is the reported speed where it exists, a position difference elsewhere, median-smoothed
over 9 frames. It is used for two things: damping team evidence while a pass is in flight, and
stationarity.

### 2.2 Possession evidence

For every frame with a ball, distances from every tracked player are computed in metres (the pitch
is 105.3 × 68 m, so normalised axes are scaled before comparison). The nearest player of each team
gives a signed margin `d_away − d_home`, mapped through a logistic with a 1.6 m scale:

```
P(home | ball visible) = sigmoid((d_away − d_home) / 1.6)
```

Two dampings pull the estimate to 50/50 where geometry is uninformative: when nobody is within
playing distance (the margin is scaled by `1 / (1 + (nearest/6 m)²)`), and when the ball is moving
faster than about 11 m/s (a pass belongs to nobody until it is received). Without the second
damping the nearest player to a ball in flight is frequently a defender it is passing.

**Restart team prior.** Geometry cannot see a possession that has not started yet: at a kick-off or
a free kick the ball sits still, the players are in their own half, and the first touch is a few
frames away. The tag names the side that will restart, so it contributes a short log-odds offset
(+home / −away) to the team evidence — 0.8 for 2 s after a set piece, 1.8 for 3 s after a kick-off.
It is evidence about the first seconds only; after that the tracking decides. Throw-ins are tagged
`N/A` and contribute nothing, so the prior is deliberately absent for the most common restart.

### 2.3 Dead-ball evidence

The baseline treats "the ball track vanished" as dead. That is wrong in both directions: the ball
vanishes in crowded boxes during play, and stays visible while it is placed for a throw-in. Dead
ball is therefore estimated separately from four signals, each smoothed over about a second
because a stoppage lasts seconds:

1. **Out of play** — ball position more than 0.75 m outside the pitch rectangle, in metres.
2. **Stationary ball** — low reported speed *or* low positional spread over 1 s. Position spread
   is the more robust of the two: tracker jitter on a still ball gives a non-zero speed.
3. **Collective player speed** — mean tracked speed, compared with *this match's own live-play
   tempo*, not an absolute value. The reference is the 55th percentile of the match's collective
   speed: the median includes the stoppages being detected and sits at a standing-player speed,
   while a stoppage is roughly half of the live-play shoulder. Without this correction the slow cue
   almost never fires, because the median of a stop-start match is itself a stopped speed.
4. **Restart tags** — a Smart Tagging set piece contributes dead ball from its start until the
   ball is actually moving for 5 consecutive frames; a goal contributes dead ball until the next
   kickoff tag. A tagged restart is strong evidence on its own; the other cues confirm it.

The four combine in log-odds with a negative bias, so a single quiet frame cannot declare a
stoppage:

```
logit P(dead) = −3.2 + 4.0·out + 2.6·still + 2.8·slow + 4.0·tag
```

### 2.4 Sequence model

Per-frame evidence is noisy at a one-second scale while possession changes at a ten-second scale.
Two mechanisms handle that:

- **Evidence smoothing** — the log-evidence columns are averaged over a centred 2 s window. A
  centred filter leaves a step at the step, rather than delaying it by half a window.
- **Sticky HMM** — states `H/A/D`; the transition matrix is built from a 14 s mean possession and a
  10 s mean stoppage, with 78 % of possession changes going directly to the other team and the
  rest via `D`. Viterbi finds the most likely path in `O(T·3²)`, renormalised each frame.

`D` is not a special case bolted on after team decoding: it is a state with its own duration and
its own evidence, so a stoppage interrupts a possession exactly once.

Each graded block is decoded independently, because the two halves are separate timelines. The
Viterbi start state is set from the block boundary: the second-half block opens 2,934 frames before
the kick-off, so that block starts on `D`; a block that opens at a kick-off starts on the side named
by the restart tag, with the tracking free to correct it within a second.

### 2.5 Cleanup and the X policy

Sub-0.25 s team runs and sub-0.4 s dead runs are absorbed into the longer neighbour (the shortest
public possession is 0.76 s, so 0.25 s is flicker, not football). Short dead-ball plugs sitting
between the *same* team are bridged into that possession.

Frames outside the graded blocks are `X`. Inside them the decoder emits a label rather than `X`:
on `macroF1` an abstention costs the same as a wrong guess, and on `spellF1` the scorer forward
fills it with the previous label — which is what the sticky decoder was already holding.

---

## 3. Possession rules implemented

- Possession changes only when the *other* team's control evidence is sustained: a single touch,
  a deflection or a ball in flight is not enough. The 2 s evidence window and the HMM transitions
  are the operational version of "controlled possession".
- A loose or contested ball keeps the previous team, matching the challenge's rule that possession
  stays with the last team that had it.
- `D` is a stoppage, never a possession. It is entered from ball-out-of-play, a stationary ball, a
  drop in collective tempo, or a tagged restart, and it ends when the ball moves again.
- Dead-ball *ownership* is never predicted; the label is `D` regardless of who will restart.
- A restart tag's side is a prior on the first seconds of the next possession, not a label: it can
  break the 50/50 tie at a kick-off, but it cannot hold a possession the tracking contradicts.
- The second-half pre-kickoff block is `D`; halftime, before kickoff and after the final whistle
  are `X`.

---

## 4. Training data

**None.** Every number in `solution/config.py` is a football prior or a measurement convention
(a 1.6 m control scale, 0.6 s interpolation limit, 14 s mean possession, 2 s evidence window,
0.8/1.8 restart log-odds). No model is fitted. The public five-minute ground truth is used only for
correctness checks — format, row alignment, the official score, and a sanity check on per-class
behaviour. It is not a training set, and no constant was fitted by optimising against it; every
constant has an independent football rationale next to it in `config.py`. No hand-labelled data of
any kind, from this game or any other, is used.

The stand-in path (`solution/fallback.py`) applies the same decoder to the repository's own
`baseline_raw.csv` with generic emission priors. It exists so the repository remains runnable on a
machine without the licensed bundle; it is not the competition entry.

---

## 5. Results and limitations

The data-driven pipeline was run on the provided tactical bundle (ATD tracking, metadata, Smart
Tagging, homography). Over the two graded halves it reports **346 possessions of mean 13.3 s**,
with `D` on **27.6 % of graded frames** and home/away split 55.0/45.0 % of live frames. A football
match is dead for roughly a quarter of its frames (20–30 %), so this is a plausible full-match
structure rather than a target.

On the public five minutes — a correctness check, not a tuning set:

| submission | macroF1 | spellF1 | poss | mean dur | D F1 |
|---|---|---|---|---|---|
| `baseline_smooth125` | 0.5812 | **0.4242** | 16 | 18.5 s | — |
| **`Pharoahs` (data-driven)** | **0.6658** | 0.4103 | 22 | 12.9 s | 0.445 |
| `fallback_hmm` | 0.6438 | 0.3429 | 25 | 10.6 s | 0.424 |
| `baseline_smooth50` | 0.6207 | 0.3256 | 26 | 10.8 s | — |

The data-driven submission has the best `macroF1` of any file on the board and the best
alive/dead F1 (0.920); its team accuracy on live frames is 0.847. It does **not** beat
`smooth125` on this window, which is the repository's documented example of a public-window winner
that loses the full match. The full-match possession count is the better structural signal:
346 possessions at 13.3 s is close to the public GT cadence of 17 possessions at 14.2 s, while
`smooth125` reports 430 possessions over the full match. No constant was selected to maximise the
public score; the tempo percentile came from the stop-start bias in the median, and the dead-ball
bias is the football base rate.

Known limitations:

- **Short untagged stoppages are under-called.** Public dead-ball precision is 1.0 but recall is
  0.286: a 5 s throw-in with the ball still tracked looks like a slow live phase to the current
  cues. Long goal celebrations, tagged restarts and clear out-of-play spells are caught; the
  four-and-a-half-second spell at 12,851 is missed entirely. This is the largest remaining source
  of lost `spellF1`.
- **Ball gaps > 0.6 s** are not reconstructed. A turnover inside a long occlusion is missed.
- **Dead-ball detection is indirect.** Referee decisions (fouls, offside) are not observable; they
  are inferred from the subsequent stoppage. A team that stands still with the ball during live
  play can be called dead.
- **Identity switches** in the 46 tracklets are not corrected. Possession is collective, so a
  single swapped slot rarely changes the team, but a swapped *ball* observation can.
- **The HMM's geometric durations** under-price very long possessions, though Viterbi recovers
  them when the evidence is clear.
- **The broadcast feed is unused**, leaving its extra ball coverage on the table.

The exposed knobs are the four dead-ball weights, `tempo_percentile` and the two HMM mean
durations in `solution/config.py`; the first thing to inspect on a new match is dead-ball
precision/recall and the distribution of `P(dead)` against player speed.

---

## 6. Reproduction

```bash
pip install -r requirements-solution.txt    # pinned numpy/pandas

# data-driven submission
python solution/run.py --data-dir data --out evaluation/results/Pharoahs.csv \
                       --cache analysis/atd_cache.npz

# correctness test without the match bundle
python tests/synthetic_match.py

# format check and public score
python evaluation/validate_submission.py evaluation/results/Pharoahs.csv
python evaluation/run_evaluation.py
```

The pipeline is deterministic: no randomness, no learned weights, no network access.

---

## 7. Note on the delivered data bundle

The tactical bundle was received and extracted locally: `BAR-ATH_tactical_atd.txt` (tracking),
`_atd.xml` (metadata), `_smart_tagging.xml` and `_homography.csv`. It is licensed for the
competition and is **not** redistributed with this repository; `data/` is gitignored. The
submission in `evaluation/results/` is generated by the command in §6, and the stand-in path
(`--fallback`) remains available on a machine without the bundle.
