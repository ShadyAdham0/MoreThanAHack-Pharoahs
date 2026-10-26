"""A stand-in submission built from the baseline_raw.csv that ships with the repository.

This is NOT the challenge solution. It never sees the tracking data. It exists so that the
repository produces a valid, self-contained file on a machine where the gated bundle is missing,
and so the sequence model can be exercised end-to-end.

The observation is the published baseline's per-frame nearest-player answer:
    H / A  : the ball was visible and a player from that team was nearest
    D      : the baseline saw no ball at all (NOT necessarily a dead ball)

That last conflation is the whole difficulty. A short D run inside one team's possession is an
occlusion and should be bridged; a long D run is more likely a stoppage. Generic priors encode
that, with no numbers taken from the public ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import encode_runs, sigmoid
from .pipeline import write_submission
from .sequence import decode

H, A, D, X = 0, 1, 2, 3

CODE_OF = {"H": H, "A": A, "D": D, "X": X}


def _raw_dead_probability(codes, cfg):
    """P(dead) from the baseline's D runs, with a generic length prior.

    A ball the baseline cannot see is weak evidence of a stoppage, and the evidence grows with
    how long it has been missing. The crossover is set at four seconds of football sense, not
    fitted: below that a blind spell is usually a crowded box, above it the ball is usually out.
    """
    n = len(codes)
    probability = np.full(n, 0.08, dtype=np.float64)   # a visible ball is usually live
    for start, stop, code in encode_runs(codes):
        if code != D:
            continue
        length = stop - start
        value = sigmoid((length - 100.0) / 55.0)        # 4 s, 2.2 s: steepness is generic
        probability[start:stop] = 0.10 + 0.75 * value
    return probability.astype(np.float32)


def build(baseline_csv, cfg=Config(), dead_scale=1.0):
    table = pd.read_csv(baseline_csv, dtype={"label": object})
    cleaned = np.array([str(value).strip().upper() for value in table["label"].values])
    unknown = sorted(set(cleaned) - set(CODE_OF))
    if unknown:
        raise ValueError("labels outside H/A/D/X: {}".format(unknown[:5]))
    codes = np.array([CODE_OF[value] for value in cleaned], dtype=np.int8)
    if len(codes) != cfg.n_frames:
        raise ValueError("{} rows, expected {}".format(len(codes), cfg.n_frames))

    dead = _raw_dead_probability(codes, cfg)
    dead = np.clip(dead * dead_scale, 1e-4, 1.0 - 1e-4)
    alive = 1.0 - dead

    # Team evidence: a visible ball and a nearest player is a decent but not certain team call;
    # a missing ball says nothing about the team.
    home_probability = np.where(codes == H, 0.76, np.where(codes == A, 0.24, 0.50))
    home_probability = np.clip(home_probability, 1e-4, 1.0 - 1e-4)

    emissions = np.empty((len(codes), 3), dtype=np.float64)
    emissions[:, H] = np.log(alive) + np.log(home_probability)
    emissions[:, A] = np.log(alive) + np.log(1.0 - home_probability)
    emissions[:, D] = np.log(dead)

    labels = np.full(cfg.n_frames, X, dtype=np.int8)
    for block, (first, last) in enumerate(cfg.graded):
        stop = last + 1
        kickoff = cfg.kickoffs[block] if block < len(cfg.kickoffs) else first
        pre_kickoff = max(0, min(kickoff - first, stop - first))
        prior = np.array([0.02, 0.02, 0.96]) if pre_kickoff > 0.5 * cfg.fps else \
            np.array([0.475, 0.475, 0.05])
        path = decode(emissions[first:stop], cfg, prior=prior)
        path[:pre_kickoff] = D
        labels[first:stop] = path

    return labels, {"baseline": codes, "dead_probability": dead}
