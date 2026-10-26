"""The sequence model: a sticky three-state HMM over H / A / D, decoded with Viterbi.

Why a sequence model at all. Per-frame nearest-player output is a jitter machine: 131 possessions
from five minutes of football. The metric counts possessions, so the decoder must believe in
persistence. A transition matrix with a 14-second mean possession does that explicitly, and lets
weak evidence accumulate before a possession is declared lost.

Durations are geometric under a plain HMM, which is not exactly football (possessions are not
memoryless), but the error it makes is to under-price very long spells, and those are rare enough
that Viterbi still finds them when the evidence is clear.
"""

from __future__ import annotations

import numpy as np

H, A, D = 0, 1, 2


def transition_matrix(mean_possession_frames, mean_dead_frames, direct_switch_share):
    """3x3 row-stochastic P(next state | current state), from mean spell lengths.

    A possession ends either directly into the other team or into a stoppage; the split is the
    share of possession changes that do not pass through D.
    """
    p_leave_possession = 1.0 / float(mean_possession_frames)
    p_leave_dead = 1.0 / float(mean_dead_frames)

    leave_to_other = p_leave_possession * direct_switch_share
    leave_to_dead = p_leave_possession * (1.0 - direct_switch_share)

    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[H, H] = 1.0 - p_leave_possession
    matrix[H, A] = leave_to_other
    matrix[H, D] = leave_to_dead

    matrix[A, A] = 1.0 - p_leave_possession
    matrix[A, H] = leave_to_other
    matrix[A, D] = leave_to_dead

    matrix[D, D] = 1.0 - p_leave_dead
    matrix[D, H] = p_leave_dead / 2.0
    matrix[D, A] = p_leave_dead / 2.0
    return matrix


def viterbi(log_emissions, log_transition, log_prior):
    """Most likely state path under per-frame log emissions. O(T * 3 * 3).

    Implemented in log space and renormalised every frame so long matches cannot underflow.
    Returns int8 codes H/A/D.
    """
    n_frames, n_states = log_emissions.shape
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    score = log_prior.astype(np.float64) + log_emissions[0].astype(np.float64)

    for frame in range(1, n_frames):
        candidates = score[:, None] + log_transition      # (previous, current)
        back[frame] = np.argmax(candidates, axis=0)
        score = np.max(candidates, axis=0) + log_emissions[frame]
        score -= score.max()

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def merge_short_runs(codes, min_team, min_dead):
    """Absorb runs shorter than the football minimum into the longer neighbour.

    The GT's shortest public possession is 19 frames (0.76 s), so a 5-frame 'possession' is a
    tracking flicker, not a possession. Merging is length-preserving and deterministic (ties go to
    the earlier neighbour).
    """
    codes = codes.copy()
    for _ in range(4 * len(codes) // max(1, min(min_team, min_dead)) + 16):
        runs = []
        start = 0
        for frame in range(1, len(codes) + 1):
            if frame == len(codes) or codes[frame] != codes[start]:
                runs.append((start, frame, int(codes[start])))
                start = frame
        if len(runs) <= 1:
            break

        changed = False
        for index, (start, stop, code) in enumerate(runs):
            minimum = min_dead if code == D else min_team
            if stop - start >= minimum:
                continue
            left = runs[index - 1] if index > 0 else None
            right = runs[index + 1] if index + 1 < len(runs) else None
            candidates = [run for run in (left, right) if run is not None]
            if not candidates:
                continue
            target = max(candidates, key=lambda run: (run[1] - run[0], -run[0]))
            codes[start:stop] = target[2]
            changed = True
            break
        if not changed:
            break
    return codes


def remove_dead_plugs(codes, max_plug):
    """Replace short D runs that sit between the *same* team with that team.

    'H H D H H' is one possession the ball tracker blinked out of, not two possessions. Long D runs
    are never removed, whatever their neighbours.
    """
    codes = codes.copy()
    for start, stop, code in _runs(codes):
        if code != D:
            continue
        if stop - start > max_plug:
            continue
        left = codes[start - 1] if start > 0 else None
        right = codes[stop] if stop < len(codes) else None
        if left is not None and left == right and left != D:
            codes[start:stop] = left
    return codes


def _runs(codes):
    start = 0
    for frame in range(1, len(codes) + 1):
        if frame == len(codes) or codes[frame] != codes[start]:
            yield start, frame, int(codes[start])
            start = frame


def smooth_evidence(emissions, window):
    """Centred rolling mean over the log-evidence columns.

    A single frame's nearest-player is a weak observation; a second of them is not. Averaging in
    log space keeps the result a likelihood, and centring means a step change is still located at
    the step rather than half a window later.
    """
    if window <= 1:
        return emissions
    half = window // 2
    padded = np.pad(emissions, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.nanmean(windows, axis=-1)


def decode(emissions, cfg, prior=None):
    """Viterbi + cleanup. `emissions` is (T, 3) log-likelihoods for H/A/D."""
    log_transition = np.log(transition_matrix(
        cfg.mean_possession_frames, cfg.mean_dead_frames, cfg.direct_switch_share))
    probabilities = np.array([0.45, 0.45, 0.10]) if prior is None else np.asarray(prior)
    log_prior = np.log(probabilities / probabilities.sum())
    emissions = smooth_evidence(emissions, cfg.evidence_smooth_frames)
    path = viterbi(emissions, log_transition, log_prior)
    path = merge_short_runs(path, cfg.min_team_run, cfg.min_dead_run)
    path = remove_dead_plugs(path, max_plug=cfg.mean_dead_frames // 4)
    # Merging can expose a new short run; one final pass is cheap and deterministic.
    return merge_short_runs(path, cfg.min_team_run, cfg.min_dead_run)
