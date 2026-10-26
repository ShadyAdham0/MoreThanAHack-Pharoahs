"""Per-frame evidence: who is near the ball, and is the ball actually in play.

There are two independent questions, and keeping them independent is the central design choice.
The baseline answers "who?" and uses "the ball track vanished" as its answer to "dead?", which
conflates an occlusion with a stoppage. Here the two are estimated separately and fused by the
sequence model:

  * possession geometry  -> P(home has the ball | ball visible)
  * play-state evidence  -> P(ball dead | ball / players / tags)

A frame where the ball is simply not tracked has no opinion about the team, but still has an
opinion about whether the players are playing.
"""

from __future__ import annotations

import warnings

import numpy as np

from .data import HOME, AWAY

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")


# ---------------------------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------------------------

def rolling_nanmedian(values, window):
    """Centred rolling median that ignores NaNs (edge-padded)."""
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with np.errstate(all="ignore"):
        out = np.nanmedian(windows, axis=1)
    return out.astype(np.float32)


def rolling_nanmean(values, window):
    """Centred rolling mean that ignores NaNs (edge-padded)."""
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with np.errstate(all="ignore"):
        out = np.nanmean(windows, axis=1)
    return out.astype(np.float32)


def fill_finite(values):
    """Fill NaNs by linear interpolation, clamping at the ends. Keeps length."""
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if good.all() or not good.any():
        return values.astype(np.float32)
    index = np.arange(len(values))
    return np.interp(index, index[good], values[good]).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def encode_runs(codes):
    """[(start, stop, code)] with stop exclusive."""
    if not len(codes):
        return []
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    return [(int(bounds[i]), int(bounds[i + 1]), codes[bounds[i]]) for i in range(len(bounds) - 1)]


# ---------------------------------------------------------------------------------------------
# ball
# ---------------------------------------------------------------------------------------------

def fill_ball(bx, by, limit):
    """Linear interpolation across short gaps only.

    Returns (bx, by, filled_mask). A gap is the run of missing frames between two observations;
    gaps longer than `limit` stay NaN because a five-second-old ball position is not evidence
    about where the ball is now.
    """
    n = len(bx)
    observed = np.isfinite(bx) & np.isfinite(by)
    if observed.sum() < 2:
        return bx.copy(), by.copy(), observed

    index = np.arange(n)
    previous = np.maximum.accumulate(np.where(observed, index, -1))
    following = np.minimum.accumulate(np.where(observed, index, n)[::-1])[::-1]
    gap = following - previous - 1
    fillable = (~observed) & (gap >= 0) & (gap <= limit) & (previous >= 0) & (following < n)

    fx, fy = bx.astype(np.float64).copy(), by.astype(np.float64).copy()
    for out, source in ((fx, bx), (fy, by)):
        interpolated = np.interp(index, index[observed], source[observed])
        out[fillable] = interpolated[fillable]
    return fx.astype(np.float32), fy.astype(np.float32), observed | fillable


def ball_speed(bx, by, bs, pitch, fps, smooth):
    """Ball speed in m/s: reported speed where available, differenced positions elsewhere."""
    length, width = pitch

    dx = np.diff(bx.astype(np.float64)) * length
    dy = np.diff(by.astype(np.float64)) * width
    differenced = np.full(len(bx), np.nan, dtype=np.float64)
    differenced[1:] = np.sqrt(dx * dx + dy * dy) * fps

    speed = bs.astype(np.float64).copy()
    missing = ~np.isfinite(speed)
    speed[missing] = differenced[missing]
    # A differenced speed across a gap is a long straight line, not a ball: leave those NaN.
    return rolling_nanmedian(speed.astype(np.float32), smooth)


# ---------------------------------------------------------------------------------------------
# possession geometry
# ---------------------------------------------------------------------------------------------

def player_distances(px, py, bx, by, pitch):
    """(n_frames, n_slots) distance in metres from every player to the ball. NaN if either absent."""
    length, width = pitch
    dx = (px - bx[:, None]).astype(np.float32) * length
    dy = (py - by[:, None]).astype(np.float32) * width
    return np.sqrt(dx * dx + dy * dy)  # NaN propagates; callers use nan-safe reductions


def possession_probability(px, py, bx, by, bs, team_of_slot, cfg):
    """P(home has the ball) in [0, 1], 0.5 where the geometry says nothing.

    Nearest-player geometry, damped twice:
      * when nobody is near the ball (a long clearance, a loose ball in flight);
      * when the ball is travelling fast (a pass belongs to nobody until it is received).
    """
    distances = player_distances(px, py, bx, by, cfg.pitch)
    home_columns = np.flatnonzero(team_of_slot == HOME)
    away_columns = np.flatnonzero(team_of_slot == AWAY)

    with np.errstate(all="ignore"):
        home = np.nanmin(distances[:, home_columns], axis=1)
        away = np.nanmin(distances[:, away_columns], axis=1)

    margin = away.astype(np.float64) - home.astype(np.float64)
    probability = np.where(np.isfinite(margin), sigmoid(margin / cfg.team_sigma_m), 0.5)

    nearest = np.fmin(home, away).astype(np.float64)
    geometry_confidence = 1.0 / (1.0 + (nearest / 6.0) ** 2)
    geometry_confidence = np.where(np.isfinite(geometry_confidence), geometry_confidence, 0.0)

    speed = np.nan_to_num(bs.astype(np.float64), nan=0.0)
    pass_damping = 1.0 / (1.0 + (speed / cfg.pass_speed_mps) ** 4)

    probability = 0.5 + (probability - 0.5) * geometry_confidence * pass_damping
    probability = np.where(np.isfinite(probability), probability, 0.5)
    return np.clip(probability, 1e-4, 1.0 - 1e-4).astype(np.float32), np.isfinite(home)


# ---------------------------------------------------------------------------------------------
# play-state evidence
# ---------------------------------------------------------------------------------------------

def _moving_from(ball_movement, start, stop, threshold=2.0, need=5):
    """First frame in [start, stop) with `need` consecutive moving frames, else stop."""
    stop = min(stop, ball_movement.shape[0])
    for frame in range(max(0, start), max(0, stop - need + 1)):
        if ball_movement[frame:frame + need].all():
            return frame
    return stop


def set_piece_dead_prior(events, ball_speed, cfg):
    """Boolean-ish array: 1 while a tagged restart is still being set up.

    Smart Tagging is a restart marker, not a dead-ball interval: the tag is a short window around
    the restart. So a set piece contributes D from its start until the ball is actually moving,
    capped so a lost ball cannot suppress play for ever. Goals contribute D until the next kickoff.
    """
    n = ball_speed.shape[0]
    prior = np.zeros(n, dtype=np.float32)
    speed = np.nan_to_num(ball_speed.astype(np.float64), nan=0.0)
    moving = speed > 2.0

    kickoff_after_goal = [event for event in events
                          if (event["tags"].get("Type") or "").lower().startswith("kick off")]

    for event in events:
        code = (event["code"] or "").lower()
        start, end = event["start"], event["end"]
        cap = start + int(cfg.set_piece_max_dead_s * cfg.fps)

        if "goal" in code and "kick off" not in code:
            # A goal stops play until the restart. Use the next kickoff tag if there is one.
            later = [other["start"] for other in kickoff_after_goal if other["start"] > start]
            stop = min(later) if later else start + int(75 * cfg.fps)
            prior[start:min(stop, n)] = 1.0
        elif "set pieces" in code:
            stop = _moving_from(moving, start, min(max(end, start + cfg.fps // 2), cap),
                                threshold=2.0, need=5)
            prior[start:max(start, stop)] = 1.0
    return prior


def positional_stillness(bx, by, pitch, window):
    """1 where the ball has not moved much over `window` frames, from positions alone.

    The reported ball speed is unreliable around occlusions, but a stationary ball is also a
    geometric fact: its positions cluster. Judged in metres so the two axes are comparable, and
    required to have most of the window tracked so a gap cannot masquerade as stillness.
    """
    length, width = pitch
    points = np.stack([bx.astype(np.float64) * length, by.astype(np.float64) * width], axis=1)
    half = window // 2
    padded = np.pad(points, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)  # (T, 2, window)
    with np.errstate(all="ignore"):
        spread = np.nanstd(windows, axis=-1).mean(axis=1)
        coverage = np.isfinite(windows).mean(axis=(1, 2))
    spread = np.where(np.isfinite(spread), spread, np.inf)
    stillness = np.exp(-(spread / 0.45) ** 2)
    return np.where(coverage > 0.6, stillness, 0.0).astype(np.float32)


def dead_ball_probability(bx, by, ball_speed_mps, player_speed, events, cfg):
    """P(ball dead) in [0, 1] from four independent kinds of evidence.

    Out of play, a stationary ball while the players stand around, a sustained drop in collective
    player speed, and a tagged restart. Each is smoothed first: a stoppage lasts seconds, while a
    single quiet frame is a tracking artefact.
    """
    length, width = cfg.pitch
    n = bx.shape[0]

    x = bx.astype(np.float64)
    y = by.astype(np.float64)
    out_metres = (np.maximum(0.0, -x) + np.maximum(0.0, x - 1.0)) * length \
        + (np.maximum(0.0, -y) + np.maximum(0.0, y - 1.0)) * width
    out_metres = np.where(np.isfinite(out_metres), out_metres, 0.0)
    out = np.clip((out_metres - cfg.out_of_play_margin_m) / 2.0, 0.0, 1.0)

    speed = np.nan_to_num(ball_speed_mps.astype(np.float64), nan=0.0)
    still_speed = np.exp(-(speed / cfg.still_speed_mps) ** 2)
    # The ball being untracked is not evidence that it is still.
    still_speed = np.where(np.isfinite(ball_speed_mps), still_speed, 0.0)
    still = np.maximum(still_speed, positional_stillness(bx, by, cfg.pitch,
                                                          cfg.dead_smooth_frames))

    collective = fill_finite(player_speed.astype(np.float64))
    # "Slow" is judged against this match's own tempo, not an absolute m/s value: tracking speed
    # varies with the provider, the camera and the state of the game. The median would include the
    # stoppages being detected and sit at a standing-player speed, so use the live-play shoulder
    # of the distribution; a stoppage is roughly half of that.
    reference = float(np.nanpercentile(collective, cfg.tempo_percentile))
    if not np.isfinite(reference) or reference <= 0.2:
        reference = cfg.live_mean_speed_mps
    slow = np.clip((reference - collective) / (0.5 * reference), 0.0, 1.0)

    out_smooth = rolling_nanmean(out.astype(np.float32), 9)
    still_smooth = rolling_nanmean(still.astype(np.float32), cfg.dead_smooth_frames)
    slow_smooth = rolling_nanmean(slow.astype(np.float32), cfg.dead_smooth_frames)

    still_sustained = np.clip((still_smooth - 0.55) / 0.45, 0.0, 1.0)
    slow_sustained = np.clip((slow_smooth - 0.35) / 0.50, 0.0, 1.0)

    tags = set_piece_dead_prior(events, ball_speed_mps, cfg) if events else np.zeros(n, np.float32)
    tags = rolling_nanmean(tags, 9)

    logit = (cfg.dead_bias
             + cfg.dead_out_weight * out_smooth
             + cfg.dead_still_weight * still_sustained
             + cfg.dead_slow_weight * slow_sustained
             + cfg.dead_tag_weight * tags)
    return np.clip(sigmoid(logit), 1e-4, 1.0 - 1e-4).astype(np.float32)


def team_tag_prior(events, names, n_frames, weight):
    """Optional weak team prior from the Smart Tagging `Team` group, as a log-odds offset.

    Off by default. Phases are not possessions, so this is evidence and not an answer; and every
    throw-in -- the restart where the side would be most valuable -- is tagged N/A.
    """
    offset = np.zeros(n_frames, dtype=np.float32)
    if weight <= 0:
        return offset
    home = (names[0] or "").lower()
    away = (names[1] or "").lower()
    for event in events:
        team = (event["tags"].get("Team") or "").lower()
        if not team or team == "n/a":
            continue
        sign = 1.0 if team == home else (-1.0 if team == away else 0.0)
        if sign:
            offset[event["start"]:event["end"]] += sign * weight
    return offset


def _side_sign(team, names):
    """+1 for home, -1 for away, 0 for unknown/N-A -- tolerant of the club-name spelling."""
    team = (team or "").strip().lower()
    if not team or team == "n/a":
        return 0.0
    home = (names[0] or "").strip().lower()
    away = (names[1] or "").strip().lower()
    for name, sign in ((home, 1.0), (away, -1.0)):
        if name and (team == name or team in name or name in team):
            return sign
    return 0.0


def restart_team_prior(events, names, n_frames, cfg):
    """Log-odds home-vs-away offset for the team a tagged restart belongs to.

    A kick-off, corner, goal kick or free kick carries the side that will restart, and the restart
    is where a possession starts. The boost is short -- a couple of seconds -- because after that
    the tracking should decide. Throw-ins are tagged N/A and contribute nothing.
    """
    offset = np.zeros(n_frames, dtype=np.float32)
    restart_hold = int(cfg.restart_team_hold_s * cfg.fps)
    kickoff_hold = int(cfg.kickoff_team_hold_s * cfg.fps)

    for event in events:
        tags = event["tags"]
        sign = _side_sign(tags.get("Team"), names)
        if not sign:
            continue
        code = (event["code"] or "").lower()
        kind = (tags.get("Type") or "").lower()

        if kind.startswith("kick off"):
            start = event["start"]
            offset[start:min(n_frames, start + kickoff_hold)] += sign * cfg.kickoff_team_weight
        elif "set pieces" in code:
            start = event["start"]
            stop = min(n_frames, max(event["end"], start) + restart_hold)
            offset[start:stop] += sign * cfg.restart_team_weight
    return offset
