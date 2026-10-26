"""Build a frame,label possession submission from the provided tactical match assets."""

from __future__ import annotations

import os
import sys

import numpy as np

from . import data as assets
from .config import DEFAULT, Config
from .features import (ball_speed, dead_ball_probability, encode_runs, fill_ball,
                       possession_probability, restart_team_prior, sigmoid,
                       team_tag_prior)
from .sequence import decode

H, A, D, X = 0, 1, 2, 3
LABEL_OF = np.array(list("HADX"))


def log_emissions(team_probability, dead_probability, tag_offset, weight):
    """(T, 3) log-likelihood of H / A / D from independent team and play-state evidence."""
    alive = 1.0 - dead_probability
    team = np.clip(team_probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
    dead = np.clip(dead_probability.astype(np.float64), 1e-6, 1.0 - 1e-6)

    out = np.empty((len(team), 3), dtype=np.float64)
    out[:, H] = np.log(alive) + np.log(team)
    out[:, A] = np.log(alive) + np.log(1.0 - team)
    out[:, D] = np.log(dead)
    if tag_offset is not None:
        out[:, H] += tag_offset
        out[:, A] -= tag_offset
    out = np.nan_to_num(out, nan=np.log(1e-6), neginf=np.log(1e-6))
    return out * weight


def build_features(paths, cfg=Config()):
    """Every per-frame signal, computed once. Returns a dict of arrays."""
    team_of_slot, names = assets.read_metadata(paths["atd_meta"])
    tracking = assets.read_atd_txt(paths["atd_raw"], cfg.n_frames, cfg.n_slots,
                                   cache=paths.get("cache"))

    bx, by, _ = fill_ball(tracking["bx"], tracking["by"], cfg.ball_fill_frames)
    speed = ball_speed(bx, by, tracking["bs"], cfg.pitch, cfg.fps, cfg.ball_speed_smooth)

    with np.errstate(all="ignore"):
        player_mean_speed = np.nanmean(tracking["ps"], axis=1)

    team_probability, ball_present = possession_probability(
        tracking["px"], tracking["py"], bx, by, speed, team_of_slot, cfg)

    events = []
    if paths.get("smart_tagging") and os.path.isfile(paths["smart_tagging"]):
        events = assets.read_smart_tagging(paths["smart_tagging"], cfg.fps, cfg.n_frames)

    dead_probability = dead_ball_probability(bx, by, speed, player_mean_speed, events, cfg)
    tag_offset = team_tag_prior(events, names, cfg.n_frames, weight=cfg.tag_team_weight)
    tag_offset += restart_team_prior(events, names, cfg.n_frames, cfg)

    return {
        "names": names,
        "team_of_slot": team_of_slot,
        "bx": bx, "by": by, "ball_speed": speed,
        "player_mean_speed": player_mean_speed,
        "ball_present": ball_present,
        "team_probability": team_probability,
        "dead_probability": dead_probability,
        "tag_offset": tag_offset,
    }


def block_prior(tag_offset, kickoff, first, stop, cfg):
    """Start-state prior for one graded block.

    The second-half broadcast block opens before the kickoff: those frames are graded and the ball
    is not in play, so D is an observation of the published boundary, not a guess. Otherwise the
    restart tag that names the kicking side starts the block on that side, with the tracking free
    to correct it a second later.
    """
    pre_kickoff = max(0, min(kickoff - first, stop - first))
    if pre_kickoff > 0.5 * cfg.fps:
        return pre_kickoff, np.array([0.02, 0.02, 0.96])

    home = 0.5
    if tag_offset is not None and first <= kickoff < stop:
        window = tag_offset[kickoff:min(stop, kickoff + int(2 * cfg.fps))]
        if len(window):
            # The tag can be a few frames late; take the strongest offset in its neighbourhood.
            home = float(sigmoid(float(window[int(np.argmax(np.abs(window)))])))
    return pre_kickoff, np.array([0.90 * home, 0.90 * (1.0 - home), 0.10])


def decode_blocks(features, cfg=Config()):
    """Decode each graded block independently, with the published kickoff boundary respected."""
    emissions = log_emissions(features["team_probability"], features["dead_probability"],
                              features["tag_offset"], cfg.evidence_weight)

    labels = np.full(cfg.n_frames, X, dtype=np.int8)
    for block, (first, last) in enumerate(cfg.graded):
        stop = last + 1
        kickoff = cfg.kickoffs[block] if block < len(cfg.kickoffs) else first
        pre_kickoff, prior = block_prior(features["tag_offset"], kickoff, first, stop, cfg)

        path = decode(emissions[first:stop], cfg, prior=prior)
        path[:pre_kickoff] = D
        labels[first:stop] = path
    return labels


def build(paths, cfg=Config()):
    features = build_features(paths, cfg)
    labels = decode_blocks(features, cfg)
    return labels, features


def write_submission(path, labels, cfg=Config()):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    strings = LABEL_OF[labels]
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(strings))
    return path


def asset_paths(data_dir, cache=None, use_smart_tagging=True):
    """Tactical asset paths for a data directory (metadata, tracking, tagging, homography)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from games import asset_paths as game_paths  # local import keeps this module importable alone

    paths = game_paths(data_dir=data_dir, source="tactical")
    required = [paths["atd_meta"], paths["atd_raw"]]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise IOError(
            "match data not found:\n  {}\n\nThe tracking bundle is gated -- see DATA.md. Unpack "
            "the download so these exist under:\n  {}\n".format(
                "\n  ".join(missing), os.path.join(data_dir, "tactical")))
    if not use_smart_tagging or not os.path.isfile(paths["smart_tagging"]):
        paths.pop("smart_tagging", None)
    if cache:
        paths["cache"] = cache
    return paths


def summarise(labels, cfg=Config()):
    """What the file claims, in the terms the validator prints."""
    runs = encode_runs(labels)
    spells = [(a, b) for a, b, code in runs if code in (H, A)]
    alive = int(np.sum((labels == H) | (labels == A)))
    return {
        "counts": {name: int(np.sum(labels == index)) for index, name in enumerate(LABEL_OF)},
        "n_spells": len(spells),
        "mean_duration_s": (alive / float(cfg.fps) / len(spells)) if spells else 0.0,
    }
