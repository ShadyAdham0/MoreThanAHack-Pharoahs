"""Readers for the provided match assets.

Only the tactical recording is used: it is the timeline a submission is indexed to. The broadcast
recording is deliberately not read (see docs/technical_report.md).

The ATD text parser is hand-written for the same reason starter/baseline.py is: it needs no
kloppy, it streams the 199k lines in one pass, and it makes the FIFA/EPTS record layout explicit.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

MISSING_PREFIX = "N"          # every missing value starts with "NaN"
FIELD_SEP = ":"
GROUP_SEP = ";"

HOME, AWAY = 0, 1
LABELS = "HADX"


# ---------------------------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------------------------

def read_metadata(path):
    """(team_of_slot, (home_name, away_name)) from the FIFA/EPTS metadata XML.

    A slot is a tracklet, not a player: 46 of them for 22 players. The first Team in the file is
    the home side (the FIFA/EPTS convention); for BAR-ATH that is Barcelona, which DATA.md states
    independently. `team_of_slot[i]` is HOME or AWAY for raw column i.
    """
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(path, len(teams)))
    home_id = teams[0].get("id")
    names = tuple((team.findtext("Name") or "").strip() for team in teams)

    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(path))
    team_of_slot = np.array([HOME if team_id == home_id else AWAY for team_id in slots],
                            dtype=np.int8)
    return team_of_slot, names


# ---------------------------------------------------------------------------------------------
# raw tracking
# ---------------------------------------------------------------------------------------------

def _group_floats(group):
    """'x,y,s' -> three floats, or None for a missing observation."""
    if not group or group[0] == MISSING_PREFIX:
        return None
    try:
        a, b, c = group.split(",")
        return float(a), float(b), float(c)
    except ValueError:
        return None


def read_atd_txt(path, n_frames, n_slots, cache=None):
    """Read the raw ATD into arrays.

    Returns a dict with float32 arrays:
        px, py, ps : (n_frames, n_slots)  player position and speed, NaN when not tracked
        bx, by, bs : (n_frames,)          ball position and speed, NaN when not tracked

    `cache` is an optional .npz path. The first parse of the real file takes a minute or two; the
    cache makes every later run instant and is written atomically so an interrupted run does not
    leave a half file behind.
    """
    if cache and os.path.isfile(cache):
        with np.load(cache) as data:
            return {key: data[key] for key in ("px", "py", "ps", "bx", "by", "bs")}

    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bs = np.full(n_frames, np.nan, dtype=np.float32)

    seen = 0
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # frame_id counts from 1; submissions from 0
            if frame < 0 or frame >= n_frames:
                continue

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots:
                    break
                values = _group_floats(group)
                if values is not None:
                    px[frame, slot], py[frame, slot], ps[frame, slot] = values

            values = _group_floats(parts[2].strip())
            if values is not None:
                bx[frame], by[frame], bs[frame] = values
            seen += 1

    if seen != n_frames:
        raise ValueError("{}: read {} frames, expected {} (alignment check failed)".format(
            path, seen, n_frames))

    arrays = {"px": px, "py": py, "ps": ps, "bx": bx, "by": by, "bs": bs}
    if cache:
        folder = os.path.dirname(cache)
        if folder:
            os.makedirs(folder, exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, cache)
    return arrays


# ---------------------------------------------------------------------------------------------
# Smart Tagging
# ---------------------------------------------------------------------------------------------

def _tags(instance):
    return {(label.findtext("group") or "").strip(): (label.findtext("text") or "").strip()
            for label in instance.findall("label")}


def read_smart_tagging(path, fps, n_frames):
    """Smart Tagging instances as dicts, with frame indices.

    `start`/`end` are seconds from the first tactical frame (verified in DATA.md), so
    frame = round(seconds * fps). Intervals are clamped to the file and half-open on the right.
    """
    out = []
    for instance in ET.parse(path).getroot().iter("instance"):
        start_s = float(instance.findtext("start") or 0.0)
        end_s = float(instance.findtext("end") or start_s)
        start = max(0, min(n_frames, int(round(start_s * fps))))
        end = max(start, min(n_frames, int(round(end_s * fps))))
        out.append({
            "code": (instance.findtext("code") or "").strip(),
            "start_s": start_s,
            "end_s": end_s,
            "start": start,
            "end": end,
            "tags": _tags(instance),
        })
    out.sort(key=lambda event: (event["start"], event["end"]))
    return out


# ---------------------------------------------------------------------------------------------
# homography (read for completeness; the tactical matrix is valid everywhere)
# ---------------------------------------------------------------------------------------------

def read_homography(path):
    """(H, reprojection_error, valid) -- H is (n_frames, 3, 3), row-major columns 1..9."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    matrix = np.stack([table["h00"], table["h01"], table["h02"],
                       table["h10"], table["h11"], table["h12"],
                       table["h20"], table["h21"], table["h22"]], axis=1)
    return matrix.reshape(-1, 3, 3), table["reprojection_error"], table["valid"].astype(bool)
