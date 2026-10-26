"""End-to-end test of the data-driven pipeline on a synthetic match.

The real bundle is gated. This fabricates the same file formats (metadata XML, raw EPTS text,
SportsCode XML) with a known possession sequence, so the parser, the geometry, the dead-ball
logic, the decoder and the submission contract are all exercised by `python tests/synthetic_match.py`.

It is a correctness test, not a model benchmark: the synthetic players are deliberately simple.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from solution import pipeline                                   # noqa: E402
from solution.config import Config                              # noqa: E402
from solution.data import read_metadata                         # noqa: E402

HOME_ID = "team_home"
AWAY_ID = "team_away"
N_SLOTS = 46


def write_metadata(path):
    players = []
    for slot in range(N_SLOTS):
        team = HOME_ID if slot < N_SLOTS // 2 else AWAY_ID
        players.append('<Player id="{}" teamId="{}"><ProviderPlayerParameters>'
                       '<ProviderParameter><Name>position_type</Name><Value>Player</Value>'
                       '</ProviderParameter></ProviderPlayerParameters></Player>'
                       .format(slot + 1, team))
    xml = ('<?xml version="1.0" encoding="utf-8"?><main><Metadata>'
           '<Teams>'
           '<Team id="{}"><Name>Home FC</Name></Team>'
           '<Team id="{}"><Name>Away FC</Name></Team>'
           '</Teams><Players>{}</Players>'
           '</Metadata></main>').format(HOME_ID, AWAY_ID, "".join(players))
    with open(path, "w") as handle:
        handle.write(xml)


def write_smart_tagging(path, segments, fps):
    instances = []
    for start, stop, code, team in segments:
        instances.append(
            '<instance><start>{:.3f}</start><end>{:.3f}</end><code>{}</code>'
            '<label><group>Team</group><text>{}</text></label>'
            '<label><group>Half</group><text>1st Half</text></label></instance>'.format(
                start / fps, stop / fps, code, team))
    xml = ('<?xml version="1.0" encoding="utf-8"?><file><ALL_INSTANCES>{}</ALL_INSTANCES>'
           '</file>').format("".join(instances))
    with open(path, "w") as handle:
        handle.write(xml)


def make_match(n_frames, fps=25):
    """A synthetic match: alternating possessions, with dead stretches between them."""
    # (label, frames): H/A possessions of 8-14 s, D of 3-6 s. Kickoff at frame 0.
    truth = np.full(n_frames, "D", dtype="<U1")
    segments = []
    frame = 0
    rng = np.random.RandomState(7)
    team = "H"
    while frame < n_frames:
        length = int(rng.randint(5 * fps, 15 * fps))
        stop = min(n_frames, frame + length)
        truth[frame:stop] = team
        segments.append((frame, stop, team))
        frame = stop
        dead = int(rng.randint(2 * fps, 6 * fps))
        frame = min(n_frames, frame + dead)
        team = "A" if team == "H" else "H"
    return truth, segments


def make_tracking(n_frames, truth, fps=25):
    """Players in a loose 4-3-3, the ball at the feet of one player of the possessing team."""
    rng = np.random.RandomState(11)
    # Normalised base positions per slot; first 23 home, next 23 away.
    base = np.zeros((N_SLOTS, 2), dtype=np.float32)
    for team in range(2):
        for index in range(N_SLOTS // 2):
            base[team * 23 + index] = (
                0.08 + 0.84 * (index % 11) / 10.0 + (0.04 if team else 0.0),
                0.10 + 0.80 * (index // 11) / 10.0 + (0.02 * (index % 4)))
    px = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    py = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    ps = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bs = np.full(n_frames, np.nan, dtype=np.float32)

    for frame in range(n_frames):
        sway = 0.01 * np.sin(frame / 50.0)
        for slot in range(N_SLOTS):
            if rng.rand() < 0.04:      # a tracking dropout
                continue
            px[frame, slot] = np.clip(base[slot, 0] + sway + rng.randn() * 0.004, 0, 1)
            py[frame, slot] = np.clip(base[slot, 1] + rng.randn() * 0.004, 0, 1)
            ps[frame, slot] = 2.0 if truth[frame] != "D" else 0.8
        if truth[frame] == "D":
            # Ball out of play, stationary at the touchline; only sometimes tracked.
            if rng.rand() > 0.35:
                bx[frame] = 0.995 + 0.01
                by[frame] = 0.55
                bs[frame] = 0.0
        else:
            team = 0 if truth[frame] == "H" else 1
            # Ball with a player near the middle of that half of the pitch.
            slot = team * 23 + 6
            if not np.isfinite(px[frame, slot]):
                continue
            bx[frame] = px[frame, slot] + 0.003
            by[frame] = py[frame, slot]
            bs[frame] = 3.0
    return px, py, ps, bx, by, bs


def write_atd(path, px, py, ps, bx, by, bs, fps=25):
    with open(path, "w") as handle:
        for frame in range(len(bx)):
            groups = []
            for slot in range(N_SLOTS):
                if np.isfinite(px[frame, slot]):
                    groups.append("{:.6f},{:.6f},{:.4f}".format(
                        px[frame, slot], py[frame, slot], ps[frame, slot]))
                else:
                    groups.append("NaN,NaN,NaN")
            ball = "NaN,NaN,NaN" if not np.isfinite(bx[frame]) else \
                "{:.6f},{:.6f},{:.4f}".format(bx[frame], by[frame], bs[frame])
            handle.write("{}:{}:{}\n".format(frame + 1, ";".join(groups), ball))


def main():
    fps = 25
    n_frames = 1800
    truth, segments = make_match(n_frames, fps)
    px, py, ps, bx, by, bs = make_tracking(n_frames, truth, fps)

    with tempfile.TemporaryDirectory() as folder:
        meta = os.path.join(folder, "match_atd.xml")
        raw = os.path.join(folder, "match_atd.txt")
        tags = os.path.join(folder, "match_smart_tagging.xml")
        write_metadata(meta)
        write_atd(raw, px, py, ps, bx, by, bs, fps)
        # A couple of tagged restarts at synthetic dead-ball starts.
        write_smart_tagging(tags, [(frame, frame + 25, "SET PIECES", "Home FC")
                                   for frame, _, _ in segments[:3]], fps)

        team_of_slot, names = read_metadata(meta)
        assert team_of_slot.shape == (N_SLOTS,)
        assert names == ("Home FC", "Away FC")

        cfg = Config(n_frames=n_frames, n_slots=N_SLOTS,
                     graded=((0, n_frames - 1),), kickoffs=(0,))
        paths = {"atd_meta": meta, "atd_raw": raw, "smart_tagging": tags}
        labels, features = pipeline.build(paths, cfg)

        # The tagged restart names a side; the prior must actually reach the emissions.
        assert features["tag_offset"][0] > 0, "restart team prior not wired into the features"

        alive = labels != pipeline.X
        assert alive.mean() > 0.95, "pipeline abstained on the graded block"
        accuracy = float(np.mean(labels[alive] ==
                                 np.array([{"H": 0, "A": 1, "D": 2}[str(v)]
                                           for v in truth[alive]], dtype=np.int8)))
        # Team call and alive/dead call are the two things being tested; 0.85 is generous room
        # for a synthetic player layout that is not a football match.
        assert accuracy > 0.85, "synthetic accuracy only {:.3f}".format(accuracy)

        out = os.path.join(folder, "submission.csv")
        pipeline.write_submission(out, labels, cfg)
        with open(out) as handle:
            lines = handle.read().splitlines()
        assert lines[0] == "frame,label"
        assert len(lines) == n_frames + 1
        assert lines[1] == "0,{}".format("HADX"[labels[0]])
        print("synthetic match: accuracy {:.3f}, {} labels, submission OK".format(
            accuracy, {name: int(np.sum(labels == index))
                       for index, name in enumerate("HADX")}))


if __name__ == "__main__":
    main()
