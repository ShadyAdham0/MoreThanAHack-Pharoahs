"""Full-length smoke test: generate a 199,319-frame ATD and run the real pipeline on it.

Checks the things a five-minute test cannot: parser throughput, memory, Viterbi over the two real
graded blocks, and that the output contract survives a full-size run. The synthetic content is not
a football match and the labels are not meaningful; only the machinery is under test.
"""

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, ".")
from solution import data as assets
from solution import pipeline
from solution.config import Config

N_FRAMES = 199_319
N_SLOTS = 46


def generate(folder):
    meta = os.path.join(folder, "match_atd.xml")
    raw = os.path.join(folder, "match_atd.txt")
    players = "".join(
        '<Player id="{}" teamId="{}"><ProviderPlayerParameters>'
        '<ProviderParameter><Name>position_type</Name><Value>Player</Value>'
        '</ProviderParameter></ProviderPlayerParameters></Player>'.format(
            slot + 1, "home" if slot < 23 else "away") for slot in range(N_SLOTS))
    with open(meta, "w") as handle:
        handle.write('<?xml version="1.0"?><main><Metadata><Teams>'
                     '<Team id="home"><Name>Home</Name></Team>'
                     '<Team id="away"><Name>Away</Name></Team>'
                     '</Teams><Players>{}</Players></Metadata></main>'.format(players))

    rng = np.random.RandomState(0)
    base = rng.rand(N_SLOTS, 2).astype(np.float32) * 0.8 + 0.1
    start = time.time()
    with open(raw, "w") as handle:
        for frame in range(N_FRAMES):
            phase = 0.01 * np.sin(frame / 200.0 + np.arange(N_SLOTS))
            positions = np.clip(base + phase[:, None].astype(np.float32), 0.01, 0.99)
            groups = ";".join("{:.5f},{:.5f},{:.3f}".format(x, y, 2.0)
                              for x, y in positions)
            team = frame // 400 % 2
            ball_slot = team * 23 + 3
            ball = "NaN,NaN,NaN" if (frame // 37) % 5 == 0 else \
                "{:.5f},{:.5f},{:.3f}".format(positions[ball_slot, 0],
                                              positions[ball_slot, 1], 1.5)
            handle.write("{}:{}:{}\n".format(frame + 1, groups, ball))
    print("wrote {} frames in {:.1f}s ({:.1f} MB)".format(
        N_FRAMES, time.time() - start, os.path.getsize(raw) / 1e6))
    return meta, raw


def main():
    with tempfile.TemporaryDirectory() as folder:
        meta, raw = generate(folder)
        cache = os.path.join(folder, "cache.npz")

        start = time.time()
        tracking = assets.read_atd_txt(raw, N_FRAMES, N_SLOTS, cache=cache)
        print("parsed in {:.1f}s".format(time.time() - start))
        start = time.time()
        assets.read_atd_txt(raw, N_FRAMES, N_SLOTS, cache=cache)
        print("cache read in {:.1f}s".format(time.time() - start))

        cfg = Config()
        start = time.time()
        labels, _ = pipeline.build({"atd_meta": meta, "atd_raw": raw, "cache": cache}, cfg)
        print("features + decode in {:.1f}s".format(time.time() - start))
        print("labels", {name: int(np.sum(labels == index))
                         for index, name in enumerate("HADX")})
        out = os.path.join(folder, "submission.csv")
        pipeline.write_submission(out, labels, cfg)
        print("submission", os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
