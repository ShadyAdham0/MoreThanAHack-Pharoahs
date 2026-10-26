"""Command line entry point.

    # the real pipeline (needs the gated bundle in ./data)
    python solution/run.py --data-dir data --out evaluation/results/my_team.csv

    # the stand-in that needs only the shipped baseline output
    python solution/run.py --fallback evaluation/results/baseline_raw.csv \
                           --out evaluation/results/fallback_hmm.csv

Validate the result before submitting:

    python evaluation/validate_submission.py evaluation/results/my_team.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from solution.config import DEFAULT, Config                      # noqa: E402
from solution import fallback, pipeline                          # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a possession submission.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", help="directory holding tactical/ and broadcast/")
    source.add_argument("--fallback", metavar="BASELINE_CSV",
                        help="build from a shipped baseline CSV instead of the tracking data")
    parser.add_argument("--out", required=True, help="output frame,label CSV")
    parser.add_argument("--cache", help="optional .npz cache of the parsed tracking arrays")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="ignore Smart Tagging (ablation)")
    parser.add_argument("--diagnostics", help="optional .npz of the intermediate evidence")
    parser.add_argument("--dead-scale", type=float, default=1.0,
                        help="fallback only: exaggerate (>1) or soften (<1) dead-ball evidence")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = DEFAULT

    if args.fallback:
        labels, extras = fallback.build(args.fallback, cfg, dead_scale=args.dead_scale)
    else:
        try:
            paths = pipeline.asset_paths(args.data_dir, cache=args.cache,
                                         use_smart_tagging=not args.no_smart_tagging)
        except IOError as error:
            print("ERROR: {}".format(error), file=sys.stderr)
            print("The scoring pipeline still runs on the shipped baselines; the stand-in is:\n"
                  "  python solution/run.py --fallback evaluation/results/baseline_raw.csv "
                  "--out evaluation/results/fallback_hmm.csv", file=sys.stderr)
            return 2
        labels, features = pipeline.build(paths, cfg)
        extras = {key: value for key, value in features.items()
                  if isinstance(value, np.ndarray)}

    pipeline.write_submission(args.out, labels, cfg)
    summary = pipeline.summarise(labels, cfg)
    print("wrote {}".format(args.out))
    print("  " + "  ".join("{} {}".format(name, count)
                           for name, count in summary["counts"].items()))
    print("  {} possessions, mean {:.1f} s".format(
        summary["n_spells"], summary["mean_duration_s"]))

    if args.diagnostics:
        folder = os.path.dirname(os.path.abspath(args.diagnostics))
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(args.diagnostics, labels=labels, **extras)
        print("  diagnostics -> {}".format(args.diagnostics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
