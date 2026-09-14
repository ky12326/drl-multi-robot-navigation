#!/usr/bin/env python3
"""Pure-Python layout test for N-robot support (2..5) — no Gazebo needed.

For every (scene, num_robots) combo, sample each eval mode + the train
curriculum sampler and assert:
  1. No IndexError / crash (the N>=5 center/cross bugs are regression-checked)
  2. Valid-layout hit rate >= threshold (default 70%)

Usage:
  source /opt/ros/foxy/setup.bash && source ~/mapf/bin/activate
  python3 src/drl_navigation_ros2/test_layouts_multi_n.py [--samples N] [--threshold X]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_BASE = Path(__file__).parent
sys.path.append(str(_BASE))

from multi_robot_env import MultiRobotEnv              # noqa: E402
from experiment_registry import SCENE1B_MODES_NR, SCENE0_MODES_NR  # noqa: E402


def _mode_sets(scene):
    if scene == "scene0":
        return [("train", "train")] + SCENE0_MODES_NR
    return [("train", "train")] + SCENE1B_MODES_NR


def test_scene(scene, num_robots, samples, threshold):
    env = MultiRobotEnv(num_robots=num_robots, scene=scene, headless=True)
    all_ok = True
    for mode, tag in _mode_sets(scene):
        env.set_reset_mode(mode)
        ok, crash = 0, False
        for _ in range(samples):
            try:
                r = env._sample_episode_layout()
                if r[0] is not None and len(r[0]) == num_robots:
                    ok += 1
            except IndexError:
                crash = True
        rate = ok / samples
        passed = (rate >= threshold) and not crash
        all_ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] scene={scene} N={num_robots} "
              f"{tag:8s} hit={rate:.0%} crash={crash}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60,
                        help="Samples per (scene, N, mode) — 60 is a good gate size")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Minimum valid-layout hit rate")
    parser.add_argument("--num-robots", type=int, default=None,
                        help="Test only this robot count (default: 2..5)")
    args = parser.parse_args()

    robots = [args.num_robots] if args.num_robots else [2, 3, 4, 5]
    all_ok = True
    for scene in ["scene0", "scene1b"]:
        for n in robots:
            print(f"\n── scene={scene}  N={n}  ({args.samples} samples/mode, "
                  f"threshold {args.threshold:.0%}) ──")
            all_ok &= test_scene(scene, n, args.samples, args.threshold)

    print("\n" + "=" * 55)
    if all_ok:
        print("✅ ALL LAYOUT TESTS PASSED (no crash, hit rate OK)")
    else:
        print("❌ SOME LAYOUT TESTS FAILED — see above")
    print("=" * 55)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
