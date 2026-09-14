#!/usr/bin/env python3
"""3-round evaluation — registry + eval_lib (fixed questions, archived results).

Runs num_rounds rounds over all eval modes of a registered experiment, using
pre-sampled fixed layouts (paired A/B capable) and archiving results as
JSON + Markdown under eval_results/.

Usage (Gazebo must be running; launch it with gui:=false for speed):
  python3 src/drl_navigation_ros2/multi_robot_eval_3round.py
  python3 src/drl_navigation_ros2/multi_robot_eval_3round.py --exp 25dim_shared_buffer_v1_extended_3r
  python3 src/drl_navigation_ros2/multi_robot_eval_3round.py --rounds 1 --episodes 5   # smoke test
"""

import argparse
import time

import rclpy

from experiment_registry import get_exp, get_baselines, SEED
from eval_lib import (
    set_seed, git_commit, load_models,
    generate_or_load_layouts, run_rounds, print_final_table, save_results,
)
from multi_robot_env import MultiRobotEnv


def main():
    parser = argparse.ArgumentParser(description="3-round eval (registry + fixed questions)")
    parser.add_argument("--exp", default="25dim_baseline_3r",
                        help="Experiment id in experiment_registry.EXPERIMENTS")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes per mode per round (default: from registry)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-layouts", action="store_true",
                        help="Use random layouts instead of pre-sampled fixed questions")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate the layout file even if it exists")
    args = parser.parse_args()

    set_seed(args.seed)
    rclpy.init(args=[])

    exp = get_exp(args.exp)
    num_robots = exp["num_robots"]
    episodes = args.episodes or exp["episodes_per_mode"]
    baselines = get_baselines(exp["scene"], num_robots)
    commit = git_commit()

    print("=" * 60)
    print(f"3-Round Evaluation  |  exp={args.exp}")
    print(f"   scene={exp['scene']}  N={num_robots}  state={exp['state_dim'] + exp['extra_state_dim']}d")
    print(f"   rounds={args.rounds} × {len(exp['eval_modes'])} modes × {episodes} ep")
    print(f"   seed={args.seed}  git={commit}  fixed_questions={not args.no_layouts}")
    print("=" * 60)

    print("\n📦 Loading models:")
    models = load_models(exp)

    env = MultiRobotEnv(num_robots=num_robots, scene=exp["scene"],
                        reward_mode=exp["reward_mode"])

    # Pre-sampled "fixed questions" (deterministic, shared across A/B runs).
    layouts = None
    if not args.no_layouts:
        layouts = generate_or_load_layouts(
            exp["scene"], num_robots, num_rounds=args.rounds,
            episodes_per_mode=episodes, seed=args.seed,
            reward_mode=exp["reward_mode"], modes=exp["eval_modes"],
            force=args.force,
        )

    total_start = time.time()
    results = run_rounds(env, models, exp["eval_modes"], args.rounds, episodes,
                         exp["max_steps"], layouts=layouts,
                         extra_state_dim=exp["extra_state_dim"])
    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    print_final_table(results, baselines, exp["eval_modes"])

    save_results(
        args.exp, results,
        meta={
            "exp_id": args.exp,
            "scene": exp["scene"], "num_robots": num_robots,
            "state_dim": exp["state_dim"] + exp["extra_state_dim"],
            "reward_mode": exp["reward_mode"],
            "model_dir": str(exp["checkpoints"][0][0].parent),
            "fixed_questions": not args.no_layouts,
            "layout_file": ("" if layouts is None else
                            f"{exp['scene']}_{num_robots}r_{args.rounds}x{episodes}_seed{args.seed}.json"),
            "seed": args.seed, "git_commit": commit,
            "num_rounds": args.rounds, "episodes_per_mode": episodes,
            "max_steps": exp["max_steps"],
        },
        baselines=baselines,
        modes=exp["eval_modes"],
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()
