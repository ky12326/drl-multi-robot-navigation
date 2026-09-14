#!/usr/bin/env python3
"""Paired A/B comparison — two experiments on the SAME fixed questions.

Both models replay the identical pre-sampled layout list (same order), so the
comparison is paired: the layout variance is removed, only the model differs.

Usage (Gazebo running; gui:=false for speed):
  python3 src/drl_navigation_ros2/multi_robot_eval_3round_compare.py \
      --exp-a social_v1_2r --exp-b 25dim_curriculum_v1_extended_3r
  python3 src/drl_navigation_ros2/multi_robot_eval_3round_compare.py \
      --exp-a X --exp-b Y --rounds 1 --episodes 5   # smoke test
"""

import argparse
import time

import numpy as np
import rclpy

from experiment_registry import get_exp, get_baselines, SEED
from eval_lib import (
    set_seed, git_commit, load_models,
    generate_or_load_layouts, run_rounds, save_results,
)
from multi_robot_env import MultiRobotEnv


def print_comparison(results_a, results_b, baselines, modes, label_a, label_b):
    print(f"\n{'=' * 100}")
    print(f"对比: {label_a}  vs  {label_b}   (配对 — 同一组固定题目)")
    print(f"{'=' * 100}")

    header = (f"{'模式':<8} {label_a+'均值':>12} {label_b+'均值':>12} {'Δ均值':>8} "
              f"{'σ_a':>6} {'σ_b':>6} 判定")
    print(header)
    print("-" * 100)

    for mode, tag in modes:
        ra = results_a.get(tag, [])
        rb = results_b.get(tag, [])
        if not ra or not rb:
            continue
        a_ag = [r["all_goal"] for r in ra]
        b_ag = [r["all_goal"] for r in rb]
        a_mean, a_std = float(np.mean(a_ag)), float(np.std(a_ag))
        b_mean, b_std = float(np.mean(b_ag)), float(np.std(b_ag))
        delta = b_mean - a_mean

        if delta > 0.05:
            flag = "✅ B 改善"
        elif delta < -0.10:
            flag = "⚠️ B 退化"
        elif abs(delta) <= 0.05:
            flag = "➡️ 持平"
        else:
            flag = "—"

        print(f"{tag:<8} {a_mean:>12.2f} {b_mean:>12.2f} {delta:>+8.2f} "
              f"{a_std:>6.2f} {b_std:>6.2f}  {flag}")

    print("-" * 100)
    print("\nPer-mode round-by-round:")
    for mode, tag in modes:
        ra = results_a.get(tag, [])
        rb = results_b.get(tag, [])
        if not ra or not rb:
            continue
        print(f"\n  [{tag}]")
        print(f"    A rounds: " + "  ".join(f"R{i}={r['all_goal']:.2f}" for i, r in enumerate(ra)))
        print(f"    B rounds: " + "  ".join(f"R{i}={r['all_goal']:.2f}" for i, r in enumerate(rb)))


def main():
    parser = argparse.ArgumentParser(description="Paired A/B eval on shared fixed questions")
    parser.add_argument("--exp-a", default="social_v1_2r",
                        help="Baseline experiment id (default: zero-shot social-v1)")
    parser.add_argument("--exp-b", default="25dim_curriculum_v1_extended_3r",
                        help="Candidate experiment id")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate the layout file even if it exists")
    args = parser.parse_args()

    set_seed(args.seed)
    rclpy.init(args=[])

    exp_a = get_exp(args.exp_a)
    exp_b = get_exp(args.exp_b)

    # Reference config comes from exp_b (the candidate); exp_a is evaluated at
    # the same scene / num_robots / modes so both see identical questions.
    scene, num_robots = exp_b["scene"], exp_b["num_robots"]
    modes = exp_b["eval_modes"]
    episodes = args.episodes or exp_b["episodes_per_mode"]
    baselines = get_baselines(scene, num_robots)
    commit = git_commit()

    print("=" * 70)
    print("Paired A/B Comparison (fixed questions)")
    print(f"   A: {args.exp_a}  |  B: {args.exp_b}")
    print(f"   scene={scene}  N={num_robots}  rounds={args.rounds} × {episodes} ep/mode")
    print(f"   seed={args.seed}  git={commit}")
    print("=" * 70)

    # One shared layout file → both models replay the same questions.
    layouts = generate_or_load_layouts(
        scene, num_robots, num_rounds=args.rounds, episodes_per_mode=episodes,
        seed=args.seed, reward_mode=exp_b["reward_mode"], modes=modes,
        force=args.force,
    )

    env = MultiRobotEnv(num_robots=num_robots, scene=scene, reward_mode=exp_b["reward_mode"])

    print(f"\n📦 Loading models A ({args.exp_a}):")
    models_a = load_models(exp_a, num_robots=num_robots)
    print(f"\n📦 Loading models B ({args.exp_b}):")
    models_b = load_models(exp_b, num_robots=num_robots)

    total_start = time.time()
    results_a = run_rounds(env, models_a, modes, args.rounds, episodes,
                           exp_b["max_steps"], layouts=layouts,
                           extra_state_dim=exp_a["extra_state_dim"])
    results_b = run_rounds(env, models_b, modes, args.rounds, episodes,
                           exp_b["max_steps"], layouts=layouts,
                           extra_state_dim=exp_b["extra_state_dim"])
    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    print_comparison(results_a, results_b, baselines, modes, args.exp_a, args.exp_b)

    layout_file = f"{scene}_{num_robots}r_{args.rounds}x{episodes}_seed{args.seed}.json"
    common_meta = {
        "scene": scene, "num_robots": num_robots,
        "state_dim": exp_b["state_dim"] + exp_b["extra_state_dim"],
        "fixed_questions": True, "layout_file": layout_file,
        "seed": args.seed, "git_commit": commit,
        "num_rounds": args.rounds, "episodes_per_mode": episodes,
        "max_steps": exp_b["max_steps"],
        "compare_with": args.exp_b,
    }
    save_results(f"{args.exp_a}__vs__{args.exp_b}__A", results_a, meta=dict(
        common_meta, exp_id=args.exp_a, model_dir=str(exp_a["checkpoints"][0][0].parent),
        compare_partner=args.exp_b), baselines=baselines, modes=modes)
    save_results(f"{args.exp_a}__vs__{args.exp_b}__B", results_b, meta=dict(
        common_meta, exp_id=args.exp_b, model_dir=str(exp_b["checkpoints"][0][0].parent),
        compare_partner=args.exp_a), baselines=baselines, modes=modes)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
