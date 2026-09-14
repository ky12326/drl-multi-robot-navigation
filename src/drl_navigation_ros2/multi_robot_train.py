#!/usr/bin/env python3
"""Generic multi-robot SAC training — one script for any robot count.

Model dir, state dim, eval modes are derived from CLI args; warm-start resolves
through the experiment registry (robot2+ copies robot0). Supports the optional
neighbor-feature state (25 + 2*(N-1) dims) and padded warm-start.

Usage (Gazebo must be running; launch it with gui:=false for speed):
  # 3-robot plain 25-dim, warm-started from the shared-buffer E30 model
  python src/drl_navigation_ros2/multi_robot_train.py \
      --tag my_v1 --num-robots 3 \
      --warmstart 25dim_shared_buffer_v1_extended_3r --epochs 15

  # 4-robot with neighbor features (state = 25 + 6 = 31 dims), padded warm-start
  python src/drl_navigation_ros2/multi_robot_train.py \
      --tag neighbors_v1 --num-robots 4 --neighbors \
      --warmstart 25dim_shared_buffer_v1_extended_3r
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch

_BASE = Path(__file__).parent
sys.path.append(str(_BASE))

rclpy.init(args=[])

from multi_robot_env import MultiRobotEnv   # noqa: E402
from SAC.SAC import SAC                      # noqa: E402
from SAC.frame_stack import FrameStacker     # noqa: E402
from replay_buffer import ReplayBuffer       # noqa: E402
from eval_lib import set_seed, git_commit     # noqa: E402
from experiment_registry import (            # noqa: E402
    SEED, MODELS_ROOT, get_exp, resolve_checkpoints, load_state_dim,
    SCENE1B_MODES_NR, SCENE0_MODES_NR,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Generic N-robot SAC training")
    p.add_argument("--tag", required=True, help="Experiment tag (model dir suffix)")
    p.add_argument("--num-robots", type=int, default=3, help="Number of robots (2..5+)")
    p.add_argument("--scene", default="scene1b", choices=["scene0", "scene1b", "scene2"])
    p.add_argument("--state-dim", type=int, default=25, help="Base state dim (no neighbors)")
    p.add_argument("--neighbors", action="store_true",
                   help="Append 2*(N-1) neighbor coords to the state")
    p.add_argument("--warmstart", default=None,
                   help="Registry exp id to warm-start from (robot2+ copies robot0)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--episodes-per-epoch", type=int, default=40)
    p.add_argument("--buffer-size", type=int, default=150000)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--seq-len", type=int, default=1,
                   help="Frame-stacking length (1 = single-frame, current behavior)")
    return p.parse_args()


def main():
    args = _parse_args()
    set_seed(args.seed)

    n = args.num_robots
    extra_dim = 2 * (n - 1) if args.neighbors else 0
    frame_dim = args.state_dim + extra_dim        # single-frame state dim
    net_input_dim = frame_dim * args.seq_len      # network input (stacked)
    name_dim = f"{frame_dim}d" if args.seq_len == 1 else f"{frame_dim}d_seq{args.seq_len}"
    save_name = f"SAC_multi_robot_{args.scene}_{n}robot_{name_dim}_{args.tag}"
    save_dir = MODELS_ROOT / save_name

    eval_modes = SCENE0_MODES_NR if args.scene == "scene0" else SCENE1B_MODES_NR

    # ---- warm-start resolution ----
    warm_ckpts = None
    old_state_dim = None
    warm_exp = None
    if args.warmstart:
        warm_exp = get_exp(args.warmstart)
        warm_ckpts = resolve_checkpoints(warm_exp, n)
        warm_dim = load_state_dim(warm_exp)
        if warm_dim < net_input_dim:
            old_state_dim = warm_dim  # padded warm-start (pad_input_weights)
        elif warm_dim > net_input_dim:
            raise SystemExit(
                f"warmstart {args.warmstart} is {warm_dim}d but target is {net_input_dim}d")

    CONFIG = {
        "max_epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "max_global_steps": 360,
        "discount": 0.99, "actor_lr": 1e-4, "critic_lr": 1e-4,
        "critic_tau": 0.005, "init_temperature": 0.1,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "training_iterations": 300,
        "train_every_n": 2,
        "eval_start_epoch": 3, "eval_every_epochs": 3,
        "eval_episodes": 20, "eval_max_steps": 420,
    }

    print("=" * 60)
    print("Generic N-robot SAC training")
    print(f"   N={n}  scene={args.scene}  state={frame_dim}d "
          f"(base {args.state_dim} + {extra_dim} neighbor)  "
          f"net_input={net_input_dim}d (seq_len={args.seq_len})")
    print(f"   warmstart={args.warmstart} (old_dim={old_state_dim})")
    print(f"   save: {save_dir}")
    print(f"   epochs: E01 → E{CONFIG['max_epochs']} × {CONFIG['episodes_per_epoch']} ep")
    print(f"   eval modes: {[t for _, t in eval_modes]}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def build_models():
        models = []
        for i in range(n):
            if warm_ckpts is not None:
                load_dir, load_name = warm_ckpts[i]
                if not (Path(load_dir) / f"{load_name}_actor.pth").exists():
                    raise FileNotFoundError(
                        f"checkpoint not found: {load_dir}/{load_name}_*.pth")
                load_kwargs = dict(load_model=True, load_directory=load_dir,
                                   load_name=load_name, old_state_dim=old_state_dim)
                tag = " (copy r0)" if i >= len(warm_exp["checkpoints"]) else ""
                src = str(load_dir)
            else:
                load_kwargs = dict(load_model=False, old_state_dim=None)
                tag = " (cold start)"
                src = "random init"
            models.append(SAC(
                state_dim=args.state_dim, action_dim=2, max_action=1,
                device=torch.device("cpu"),
                discount=CONFIG["discount"],
                actor_lr=CONFIG["actor_lr"], critic_lr=CONFIG["critic_lr"],
                critic_tau=CONFIG["critic_tau"],
                init_temperature=CONFIG["init_temperature"],
                save_every=0,
                extra_state_dim=extra_dim,
                seq_len=args.seq_len,
                save_directory=save_dir / str(i),
                model_name=f"{save_name}_{i}",
                log_dir=None,
                **load_kwargs,
            ))
            print(f"   robot{i}: {src}{tag}")
        return models

    def _nf(obs):
        """Neighbor features from an obs tuple, or None when not enabled."""
        if extra_dim > 0 and len(obs) > 8:
            return obs[8]
        return None

    # ------------------------------------------------------------------
    # Stats helpers (parameterized by n)
    # ------------------------------------------------------------------
    def new_stats():
        return {
            "episodes": 0, "all_goal": 0, "any_collision": 0, "deadlock": 0,
            "sum_min_pairwise": 0.0, "valid_eps": 0,
            "per_robot": [
                {"goal": 0, "collision": 0, "timeout": 0, "sum_steps": 0}
                for _ in range(n)
            ],
        }

    def new_ep_metrics():
        return {
            "min_pairwise": float("inf"), "no_progress_steps": 0,
            "prev_distances": [None] * n,
        }

    def update_episode_metrics(env, ep):
        wp = env._last_world_position
        if wp is None:
            return
        pos = []
        for i in range(n):
            if wp[i] is not None:
                pos.append(np.asarray([float(wp[i].x), float(wp[i].y)], dtype=np.float32))
            else:
                pos.append(None)
        valid = [p for p in pos if p is not None]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                d = float(np.linalg.norm(valid[i] - valid[j]))
                ep["min_pairwise"] = min(ep["min_pairwise"], d)
        targets = env.targets
        if targets is None or any(p is None for p in pos):
            return
        any_progress = False
        for i, p in enumerate(pos):
            if p is None or i >= len(targets):
                continue
            cur_d = float(np.linalg.norm(p - np.asarray(targets[i], dtype=np.float32)))
            prev_d = ep["prev_distances"][i]
            if prev_d is None:
                ep["prev_distances"][i] = cur_d
                continue
            if cur_d < prev_d - 0.02:
                any_progress = True
            ep["prev_distances"][i] = cur_d
        if any_progress:
            ep["no_progress_steps"] = 0
        else:
            ep["no_progress_steps"] += 1

    def merge_episode(stats, ep_metrics, final_status):
        stats["episodes"] += 1
        if all(s == "goal" for s in final_status):
            stats["all_goal"] += 1
        if any(s == "collision" for s in final_status):
            stats["any_collision"] += 1
        if ep_metrics["no_progress_steps"] >= 60:
            stats["deadlock"] += 1
        if np.isfinite(ep_metrics["min_pairwise"]):
            stats["valid_eps"] += 1
            stats["sum_min_pairwise"] += ep_metrics["min_pairwise"]
        for i in range(n):
            stats["per_robot"][i][final_status[i]] += 1

    def print_stats(stats, tag="train"):
        eps = max(stats["episodes"], 1)
        all_goal = stats["all_goal"] / eps
        col_rate = stats["any_collision"] / eps
        deadlock = stats["deadlock"] / eps
        avg_pd = stats["sum_min_pairwise"] / max(stats["valid_eps"], 1)
        parts = [f"all_goal={all_goal:.2f}  collision={col_rate:.2f}  deadlock={deadlock:.2f}",
                 f"min_pair_dist={avg_pd:.2f}"]
        for i in range(n):
            s = stats["per_robot"][i]
            parts.append(
                f"r{i}: G={s['goal']/eps:.2f} C={s['collision']/eps:.2f} "
                f"T={s['timeout']/eps:.2f}  steps={s['sum_steps']/eps:.0f}")
        print(f"  [{tag}] " + "  |  ".join(parts))

    def run_eval(env, models, mode):
        stats = new_stats()
        old_mode = env.reset_mode
        try:
            env.set_reset_mode(mode)
            stackers = [FrameStacker(frame_dim, args.seq_len) for _ in range(n)]
            for _ in range(CONFIG["eval_episodes"]):
                obs_list = env.reset()
                active = [True] * n
                final_status = [None] * n
                ep_metrics = new_ep_metrics()
                robot_steps = [0] * n
                # Prime frame stackers with the first frame (fills history)
                for i in range(n):
                    stackers[i].reset()
                    frame0, _ = models[i].prepare_state(
                        *obs_list[i][:7], _nf(obs_list[i]))
                    for _ in range(args.seq_len):
                        stackers[i].push(frame0)
                gs = 0
                while gs < CONFIG["eval_max_steps"] and any(active):
                    cmds = []
                    for i in range(n):
                        if active[i]:
                            frame, _ = models[i].prepare_state(
                                *obs_list[i][:7], _nf(obs_list[i]))
                            stacked = stackers[i].push(frame)
                            action = models[i].get_action(stacked, False)
                            cmds.append(((float(action[0]) + 1.0) / 2.0, float(action[1])))
                        else:
                            cmds.append((0.0, 0.0))
                    next_obs = env.step(cmds, active_mask=active, debug_context=mode)
                    update_episode_metrics(env, ep_metrics)
                    for i in range(n):
                        if not active[i]:
                            continue
                        robot_steps[i] += 1
                        _, terminal = models[i].prepare_state(
                            *next_obs[i][:7], _nf(next_obs[i]))
                        if terminal:
                            active[i] = False
                            final_status[i] = "goal" if next_obs[i][5] else "collision"
                    obs_list = next_obs
                    gs += 1
                for i in range(n):
                    if final_status[i] is None:
                        final_status[i] = "timeout"
                    stats["per_robot"][i]["sum_steps"] += robot_steps[i]
                merge_episode(stats, ep_metrics, final_status)
            return stats
        finally:
            env.set_reset_mode(old_mode)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    env = MultiRobotEnv(num_robots=n, scene=args.scene, reward_mode="social_v1")
    models = build_models()

    shared_buffer = ReplayBuffer(buffer_size=CONFIG["buffer_size"], random_seed=4242)

    epoch = 1
    global_episode = 0
    while epoch <= CONFIG["max_epochs"]:
        epoch_start = time.time()
        train_stats = new_stats()

        for _ep_in_epoch in range(CONFIG["episodes_per_epoch"]):
            env.set_reset_mode("train")
            obs_list = env.reset()
            active = [True] * n
            final_status = [None] * n
            ep_metrics = new_ep_metrics()
            robot_steps = [0] * n
            # Frame stackers (seq_len=1 → identity, current behavior)
            stackers = [FrameStacker(frame_dim, args.seq_len) for _ in range(n)]
            for i in range(n):
                frame0, _ = models[i].prepare_state(
                    *obs_list[i][:7], _nf(obs_list[i]))
                for _ in range(args.seq_len):
                    stackers[i].push(frame0)
            gs = 0

            while gs < CONFIG["max_global_steps"] and any(active):
                states = [None] * n
                raw_actions = [None] * n
                cmds = []
                for i in range(n):
                    if active[i]:
                        frame, _ = models[i].prepare_state(
                            *obs_list[i][:7], _nf(obs_list[i]))
                        states[i] = stackers[i].push(frame)
                        action = models[i].get_action(states[i], add_noise=True)
                        raw_actions[i] = action
                        cmds.append(((float(action[0]) + 1.0) / 2.0, float(action[1])))
                    else:
                        cmds.append((0.0, 0.0))

                next_obs = env.step(cmds, active_mask=active, debug_context="train")
                update_episode_metrics(env, ep_metrics)

                for i in range(n):
                    if not active[i]:
                        continue
                    robot_steps[i] += 1
                    frame_next, terminal = models[i].prepare_state(
                        *next_obs[i][:7], _nf(next_obs[i]))
                    next_state = stackers[i].push(frame_next)
                    if terminal:
                        active[i] = False
                        final_status[i] = "goal" if next_obs[i][5] else "collision"
                    if states[i] is not None and raw_actions[i] is not None:
                        reward = next_obs[i][7]
                        shared_buffer.add(states[i], raw_actions[i], reward, terminal, next_state)

                obs_list = next_obs
                gs += 1

            for i in range(n):
                if final_status[i] is None:
                    final_status[i] = "timeout"
                train_stats["per_robot"][i]["sum_steps"] += robot_steps[i]
            merge_episode(train_stats, ep_metrics, final_status)
            global_episode += 1

            if global_episode % CONFIG["train_every_n"] == 0:
                if len(shared_buffer) >= CONFIG["batch_size"]:
                    for i in range(n):
                        models[i].train(shared_buffer, CONFIG["training_iterations"],
                                        CONFIG["batch_size"])

        elapsed = time.time() - epoch_start
        print(f"\nE{epoch:02d}  ep={global_episode}  time={elapsed:.0f}s  "
              f"buf={len(shared_buffer)}")
        print_stats(train_stats, "train")

        if (epoch >= CONFIG["eval_start_epoch"]
                and (epoch - CONFIG["eval_start_epoch"]) % CONFIG["eval_every_epochs"] == 0):
            print(f"  ── eval (epoch {epoch}) ──")
            for mode, tag in eval_modes:
                eval_stats = run_eval(env, models, mode)
                print_stats(eval_stats, tag)

        for i in range(n):
            save_i = save_dir / str(i)
            save_i.mkdir(parents=True, exist_ok=True)
            models[i].save(f"{save_name}_{i}", save_i)

        # Self-describing config snapshot — makes the model dir reproducible.
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "save_name": save_name,
                "scene": args.scene, "num_robots": n,
                "state_dim": args.state_dim, "extra_state_dim": extra_dim,
                "seq_len": args.seq_len, "net_input_dim": net_input_dim,
                "warmstart": args.warmstart, "old_state_dim": old_state_dim,
                "seed": args.seed, "git_commit": git_commit(),
                "config": CONFIG,
                "command": " ".join(sys.argv),
            }
            (save_dir / "experiment_meta.json").write_text(
                json.dumps(meta, indent=1, default=str))
        except Exception as exc:
            print(f"   ⚠️  experiment_meta.json write failed: {exc}")

        epoch += 1

    rclpy.shutdown()
    print("\n" + "=" * 60)
    print(f"✅ Training complete — {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
