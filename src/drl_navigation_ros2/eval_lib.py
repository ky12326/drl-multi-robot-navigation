"""Shared evaluation library — one copy of the eval loop, used by every eval script.

Provides:
  * set_seed / git_commit          — reproducibility helpers
  * generate_or_load_layouts       — pre-sampled "fixed questions" for paired A/B eval
  * run_one_eval                   — the actual eval loop (with forced-layout replay)
  * print_final_table / save_results — console output + JSON/Markdown archiving

The eval scripts (multi_robot_eval_*.py) become thin wrappers around this module;
model paths / baselines / modes come from `experiment_registry`.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from SAC.SAC_utils import set_seed_everywhere
from SAC.frame_stack import FrameStacker
from experiment_registry import SEED, MODELS_ROOT

# Where pre-sampled eval layouts and results live.
_LAYOUTS_DIR = MODELS_ROOT.parent / "eval_layouts"
_RESULTS_DIR = MODELS_ROOT.parent / "eval_results"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    """Seed random / numpy / torch (+cuda) for reproducible runs."""
    set_seed_everywhere(seed)


def git_commit():
    """Return the current git HEAD short hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Fixed-question generation  (deterministic given seed + scene + num_robots)
# ---------------------------------------------------------------------------
def _layout_file_path(scene, num_robots, num_rounds, episodes_per_mode, seed):
    return _LAYOUTS_DIR / f"{scene}_{num_robots}r_{num_rounds}x{episodes_per_mode}_seed{seed}.json"


def _generate_layouts(scene, num_robots, num_rounds, episodes_per_mode, seed,
                      reward_mode, modes):
    """Sample valid layouts per eval mode using a headless env (no Gazebo, seconds).

    modes: list of (mode_name, tag). Returns
    {mode_name: [ [starts, targets, headings], ... ]} with
    num_rounds * episodes_per_mode entries per mode.
    """
    from multi_robot_env import MultiRobotEnv  # local import (needs ROS env, no Gazebo)

    env = MultiRobotEnv(num_robots=num_robots, scene=scene,
                        reward_mode=reward_mode, headless=True)

    need = num_rounds * episodes_per_mode
    result = {}
    for mode, tag in modes:
        env.set_reset_mode(mode)
        layouts = []
        attempts = 0
        while len(layouts) < need and attempts < need * 200:
            attempts += 1
            starts, targets, headings = env._sample_episode_layout()
            if starts is None:
                continue
            # Re-validate so a stale/failed probe can never sneak in.
            if not env._layout_valid(starts, targets):
                continue
            layouts.append([starts, targets, headings])
        if len(layouts) < need:
            raise RuntimeError(
                f"Layout generation failed for mode '{tag}': only {len(layouts)}/{need} "
                f"valid layouts after {attempts} attempts. Check the mode sampler.")
        result[mode] = layouts
        print(f"   [layouts] {tag}: {len(layouts)} valid layouts "
              f"(attempts={attempts})")
    return result


def generate_or_load_layouts(scene, num_robots, num_rounds=3, episodes_per_mode=20,
                             seed=SEED, reward_mode="social_v1", modes=None,
                             force=False):
    """Return pre-sampled layouts, generating + caching them if missing.

    Deterministic: same (scene, num_robots, rounds, eps, seed) → same file, so
    changing any parameter produces a NEW file and old experiments keep theirs.
    """
    _LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _layout_file_path(scene, num_robots, num_rounds, episodes_per_mode, seed)

    if path.exists() and not force:
        print(f"   [layouts] loading existing: {path}")
        data = json.loads(path.read_text())
        return data["layouts"]

    print(f"   [layouts] generating → {path} (seed={seed})")
    set_seed(seed)  # deterministic generation
    layouts = _generate_layouts(scene, num_robots, num_rounds, episodes_per_mode,
                                seed, reward_mode=reward_mode, modes=modes)
    path.write_text(json.dumps({
        "meta": {
            "scene": scene, "num_robots": num_robots,
            "num_rounds": num_rounds, "episodes_per_mode": episodes_per_mode,
            "seed": seed, "reward_mode": reward_mode,
        },
        "layouts": layouts,
    }, indent=1))
    print(f"   [layouts] saved {len(layouts)} modes → {path}")
    return layouts


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def new_stats(num_robots):
    return {
        "episodes": 0, "all_goal": 0, "any_collision": 0, "deadlock": 0,
        "sum_min_pairwise": 0.0, "valid_eps": 0,
        "per_robot": [
            {"goal": 0, "collision": 0, "timeout": 0, "sum_steps": 0}
            for _ in range(num_robots)
        ],
    }


def new_ep_metrics(num_robots):
    return {
        "min_pairwise": float("inf"), "no_progress_steps": 0,
        "prev_distances": [None] * num_robots,
    }


def update_episode_metrics(env, ep, num_robots):
    wp = env._last_world_position
    if wp is None:
        return
    pos = []
    for i in range(num_robots):
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


def merge_episode(stats, ep_metrics, final_status, num_robots):
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
    for i in range(num_robots):
        stats["per_robot"][i][final_status[i]] += 1


def summarize_stats(stats, num_robots):
    eps = max(stats["episodes"], 1)
    return {
        "all_goal": stats["all_goal"] / eps,
        "any_collision": stats["any_collision"] / eps,
        "deadlock": stats["deadlock"] / eps,
        "avg_pairwise_min_dist": stats["sum_min_pairwise"] / max(stats["valid_eps"], 1),
        "per_robot": [
            {
                "goal": s["goal"] / eps,
                "collision": s["collision"] / eps,
                "timeout": s["timeout"] / eps,
                "avg_steps": s["sum_steps"] / eps,
            }
            for s in stats["per_robot"]
        ],
    }


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
def run_one_eval(env, models, mode, episodes, max_steps,
                 layouts=None, extra_state_dim=0):
    """Run one eval pass for one mode.

    layouts: optional list of [starts, targets, headings]; when given, each
    episode replays one fixed question via env.set_forced_layout (paired A/B).
    """
    num_robots = env.num_robots
    # Frame stacking (optional; seq_len=1 → current single-frame behavior)
    seq_len = getattr(models[0], "seq_len", 1)
    frame_dim = getattr(models[0], "actual_state_dim", 0)
    stats = new_stats(num_robots)
    old_mode = env.reset_mode
    try:
        env.set_reset_mode(mode)
        stackers = ([FrameStacker(frame_dim, seq_len) for _ in range(num_robots)]
                    if seq_len > 1 else None)
        for ep_idx in range(episodes):
            if layouts is not None and len(layouts) > 0:
                starts, targets, headings = layouts[ep_idx % len(layouts)]
                env.set_forced_layout(starts, targets, headings)
            obs_list = env.reset()
            if stackers is not None:
                for i in range(num_robots):
                    stackers[i].reset()
                    frame0, _ = models[i].prepare_state(
                        *obs_list[i][:7], (obs_list[i][8] if extra_state_dim > 0 else None))
                    for _ in range(seq_len):
                        stackers[i].push(frame0)
            active = [True] * num_robots
            final_status = [None] * num_robots
            ep_metrics = new_ep_metrics(num_robots)
            robot_steps = [0] * num_robots
            gs = 0
            while gs < max_steps and any(active):
                cmds = []
                for i in range(num_robots):
                    if active[i]:
                        scan, dist, cos, sin, col, goal, a, _, nf = obs_list[i]
                        nf_use = nf if extra_state_dim > 0 else None
                        state, _ = models[i].prepare_state(scan, dist, cos, sin, col, goal, a, nf_use)
                        if stackers is not None:
                            state = stackers[i].push(state)
                        action = models[i].get_action(state, False)
                        cmds.append(((float(action[0]) + 1.0) / 2.0, float(action[1])))
                    else:
                        cmds.append((0.0, 0.0))
                next_obs = env.step(cmds, active_mask=active, debug_context=mode)
                update_episode_metrics(env, ep_metrics, num_robots)
                for i in range(num_robots):
                    if not active[i]:
                        continue
                    robot_steps[i] += 1
                    scan, dist, cos, sin, col, goal, a, _, nf = next_obs[i]
                    nf_use = nf if extra_state_dim > 0 else None
                    _, terminal = models[i].prepare_state(scan, dist, cos, sin, col, goal, a, nf_use)
                    if terminal:
                        active[i] = False
                        final_status[i] = "goal" if goal else "collision"
                obs_list = next_obs
                gs += 1
            for i in range(num_robots):
                if final_status[i] is None:
                    final_status[i] = "timeout"
                stats["per_robot"][i]["sum_steps"] += robot_steps[i]
            merge_episode(stats, ep_metrics, final_status, num_robots)
        return summarize_stats(stats, num_robots)
    finally:
        env.set_reset_mode(old_mode)


def run_rounds(env, models, modes, num_rounds, episodes_per_mode, max_steps,
               layouts=None, extra_state_dim=0, verbose=True):
    """Run num_rounds rounds over all modes.

    layouts: {mode_name: [layouts]} of length num_rounds * episodes_per_mode;
    each round uses a contiguous slice so both models see identical questions.
    Returns results_by_mode = {tag: [round_summary, ...]}.
    """
    results_by_mode = {tag: [] for _, tag in modes}
    for round_idx in range(num_rounds):
        round_start = time.time()
        if verbose:
            print(f"\n{'─' * 60}\nRound {round_idx + 1}/{num_rounds}\n{'─' * 60}")
        for mode, tag in modes:
            seg = None
            if layouts is not None and mode in layouts:
                seg = layouts[mode][
                    round_idx * episodes_per_mode:(round_idx + 1) * episodes_per_mode]
            s = run_one_eval(env, models, mode, episodes_per_mode, max_steps,
                             layouts=seg, extra_state_dim=extra_state_dim)
            results_by_mode[tag].append(s)
            if verbose:
                print_round_result(round_idx + 1, tag, s)
        if verbose:
            print(f"  Round {round_idx + 1} done in {time.time() - round_start:.0f}s")
    return results_by_mode


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_round_result(round_idx, tag, s):
    parts = [f"all_goal={s['all_goal']:.2f}  collision={s['any_collision']:.2f}"]
    parts.append(f"deadlock={s['deadlock']:.2f}  min_pair_dist={s['avg_pairwise_min_dist']:.2f}")
    for i, r in enumerate(s["per_robot"]):
        parts.append(f"r{i}: G={r['goal']:.2f} C={r['collision']:.2f} T={r['timeout']:.2f} s={r['avg_steps']:.0f}")
    print(f"  R{round_idx} [{tag:6s}] " + "  |  ".join(parts))


def _mode_summary(rounds):
    """Aggregate a list of per-round summaries into mean/std/peak/min."""
    ag = [r["all_goal"] for r in rounds]
    col = [r["any_collision"] for r in rounds]
    return {
        "mean": float(np.mean(ag)), "std": float(np.std(ag)),
        "peak": float(np.max(ag)), "min": float(np.min(ag)),
        "mean_collision": float(np.mean(col)),
    }


def print_final_table(results_by_mode, baselines, modes):
    print(f"\n{'=' * 90}")
    print("综合统计 (per-mode across rounds)")
    print(f"{'=' * 90}")
    header = (f"{'模式':<8} {'轮次均值':>8} {'σ':>6} {'峰值':>8} {'谷值':>8} "
              f"{'基线':>8} {'Δ均值':>8} {'Δ峰值':>8} {'碰撞均值':>8}")
    print(header)
    print("-" * 90)
    for mode, tag in modes:
        rounds = results_by_mode.get(tag, [])
        if not rounds:
            continue
        sm = _mode_summary(rounds)
        baseline = baselines.get(tag)
        b_str = f"{baseline:.2f}" if baseline is not None else "N/A"
        d_mean = f"{sm['mean'] - baseline:+.2f}" if baseline is not None else "N/A"
        d_peak = f"{sm['peak'] - baseline:+.2f}" if baseline is not None else "N/A"
        print(f"{tag:<8} {sm['mean']:>8.2f} {sm['std']:>6.2f} {sm['peak']:>8.2f} "
              f"{sm['min']:>8.2f} {b_str:>8} {d_mean:>8} {d_peak:>8} {sm['mean_collision']:>8.2f}")
    print("-" * 90)
    print("✅ eval complete")


def save_results(exp_id, results_by_mode, meta, baselines=None, modes=None):
    """Archive eval results as JSON + Markdown under eval_results/.

    meta: dict (exp_id, scene, num_robots, state_dim, model_label, layout_file,
                seed, git_commit, num_rounds, episodes_per_mode, max_steps, ...)
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{exp_id}_{ts}"
    json_path = _RESULTS_DIR / f"{stem}.json"
    md_path = _RESULTS_DIR / f"{stem}.md"

    # Build the machine-readable payload.
    payload = {
        "meta": dict(meta or {}),
        "baselines": baselines or {},
        "results": {
            tag: {
                "rounds": rounds,
                "summary": _mode_summary(rounds),
            }
            for tag, rounds in results_by_mode.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=1, default=str))

    # Human-readable markdown summary.
    lines = [f"# Eval: {exp_id}", ""]
    for k, v in (meta or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    if baselines:
        lines.append("## Baselines")
        lines.append("")
        for tag in sorted(baselines):
            lines.append(f"- {tag}: {baselines[tag]:.2f}")
        lines.append("")
    lines.append("## Results (mean ± std across rounds)")
    lines.append("")
    lines.append("| mode | mean | σ | peak | min | collision | baseline | Δmean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tag, r in payload["results"].items():
        s = r["summary"]
        b = (baselines or {}).get(tag)
        d = f"{s['mean'] - b:+.2f}" if b is not None else "N/A"
        b_s = f"{b:.2f}" if b is not None else "N/A"
        lines.append(f"| {tag} | {s['mean']:.2f} | {s['std']:.2f} | {s['peak']:.2f} "
                     f"| {s['min']:.2f} | {s['mean_collision']:.2f} | {b_s} | {d} |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\n📁 results saved →\n   {json_path}\n   {md_path}")
    return json_path, md_path


def load_models(exp, num_robots=None, device="cpu", swap=False):
    """Load one SAC per robot from a registry experiment (robot2+ copies robot0)."""
    from SAC.SAC import SAC
    import torch

    from experiment_registry import resolve_checkpoints as _resolve

    if num_robots is None:
        num_robots = exp["num_robots"]
    specs = _resolve(exp, num_robots)
    if swap and len(specs) >= 2:
        specs[0], specs[1] = specs[1], specs[0]
        print("   🔄 r0 ↔ r1 权重交换")
    models = []
    n_ckpts = len(exp["checkpoints"])
    for i, (load_dir, load_name) in enumerate(specs):
        models.append(SAC(
            state_dim=exp["state_dim"], action_dim=2, max_action=1,
            device=torch.device(device),
            save_every=0, load_model=True,
            extra_state_dim=exp["extra_state_dim"],
            save_directory=load_dir, model_name=load_name,
            load_directory=load_dir, load_name=load_name,
            log_dir=None,
        ))
        tag = " (copy r0)" if i >= n_ckpts else ""
        print(f"   robot{i}: {load_dir}{tag}")
    return models
