"""Centralized experiment registry + shared constants.

Single source of truth for:
  * SEED (global random seed — deterministic layout regeneration)
  * Baseline values (fixes the "old vs new baseline" drift seen in eval scripts)
  * Experiment definitions (model dirs, state dims, eval modes, hyperparams)

Usage:
    from experiment_registry import SEED, get_exp, resolve_checkpoints, get_baselines

Every train / eval / dry-run script should read model paths, baselines, modes and
hyperparams from here instead of re-declaring them, so adding a new experiment is
one registry entry, not a copied script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
# Fixed global seed. Same seed + same (scene, num_robots, modes) → identical
# pre-sampled eval layouts, so regenerating after a config change is deterministic.
SEED = 42

# Where trained weights live (relative to the repo root).
MODELS_ROOT = Path("src/drl_navigation_ros2/models")

# Default eval protocol.
DEFAULT_EPISODES_PER_MODE = 20
DEFAULT_NUM_ROUNDS = 3
DEFAULT_MAX_STEPS = 420

# ---------------------------------------------------------------------------
# Authoritative baselines  (scene, num_robots, tag) → all_goal_rate
#
# scene1b 3-robot values are the NEW MultiRobotEnv zero-shot baselines
# (DECISIONS.md 2026-06-12: "旧基线不可再用"):
#     queue 0.13 / cross 0.48 / center 0.58 / random 0.40
# The 2-robot / scene0 values are legacy (old-env / May era) — kept centralized
# so they live in one place, but re-verify before relying on them.
# ---------------------------------------------------------------------------
BASELINES = {
    # ---- new (verified) ----
    ("scene1b", 3, "queue"):  0.13,
    ("scene1b", 3, "cross"):  0.48,
    ("scene1b", 3, "center"): 0.58,
    ("scene1b", 3, "random"): 0.40,
    # ---- legacy (old-env) ----
    ("scene1b", 2, "open"):   0.69,
    ("scene1b", 2, "cross"):  0.79,
    ("scene1b", 2, "center"): 0.74,
    ("scene0",  3, "parallel"): 0.62,
    ("scene0",  3, "cross"):    0.60,
    ("scene0",  3, "mixed"):    0.83,
}


def get_baselines(scene, num_robots):
    """Return {tag: baseline} for the given scene + robot count."""
    return {tag: v for (s, n, tag), v in BASELINES.items()
            if s == scene and n == num_robots}


# ---------------------------------------------------------------------------
# Eval mode lists
# ---------------------------------------------------------------------------
SCENE1B_MODES_3R = [
    ("scene1b_3r_queue",  "queue"),
    ("scene1b_3r_cross",  "cross"),
    ("scene1b_3r_center", "center"),
    ("scene1b_3r_random", "random"),
]

# Generic N-robot modes (work for any N: 3, 4, 5, ...). Registered in the env
# as scene1b_nr_* alongside the 3r aliases.
SCENE1B_MODES_NR = [
    ("scene1b_nr_queue",  "queue"),
    ("scene1b_nr_cross",  "cross"),
    ("scene1b_nr_center", "center"),
    ("scene1b_nr_random", "random"),
]

SCENE1B_MODES_2R = [
    ("bottleneck_open_eval",        "open"),
    ("bottleneck_cross_eval",       "cross"),
    ("bottleneck_center_goal_eval", "center"),
]

SCENE0_MODES_3R = [
    ("scene0_3r_parallel", "parallel"),
    ("scene0_3r_cross",    "cross"),
    ("scene0_3r_mixed",    "mixed"),
]

SCENE0_MODES_NR = [
    ("scene0_nr_parallel", "parallel"),
    ("scene0_nr_cross",    "cross"),
    ("scene0_nr_mixed",    "mixed"),
]

SCENE0_MODES_2R = [
    ("scene0_2r_parallel", "open"),
    ("scene0_2r_cross",    "cross"),
]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def _ckpt(model_dir, name, i):
    """Checkpoint source for robot i of a <model_dir>/<i>/<name>_<i> layout."""
    return (MODELS_ROOT / model_dir / str(i), f"{name}_{i}")


def _exp(**kw):
    """Fill defaults shared by all experiments."""
    defaults = dict(
        reward_mode="social_v1",
        episodes_per_mode=DEFAULT_EPISODES_PER_MODE,
        max_steps=DEFAULT_MAX_STEPS,
    )
    defaults.update(kw)
    return defaults


EXPERIMENTS = {
    # ---- 2-robot social-v1  (main model, also the zero-shot source for 3r) ----
    "social_v1_2r": _exp(
        scene="scene1b", num_robots=2,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            (MODELS_ROOT / "SAC_multi_robot_scene1b_bottleneck_social_v1_0",
             "SAC_multi_robot_scene1b_bottleneck_social_v1_0"),
            (MODELS_ROOT / "SAC_multi_robot_scene1b_bottleneck_social_v1_1",
             "SAC_multi_robot_scene1b_bottleneck_social_v1_1"),
        ],
        eval_modes=SCENE1B_MODES_2R,
    ),

    # ---- 3-robot Scene1b, 25-dim  ----
    "25dim_baseline_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_25dim_baseline",
                  "SAC_multi_robot_scene1b_3robot_25dim_baseline", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),
    "25dim_curriculum_v1_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_25dim_curriculum_v1",
                  "SAC_multi_robot_scene1b_3robot_25dim_curriculum_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),
    "25dim_curriculum_v1_extended_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_25dim_curriculum_v1_extended",
                  "SAC_multi_robot_scene1b_3robot_25dim_curriculum_v1_extended", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),
    "25dim_shared_buffer_v1_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_25dim_shared_buffer_v1",
                  "SAC_multi_robot_scene1b_3robot_25dim_shared_buffer_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),
    "25dim_shared_buffer_v1_extended_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_25dim_shared_buffer_v1_extended",
                  "SAC_multi_robot_scene1b_3robot_25dim_shared_buffer_v1_extended", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),

    # ---- 3-robot Scene1b, 29-dim  ----
    "29dim_expanded_v1_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=4,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_29dim_expanded_v1",
                  "SAC_multi_robot_scene1b_3robot_29dim_expanded_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),
    "29dim_shared_buffer_v1_3r": _exp(
        scene="scene1b", num_robots=3,
        state_dim=25, extra_state_dim=4,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_3robot_29dim_shared_buffer_v1",
                  "SAC_multi_robot_scene1b_3robot_29dim_shared_buffer_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_3R,
    ),

    # ---- 4-robot / 5-robot placeholders ----
    # Model dirs do not exist until you train with the parameterized train
    # script. Entries become usable once the checkpoint dirs are created.
    "25dim_4robot_v1": _exp(
        scene="scene1b", num_robots=4,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_4robot_25d_v1",
                  "SAC_multi_robot_scene1b_4robot_25d_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_NR,
    ),
    "31dim_4robot_neighbors_v1": _exp(
        scene="scene1b", num_robots=4,
        state_dim=25, extra_state_dim=6,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_4robot_31d_neighbors_v1",
                  "SAC_multi_robot_scene1b_4robot_31d_neighbors_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_NR,
    ),
    "25dim_5robot_v1": _exp(
        scene="scene1b", num_robots=5,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_5robot_25d_v1",
                  "SAC_multi_robot_scene1b_5robot_25d_v1", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_NR,
    ),

    # ---- 4-robot smoke model (trained with multi_robot_train.py, 1 epoch) ----
    "25dim_4robot_smoke_4r": _exp(
        scene="scene1b", num_robots=4,
        state_dim=25, extra_state_dim=0,
        checkpoints=[
            _ckpt("SAC_multi_robot_scene1b_4robot_25d_smoke_4r",
                  "SAC_multi_robot_scene1b_4robot_25d_smoke_4r", i) for i in range(2)
        ],
        eval_modes=SCENE1B_MODES_NR,
    ),
}


def get_exp(exp_id):
    """Return the experiment dict, or raise KeyError with a helpful message."""
    if exp_id not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment '{exp_id}'. Available: {sorted(EXPERIMENTS)}"
        )
    return EXPERIMENTS[exp_id]


def resolve_checkpoints(exp, num_robots=None):
    """Return list of (load_dir, load_name) — one per robot.

    Robots with index >= len(exp['checkpoints']) copy robot0, matching the
    existing "robot2 copies robot0" convention used across train/eval scripts.
    """
    if num_robots is None:
        num_robots = exp["num_robots"]
    ckpts = exp["checkpoints"]
    out = []
    for i in range(num_robots):
        out.append(ckpts[i] if i < len(ckpts) else ckpts[0])
    return out


def load_state_dim(exp):
    """Actual state dim the model was built with (state_dim + extra_state_dim)."""
    return exp["state_dim"] + exp["extra_state_dim"]


if __name__ == "__main__":
    for eid in sorted(EXPERIMENTS):
        exp = EXPERIMENTS[eid]
        print(f"{eid:38s} scene={exp['scene']:<8} N={exp['num_robots']} "
              f"state={load_state_dim(exp):3d}d  modes={[t for _, t in exp['eval_modes']]}")
