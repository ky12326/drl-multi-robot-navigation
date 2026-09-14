"""Unit tests for experiment_registry (pure Python, no ROS)."""

from experiment_registry import (
    EXPERIMENTS, get_exp, get_baselines, resolve_checkpoints, load_state_dim,
)


def test_all_experiments_resolve():
    assert EXPERIMENTS, "registry should not be empty"
    for eid in EXPERIMENTS:
        exp = get_exp(eid)
        assert exp["scene"] in ("scene0", "scene1b")
        assert exp["num_robots"] >= 2
        assert exp["eval_modes"], f"{eid} has no eval modes"


def test_resolve_checkpoints_pads_to_n():
    exp = get_exp("25dim_4robot_smoke_4r")
    specs = resolve_checkpoints(exp, 4)
    assert len(specs) == 4
    # robot0/1 own checkpoints; robot2/3 copy robot0 (project convention)
    assert specs[1] != specs[0]
    assert specs[2] == specs[0]
    assert specs[3] == specs[0]


def test_load_state_dim():
    assert load_state_dim(get_exp("25dim_baseline_3r")) == 25
    assert load_state_dim(get_exp("31dim_4robot_neighbors_v1")) == 31


def test_baselines_are_the_new_values():
    # New zero-shot baselines (DECISIONS 2026-06-12) — old values must NOT be used
    b = get_baselines("scene1b", 3)
    assert b["queue"] == 0.13
    assert b["cross"] == 0.48
    assert b["center"] == 0.58
    assert b["random"] == 0.40
