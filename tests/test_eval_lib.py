"""Unit tests for eval_lib (pure Python, no Gazebo)."""

import json

import numpy as np

from eval_lib import (
    new_stats, new_ep_metrics, merge_episode, summarize_stats,
    _mode_summary, save_results, run_rounds,
)


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------
def test_stats_aggregation():
    n = 3
    stats = new_stats(n)
    for _ in range(10):
        ep = new_ep_metrics(n)
        ep["min_pairwise"] = 1.5
        merge_episode(stats, ep, ["goal"] * n, n)
    s = summarize_stats(stats, n)
    assert s["all_goal"] == 1.0
    assert abs(s["avg_pairwise_min_dist"] - 1.5) < 1e-6
    assert s["per_robot"][0]["goal"] == 1.0


def test_mode_summary():
    rounds = [{"all_goal": 0.5, "any_collision": 0.2},
              {"all_goal": 0.7, "any_collision": 0.1}]
    sm = _mode_summary(rounds)
    assert abs(sm["mean"] - 0.6) < 1e-6
    assert abs(sm["peak"] - 0.7) < 1e-6
    assert abs(sm["min"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# save_results archiving
# ---------------------------------------------------------------------------
def test_save_results(tmp_path, monkeypatch):
    import eval_lib
    monkeypatch.setattr(eval_lib, "_RESULTS_DIR", tmp_path)
    results = {"queue": [{"all_goal": 0.5, "any_collision": 0.2, "deadlock": 0.1,
                          "avg_pairwise_min_dist": 1.5,
                          "per_robot": [{"goal": 0.5, "collision": 0.2,
                                         "timeout": 0.3, "avg_steps": 100}]}]}
    j, m = save_results("test_exp", results, meta={"exp_id": "test_exp", "seed": 42},
                        baselines={"queue": 0.13}, modes=[("q", "queue")])
    assert j.exists() and m.exists()
    data = json.loads(j.read_text())
    assert data["meta"]["exp_id"] == "test_exp"
    assert data["results"]["queue"]["summary"]["mean"] == 0.5
    assert "queue" in m.read_text()


# ---------------------------------------------------------------------------
# Paired fixed-question replay (mock env)
# ---------------------------------------------------------------------------
class _MockEnv:
    def __init__(self, num_robots=3):
        self.num_robots = num_robots
        self.reset_mode = "train"
        self._forced = None
        self._last_world_position = [None] * num_robots
        self.targets = [[0, 0]] * num_robots
        self.forced_seen = []

    def set_reset_mode(self, m):
        self.reset_mode = m

    def set_forced_layout(self, starts, targets, headings=None):
        self._forced = (starts, targets, headings)
        self.forced_seen.append(round(starts[0][0], 1))

    def reset(self):
        return [(np.full(180, 3.0, np.float32), 1.0, 1.0, 0.0, False, False,
                 [0.0, 0.0], 0.0, [0.0] * 4) for _ in range(self.num_robots)]

    def step(self, cmds, active_mask, debug_context="step"):
        return self.reset()


class _MockModel:
    def prepare_state(self, *a):
        return np.zeros(25, np.float32), False

    def get_action(self, state, add_noise=False):
        return np.array([0.0, 0.0])


def test_paired_replay_order():
    env = _MockEnv(3)
    models = [_MockModel() for _ in range(3)]
    layouts = {
        "q": [[[[0.1, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None],
              [[[0.2, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None],
              [[[0.3, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None],
              [[[0.4, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None]],
        "c": [[[[0.5, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None],
              [[[0.6, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None],
              [[[0.7, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None],
              [[[0.8, 0], [1, 0], [2, 0]], [[0, 0]] * 3, None]],
    }
    modes = [("q", "queue"), ("c", "cross")]
    run_rounds(env, models, modes, num_rounds=2, episodes_per_mode=2,
               max_steps=5, layouts=layouts, extra_state_dim=0, verbose=False)
    # Round0: q[0:2], c[0:2]; Round1: q[2:4], c[2:4] — same order every model
    assert env.forced_seen == [0.1, 0.2, 0.5, 0.6, 0.3, 0.4, 0.7, 0.8]
