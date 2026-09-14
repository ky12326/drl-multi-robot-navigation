"""Layout-generation tests for N=2..5 (headless env, no Gazebo).

Skips the whole module if ROS (rclpy) isn't importable, so the pure-logic
tests (frame_stack / registry / eval_lib / sac_utils) still run standalone.
"""

import pytest

pytest.importorskip("rclpy")  # multi_robot_env needs rclpy importable

from multi_robot_env import MultiRobotEnv            # noqa: E402
from experiment_registry import SCENE1B_MODES_NR, SCENE0_MODES_NR  # noqa: E402

SCENES_N = [
    ("scene0", 2), ("scene0", 3), ("scene0", 4), ("scene0", 5),
    ("scene1b", 2), ("scene1b", 3), ("scene1b", 4), ("scene1b", 5),
]


def _modes(scene):
    return ([("train", "train")] +
            (SCENE1B_MODES_NR if scene == "scene1b" else SCENE0_MODES_NR))


@pytest.mark.parametrize("scene,n", SCENES_N)
def test_layouts_no_crash_and_valid(scene, n):
    """Every (scene, N) mode must produce valid layouts without IndexError."""
    env = MultiRobotEnv(num_robots=n, scene=scene, headless=True)
    for mode, tag in _modes(scene):
        env.set_reset_mode(mode)
        ok = 0
        for _ in range(20):
            r = env._sample_episode_layout()
            if r[0] is not None and len(r[0]) == n:
                ok += 1
        assert ok / 20 >= 0.7, f"{scene} N={n} {tag}: hit rate {ok/20:.0%}"
