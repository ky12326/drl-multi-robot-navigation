#!/usr/bin/env python3
"""Parameterized multi-robot environment — no inheritance, no 2-robot hardcoding.

Usage:
  env = MultiRobotEnv(num_robots=3, scene="scene1b", reward_mode="social_v1")
  obs_list = env.reset()
  obs_list = env.step(actions, active_mask)

Supports N >= 1 robots, scene="scene0"|"scene1b"|"scene2", reward_mode="social_v1"|"stage4".
"""

import math
import time
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from squaternion import Quaternion

from multi_robot_ros_nodes import (
    CmdVelPublisher,
    GazeboModelStateSubscriber,
    MarkerPublisher,
    PhysicsClient,
    ResetWorldClient,
    SensorSubscriber,
    SetModelStateClient,
)

# =============================================================================
# Scene geometry configurations  (copied from old env files — do not hand-edit)
# =============================================================================
SCENE_CONFIGS = {
    "scene1b": {
        "radius": 3.50,
        "obstacles": [
            {"center": [-1.90,  1.90], "half_size": [0.80, 0.80]},
            {"center": [ 1.90,  1.90], "half_size": [0.80, 0.80]},
            {"center": [-1.90, -1.90], "half_size": [0.80, 0.80]},
            {"center": [ 1.90, -1.90], "half_size": [0.80, 0.80]},
        ],
        # bottleneck zone used by conflict reward gate (optional)
        "bottleneck_zone": {"x_min": -1.75, "x_max": 1.75, "y_min": -1.75, "y_max": 1.75},
    },
    "scene0": {
        "radius": 3.20,
        "obstacles": [
            # From world file: multi_robot_drl_scene0_open_explicit_from_models.world
            {"center": [2.86, -3.00], "half_size": [0.50, 0.50]},
            {"center": [-2.93, 3.17], "half_size": [0.15, 0.35]},
            {"center": [-2.93, -3.00], "half_size": [0.125, 0.125]},
            {"center": [2.83, 2.93], "half_size": [1.00, 0.75]},
        ],
    },
}


# =============================================================================
# Fixed eval layouts  (fallback when random sampler fails)
# =============================================================================
_FIXED_SCENE1B_QUEUE_3R = [
    ([[-3.50, 2.00], [-3.50, 0.00], [-3.50, -2.00]], [[3.50, 2.00], [3.50, 0.00], [3.50, -2.00]]),
    ([[3.50, 2.00], [3.50, 0.00], [3.50, -2.00]], [[-3.50, 2.00], [-3.50, 0.00], [-3.50, -2.00]]),
    ([[2.00, -3.50], [0.00, -3.50], [-2.00, -3.50]], [[2.00, 3.50], [0.00, 3.50], [-2.00, 3.50]]),
    ([[2.00, 3.50], [0.00, 3.50], [-2.00, 3.50]], [[2.00, -3.50], [0.00, -3.50], [-2.00, -3.50]]),
]

_FIXED_SCENE1B_CROSS_3R = [
    ([[-3.50, 0.00], [3.50, 0.00], [0.00, -3.50]], [[3.50, 0.00], [-3.50, 0.00], [0.00, 3.50]]),
    ([[3.50, 0.00], [-3.50, 0.00], [0.00, 3.50]], [[-3.50, 0.00], [3.50, 0.00], [0.00, -3.50]]),
    ([[0.00, 3.50], [0.00, -3.50], [-3.50, 0.00]], [[0.00, -3.50], [0.00, 3.50], [3.50, 0.00]]),
    ([[0.00, -3.50], [0.00, 3.50], [3.50, 0.00]], [[0.00, 3.50], [0.00, -3.50], [-3.50, 0.00]]),
    ([[-3.50, 0.80], [3.50, -0.60], [0.50, -3.50]], [[3.50, -0.60], [-3.50, 0.80], [-0.30, 3.50]]),
]

_FIXED_SCENE1B_CENTER_3R = [
    ([[-3.50, 0.00], [3.50, 0.00], [0.00, -3.50]], [[0.70, 0.70], [-0.70, -0.50], [0.00, -0.90]]),
    ([[3.50, 0.00], [-3.50, 0.00], [0.50, 3.50]], [[-0.50, 0.80], [0.80, -0.50], [-0.90, 0.00]]),
    ([[-3.50, 0.00], [0.00, -3.50], [0.00, 3.50]], [[0.90, -0.40], [-0.40, 0.80], [0.00, 0.00]]),
]

_FIXED_SCENE0_PARALLEL_3R = [
    ([[-3.20, 1.00], [-3.20, 0.00], [-3.20, -1.00]], [[3.20, 1.00], [3.20, 0.00], [3.20, -1.00]]),
    ([[3.20, 1.00], [3.20, 0.00], [3.20, -1.00]], [[-3.20, 1.00], [-3.20, 0.00], [-3.20, -1.00]]),
]

_FIXED_SCENE0_CROSS_3R = [
    ([[-3.20, 0.00], [3.20, 0.00], [0.00, -3.20]], [[3.20, 0.00], [-3.20, 0.00], [0.00, 3.20]]),
    ([[3.20, 0.00], [-3.20, 0.00], [0.00, 3.20]], [[-3.20, 0.00], [3.20, 0.00], [0.00, -3.20]]),
    ([[0.00, 3.20], [0.00, -3.20], [-3.20, 0.00]], [[0.00, -3.20], [0.00, 3.20], [3.20, 0.00]]),
    ([[0.00, -3.20], [0.00, 3.20], [3.20, 0.00]], [[0.00, 3.20], [0.00, -3.20], [-3.20, 0.00]]),
    ([[-3.20, 0.70], [3.20, -0.50], [0.40, -3.20]], [[3.20, -0.50], [-3.20, 0.70], [-0.20, 3.20]]),
]

# =============================================================================
# Geometry helpers  (shared by all samplers)
# =============================================================================
class _Geom:
    @staticmethod
    def side_point(direction, offset, radius):
        return {"left": [-radius, offset], "right": [radius, offset],
                "bottom": [offset, -radius], "top": [offset, radius]}[direction]

    @staticmethod
    def opposite(d):
        return {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[d]

    @staticmethod
    def distribute_offsets(num_robots, spacing=1.0):
        """Return N evenly-spaced offsets centred on 0.  N=2→[-0.5,0.5], N=3→[-1,0,1]."""
        if num_robots == 1:
            return [0.0]
        step = spacing
        start = -(num_robots - 1) * step / 2.0
        return [start + i * step for i in range(num_robots)]


# =============================================================================
# MultiRobotEnv  (the ONE env class)
# =============================================================================
class MultiRobotEnv:
    """Parameterized multi-robot navigation environment.

    Parameters
    ----------
    num_robots : int
    scene : "scene0" | "scene1b" | "scene2"
    reward_mode : "social_v1" | "stage4"
        social_v1 = source + anti_stall + social_norm_proximity + conflict_bias
        stage4    = social_v1 + K-step lookahead + progressive yielding  (future)
    """

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------
    def __init__(
        self,
        num_robots: int = 2,
        scene: str = "scene1b",
        reward_mode: str = "social_v1",
        headless: bool = False,
        # geometry / goal
        init_target_distance: float = 2.0,
        target_dist_increase: float = 0.001,
        max_target_dist: float = 8.0,
        target_reached_delta: float = 0.30,
        collision_delta: float = 0.30,
        reset_target_min_dist: float = 1.5,
        # robot size
        robot_half_size: float = 0.133,
        robot_collision_radius: float = 0.28,
        # physics
        spawn_wall_margin: float = 0.50,
        wall_collision_limit: float = 3.98,
        reset_probe_min_scan: float = 0.50,
        # social norm (base)
        social_norm_enabled: bool = True,
        social_warning_dist: float = 1.20,
        social_safe_dist: float = 0.75,
        social_penalty_scale: float = 0.18,
        social_close_penalty_scale: float = 0.22,
        social_front_speed_penalty: float = 0.16,
        social_max_penalty: float = 0.55,
        # anti-stall
        progress_epsilon: float = 0.03,
        stall_step_threshold: int = 12,
        stall_penalty_per_step: float = 0.02,
        max_stall_penalty: float = 0.25,
        # conflict bias  (social_v1 subclass in old code)
        conflict_social_enabled: bool = True,
        conflict_reward_use_bottleneck_gate: bool = False,
        preferred_pass_side: str = "right",
        conflict_warning_dist: float = 1.65,
        conflict_safe_dist: float = 0.85,
        conflict_local_clearance_threshold: float = 1.35,
        conflict_head_on_cos_threshold: float = -0.35,
        conflict_front_score_threshold: float = 0.15,
        conflict_bias_penalty_scale: float = 0.045,
        conflict_close_penalty_scale: float = 0.060,
        conflict_preferred_side_discount: float = 0.35,
        conflict_opposite_side_extra: float = 0.15,
        conflict_deadlock_penalty: float = 0.040,
        conflict_deadlock_speed: float = 0.04,
        conflict_max_extra_penalty: float = 0.28,
        # misc
        verbose_debug: bool = False,
    ):
        # ---- scene identity ----
        self.scene = scene
        self.reward_mode = reward_mode
        if scene not in SCENE_CONFIGS:
            raise ValueError(f"Unknown scene '{scene}'. Choose: {list(SCENE_CONFIGS)}")
        sc = SCENE_CONFIGS[scene]
        self._scene_radius = sc["radius"]
        self.fixed_obstacles = sc["obstacles"]
        self.fixed_obstacle_positions = [o["center"].copy() for o in sc["obstacles"]]
        self.fixed_obstacle_half_sizes = [o["half_size"].copy() for o in sc["obstacles"]]
        self.bottleneck_zone = sc.get("bottleneck_zone", None)

        # ---- robot identity ----
        self.num_robots = num_robots
        self.namespaces = [f"robot{i + 1}" for i in range(num_robots)]
        self.robot_names = self.namespaces.copy()

        # ---- ROS nodes  (per-robot + shared) ----
        self._headless = headless
        if headless:
            self.cmd_publishers = []
            self.sensor_subscribers = []
            self.marker_publishers = []
            self.robot_state_client = None
            self.world_reset = None
            self.physics_client = None
            self.model_state_subscriber = None
        else:
            self.cmd_publishers = [CmdVelPublisher(namespace=ns) for ns in self.namespaces]
            self.sensor_subscribers = [SensorSubscriber(namespace=ns) for ns in self.namespaces]
            self.marker_publishers = [MarkerPublisher(namespace=ns) for ns in self.namespaces]
            self.robot_state_client = SetModelStateClient(namespace="scheduler")
            self.world_reset = ResetWorldClient(namespace="scheduler")
            self.physics_client = PhysicsClient(namespace="scheduler")
            self.model_state_subscriber = GazeboModelStateSubscriber()

        # ---- geometry / goal params ----
        self.robot_half_size = robot_half_size
        self.robot_collision_radius = robot_collision_radius
        self.spawn_wall_margin = spawn_wall_margin
        self.wall_collision_limit = wall_collision_limit
        self.reset_probe_min_scan = reset_probe_min_scan
        self.init_target_distance = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.reset_target_min_dist = reset_target_min_dist
        self.verbose_debug = verbose_debug

        # ---- per-robot dynamic state ----
        self.targets = [[0.0, 0.0] for _ in range(num_robots)]
        self.current_robot_positions = [[0.0, 0.0] for _ in range(num_robots)]
        self.current_target_positions = [[0.0, 0.0] for _ in range(num_robots)]
        self.target_dist = [init_target_distance for _ in range(num_robots)]
        self._last_scan = [None for _ in range(num_robots)]
        self._last_position = [None for _ in range(num_robots)]
        self._last_orientation = [None for _ in range(num_robots)]
        self._last_world_position = [None for _ in range(num_robots)]
        self._last_world_orientation = [None for _ in range(num_robots)]
        self.last_event_sources = ["none" for _ in range(num_robots)]
        self.last_reset_distances = [0.0 for _ in range(num_robots)]

        # anti-stall state
        self.progress_epsilon = progress_epsilon
        self.stall_step_threshold = stall_step_threshold
        self.stall_penalty_per_step = stall_penalty_per_step
        self.max_stall_penalty = max_stall_penalty
        self.prev_goal_distance = [None for _ in range(num_robots)]
        self.no_progress_steps = [0 for _ in range(num_robots)]

        # social norm state
        self.social_norm_enabled = social_norm_enabled
        self.social_warning_dist = social_warning_dist
        self.social_safe_dist = social_safe_dist
        self.social_penalty_scale = social_penalty_scale
        self.social_close_penalty_scale = social_close_penalty_scale
        self.social_front_speed_penalty = social_front_speed_penalty
        self.social_max_penalty = social_max_penalty

        # conflict bias state
        self.conflict_social_enabled = conflict_social_enabled
        self.conflict_reward_use_bottleneck_gate = conflict_reward_use_bottleneck_gate
        self.preferred_pass_side = preferred_pass_side
        self.conflict_warning_dist = conflict_warning_dist
        self.conflict_safe_dist = conflict_safe_dist
        self.conflict_local_clearance_threshold = conflict_local_clearance_threshold
        self.conflict_head_on_cos_threshold = conflict_head_on_cos_threshold
        self.conflict_front_score_threshold = conflict_front_score_threshold
        self.conflict_bias_penalty_scale = conflict_bias_penalty_scale
        self.conflict_close_penalty_scale = conflict_close_penalty_scale
        self.conflict_preferred_side_discount = conflict_preferred_side_discount
        self.conflict_opposite_side_extra = conflict_opposite_side_extra
        self.conflict_deadlock_penalty = conflict_deadlock_penalty
        self.conflict_deadlock_speed = conflict_deadlock_speed
        self.conflict_max_extra_penalty = conflict_max_extra_penalty

        # ---- template system ----
        self.reset_mode = "train"
        self.eval_template_counters = {}
        self._eval_obstacle_positions = None  # used by _clear_obstacle_eval_context
        self._train_sampler_fn = None
        self._eval_samplers = {}   # mode_name → sampler_fn
        self._fixed_layouts = {}   # mode_name → [(starts, targets), ...]
        self._register_default_samplers()

        # Layout hit tracking (custom = random sampler, fallback = fixed layout)
        self._hit = {"custom": 0, "fallback": 0}

        # Optional pre-sampled layout to replay on next reset ("fixed questions").
        self._forced_layout = None

        # Wait for first sensor frame — without this, first episode has no
        # LiDAR data and the robot sees an empty world (scan = all 7.0).
        # More robots take longer to be ready, so scale the timeout.
        if not headless:
            self._wait_for_sensors(timeout=5.0 * max(1, num_robots / 2))

        print(f"🤖 MultiRobotEnv: {num_robots} robots, scene={scene}, "
              f"reward={reward_mode}, collision_delta={collision_delta:.2f}")

    # ------------------------------------------------------------------
    # Template system
    # ------------------------------------------------------------------
    def _register_default_samplers(self):
        """Register all train + eval samplers for the configured scene."""
        if self.scene == "scene1b":
            self._register_scene1b_samplers()
        elif self.scene == "scene0":
            self._register_scene0_samplers()
        self._train_sampler_fn = self._sample_train_layout

    def register_eval_sampler(self, mode_name, sampler_fn, fixed_layouts=None):
        self._eval_samplers[mode_name] = sampler_fn
        if fixed_layouts:
            self._fixed_layouts[mode_name] = fixed_layouts

    # ------------------------------------------------------------------
    # Fixed-template helper  (N-robot compatible version of old _fixed_template)
    # ------------------------------------------------------------------
    def _fixed_template(self, mode: str, templates):
        """Cyclic dispatch from a list of fixed (starts, targets) pairs."""
        start_idx = self.eval_template_counters.get(mode, 0)
        for k in range(len(templates)):
            idx = (start_idx + k) % len(templates)
            robot_positions, target_positions = templates[idx]
            robot_positions = [[float(x), float(y)] for x, y in robot_positions]
            target_positions = [[float(x), float(y)] for x, y in target_positions]
            if not all(self._robot_spawn_valid_eval(robot_positions, i)
                       for i in range(len(robot_positions))):
                continue
            # Check all target-pair distances (was only [0]-[1] in old 2r code)
            pair_ok = True
            for a in range(len(target_positions)):
                for b in range(a + 1, len(target_positions)):
                    if np.linalg.norm(np.asarray(target_positions[a]) - np.asarray(target_positions[b])) < 0.80:
                        pair_ok = False
            if not pair_ok:
                continue
            checked = []
            ok = True
            for tgt in target_positions:
                if not self._target_valid_eval(tgt, robot_positions, checked, near_obstacle=False):
                    ok = False; break
                checked.append(tgt)
            if not ok: continue
            self.eval_template_counters[mode] = idx + 1
            headings = [self._heading_to_target(robot_positions[i], target_positions[i])
                        for i in range(len(robot_positions))]
            return robot_positions, target_positions, headings
        self.eval_template_counters[mode] = start_idx + 1
        return None, None, None

    # ==================================================================
    # Scene1b samplers  (2-robot: exact copies from old env;
    #                    3-robot: from eval script with N-robot adaptation)
    # ==================================================================
    def _register_scene1b_samplers(self):
        r = self._scene_radius      # 3.50 — for 3r / training
        r2 = 3.35                   # EXACT old _side_point default for 2r eval

        # -- 2-robot eval: EXACT copies of old MultiRobotSyncResetEnv samplers --
        def bottleneck_open_2r():
            self._clear_obstacle_eval_context()
            tmpls = []
            for direction in ["left", "right", "bottom", "top"]:
                opp = _Geom.opposite(direction)
                for lane_sign in [1.0, -1.0]:
                    rb = [_Geom.side_point(direction, -0.78 * lane_sign, r2),
                          _Geom.side_point(direction,  0.78 * lane_sign, r2)]
                    tb = [_Geom.side_point(opp, -0.45 * lane_sign, r2),
                          _Geom.side_point(opp,  0.45 * lane_sign, r2)]
                    tmpls.append((rb, tb))
            return self._fixed_template("bottleneck_open_eval", tmpls)

        def bottleneck_cross_2r():
            self._clear_obstacle_eval_context()
            tmpls = []
            for d0 in ["left", "right", "bottom", "top"]:
                for d1 in ["left", "right", "bottom", "top"]:
                    if d1 == d0: continue
                    rb = [_Geom.side_point(d0, -0.55, r2), _Geom.side_point(d1, -0.55, r2)]
                    tb = [_Geom.side_point(_Geom.opposite(d0), 0.55, r2),
                          _Geom.side_point(_Geom.opposite(d1), 0.55, r2)]
                    tmpls.append((rb, tb))
            return self._fixed_template("bottleneck_cross_eval", tmpls)

        def bottleneck_center_goal_2r():
            self._clear_obstacle_eval_context()
            tmpls = [
                ([[-3.35, -0.55], [ 3.35,  0.55]], [[ 0.45,  0.25], [-0.45, -0.25]]),
                ([[-3.35,  0.55], [ 3.35, -0.55]], [[ 0.45, -0.25], [-0.45,  0.25]]),
                ([[-0.55, -3.35], [ 0.55,  3.35]], [[ 0.25,  0.45], [-0.25, -0.45]]),
                ([[ 0.55, -3.35], [-0.55,  3.35]], [[-0.25,  0.45], [ 0.25, -0.45]]),
                ([[-3.35, -0.55], [ 0.55, -3.35]], [[ 0.45,  0.25], [-0.25,  0.45]]),
                ([[ 3.35,  0.55], [-0.55,  3.35]], [[-0.45, -0.25], [ 0.25, -0.45]]),
                ([[-3.35,  0.55], [ 0.55,  3.35]], [[ 0.45, -0.25], [-0.25, -0.45]]),
                ([[ 3.35, -0.55], [-0.55, -3.35]], [[-0.45,  0.25], [ 0.25,  0.45]]),
            ]
            return self._fixed_template("bottleneck_center_goal_eval", tmpls)

        self.register_eval_sampler("bottleneck_open_eval", bottleneck_open_2r)
        self.register_eval_sampler("bottleneck_cross_eval", bottleneck_cross_2r)
        self.register_eval_sampler("bottleneck_center_goal_eval", bottleneck_center_goal_2r)
        # Legacy aliases (exact names from old _sample_episode_layout dispatch)
        self.register_eval_sampler("open_eval", bottleneck_open_2r)
        self.register_eval_sampler("middle_open_eval", bottleneck_open_2r)
        self.register_eval_sampler("cross_eval", bottleneck_cross_2r)
        self.register_eval_sampler("middle_cross_eval", bottleneck_cross_2r)
        self.register_eval_sampler("center_goal_eval", bottleneck_center_goal_2r)
        self.register_eval_sampler("obstacle_goal_eval", bottleneck_center_goal_2r)
        self.register_eval_sampler("middle_obstacle_near_eval", bottleneck_center_goal_2r)

        # -- 3-robot eval: from eval script, generalized for N --
        def queue_nr():
            return self._sample_queue_nr(self.num_robots)
        def cross_nr():
            return self._sample_cross_nr(self.num_robots)
        def center_nr():
            return self._sample_center_nr(self.num_robots)
        def random_nr():
            return self._sample_random_nr(self.num_robots)

        self.register_eval_sampler("scene1b_3r_queue", queue_nr, _FIXED_SCENE1B_QUEUE_3R)
        self.register_eval_sampler("scene1b_3r_cross", cross_nr, _FIXED_SCENE1B_CROSS_3R)
        self.register_eval_sampler("scene1b_3r_center", center_nr, _FIXED_SCENE1B_CENTER_3R)
        self.register_eval_sampler("scene1b_3r_random", random_nr)

        # -- generic N-robot eval modes (any N: 3, 4, 5, ...) --
        self.register_eval_sampler("scene1b_nr_queue", queue_nr)
        self.register_eval_sampler("scene1b_nr_cross", cross_nr)
        self.register_eval_sampler("scene1b_nr_center", center_nr)
        self.register_eval_sampler("scene1b_nr_random", random_nr)

    # -- N-robot queue (same side → opposite) --
    def _sample_queue_nr(self, n):
        self._clear_obstacle_eval_context()
        r = self._scene_radius
        for _ in range(120):
            d = ["left", "right", "bottom", "top"][np.random.randint(4)]
            opp = _Geom.opposite(d)
            offsets = _Geom.distribute_offsets(n, spacing=1.0)
            np.random.shuffle(offsets)
            rb = [_Geom.side_point(d, o, r) for o in offsets]
            tb = [_Geom.side_point(opp, o + np.random.uniform(-0.12, 0.12), r) for o in offsets]
            result = self._instantiate_template(rb, tb, robot_jitter=(0.04, 0.04),
                                                 target_jitter=(0.04, 0.04),
                                                 near_obstacle_targets=False, max_trials=20)
            if result[0] is not None: return result
        return None, None, None

    # -- N-robot cross --
    def _sample_cross_nr(self, n):
        """For N robots: put pairs on opposite sides (headon), rest cross.

        Pairs use MIRRORED lanes (robot0 at lane on side d, robot1 at -lane on
        the opposite side), so each robot's target lands straight across at its
        own lane — keeping it >= 2*lane away from the opposing robot (this was
        the old bug: targets were 0.5-0.7m from the opposing robot, rejected).

        Safe for any N: pairs use dirs[p % 4] (cycles once n_pairs > 4), and
        the leftover robot never indexes past the 4-element direction pool.
        """
        self._clear_obstacle_eval_context()
        r = self._scene_radius
        attempts = 150
        for _ in range(attempts):
            dirs = ["left", "right", "bottom", "top"]
            np.random.shuffle(dirs)
            rb, tb = [], []
            used_sides = set()
            side_offsets = {}  # side -> offsets already placed on it
            n_pairs = n // 2
            leftover = n % 2
            for p in range(n_pairs):
                # Headon pair. dirs[p % 4] stays in-bounds for any n_pairs.
                d = dirs[p % 4]
                opp = _Geom.opposite(d)
                lane = float(np.random.uniform(0.75, 1.20) * np.random.choice([-1.0, 1.0]))
                rb.append(_Geom.side_point(d, lane, r))
                rb.append(_Geom.side_point(opp, -lane, r))
                tb.append(_Geom.side_point(opp, lane, r))
                tb.append(_Geom.side_point(d, -lane, r))
                used_sides.update([d, opp])
                side_offsets.setdefault(d, []).append(lane)
                side_offsets.setdefault(opp, []).append(-lane)
            if leftover:
                # Leftover single robot: prefer an unused side; when all sides
                # are taken, pick one and stay ~1.0m clear of its existing robots.
                remaining = [d for d in dirs if d not in used_sides]
                if remaining:
                    d = remaining[0]
                    off = float(np.random.uniform(-0.30, 0.30))
                else:
                    d = dirs[np.random.randint(4)]
                    off = 0.0
                    for e in side_offsets.get(d, []):
                        for cand in (e + 1.0, e - 1.0):
                            if abs(cand) <= 1.5:
                                off = cand
                                break
                        else:
                            continue
                        break
                opp = _Geom.opposite(d)
                rb.append(_Geom.side_point(d, off, r))
                tb.append(_Geom.side_point(opp, off, r))
            # Trim to exactly n (harmless; 2*n_pairs + leftover == n).
            rb, tb = rb[:n], tb[:n]
            result = self._instantiate_template(rb, tb, robot_jitter=(0.04, 0.04),
                                                 target_jitter=(0.04, 0.04),
                                                 near_obstacle_targets=False, max_trials=30)
            if result[0] is not None: return result
        return None, None, None

    # -- N-robot center --
    def _sample_center_nr(self, n):
        self._clear_obstacle_eval_context()
        r = self._scene_radius
        attempts = 300 if n > 4 else 200
        for _ in range(attempts):
            if n <= 4:
                # Original behavior for 2/3/4 robots (unchanged).
                dirs = ["left", "right", "bottom", "top"]
                np.random.shuffle(dirs); dirs = dirs[:n]
                rb = [_Geom.side_point(dirs[i], np.random.uniform(-0.95, 0.95), r)
                      for i in range(n)]
            else:
                # n > 4: cycle the 4 sides; robots sharing a side are offset
                # evenly (>= 0.9m apart) so the layout stays valid.
                sides = ["left", "right", "bottom", "top"]
                np.random.shuffle(sides)
                # First count how many robots land on each side (i % 4), then
                # assign each side's offsets from a single distribute_offsets
                # call so same-side robots are consistently spaced.
                side_counts = {s: 0 for s in sides}
                for i in range(n):
                    side_counts[sides[i % 4]] += 1
                side_offsets = {
                    s: _Geom.distribute_offsets(side_counts[s], spacing=1.0)
                    for s in sides
                }
                side_idx = {s: 0 for s in sides}
                rb = []
                for i in range(n):
                    side = sides[i % 4]
                    off = side_offsets[side][side_idx[side]]
                    side_idx[side] += 1
                    rb.append(_Geom.side_point(side, float(np.clip(off, -1.2, 1.2)), r))
            if n <= 4:
                tb = [[np.random.uniform(-0.95, 0.95), np.random.uniform(-0.95, 0.95)]
                      for _ in range(n)]
            else:
                # Structured ring targets (radius 0.75): pairwise arc distance
                # 2*0.75*sin(pi/n) >= 0.8 for n=5..8, keeping them all near center.
                tb = [[0.75 * np.cos(2 * np.pi * i / n),
                       0.75 * np.sin(2 * np.pi * i / n)] for i in range(n)]
            result = self._instantiate_template(rb, tb, robot_jitter=(0.04, 0.04),
                                                 target_jitter=(0.0, 0.0),
                                                 near_obstacle_targets=True, max_trials=30)
            if result[0] is not None: return result
        return None, None, None

    # -- N-robot random --
    def _sample_random_nr(self, n):
        self._clear_obstacle_eval_context()
        dirs_pool = ["left", "right", "bottom", "top"]
        r = self._scene_radius
        for _ in range(200):
            r_sides = [dirs_pool[np.random.randint(4)] for _ in range(n)]
            t_sides = [dirs_pool[np.random.randint(4)] for _ in range(n)]
            used = {}
            r_offsets = []
            for side in r_sides:
                off = np.random.uniform(-1.00, 1.00)
                if side in used:
                    off = used[side] + (1.00 if np.random.rand() < 0.5 else -1.00)
                    off = np.clip(off, -1.50, 1.50)
                used[side] = off
                r_offsets.append(off)
            # Space targets on the same side (>= 1.0m) to cut rejection rate at
            # higher N (5+ robots sharing 4 sides).
            t_side_counts = {}
            for s in t_sides:
                t_side_counts[s] = t_side_counts.get(s, 0) + 1
            t_side_offsets = {
                s: _Geom.distribute_offsets(t_side_counts[s], spacing=1.0)
                for s in t_side_counts
            }
            t_side_idx = {}
            t_offsets = []
            for s in t_sides:
                k = t_side_idx.get(s, 0)
                t_side_idx[s] = k + 1
                t_offsets.append(float(np.clip(t_side_offsets[s][k], -1.5, 1.5)))
            rb = [_Geom.side_point(r_sides[i], r_offsets[i], r) for i in range(n)]
            tb = [_Geom.side_point(t_sides[i], t_offsets[i], r) for i in range(n)]
            result = self._instantiate_template(rb, tb, robot_jitter=(0.03, 0.03),
                                                 target_jitter=(0.04, 0.04),
                                                 near_obstacle_targets=False, max_trials=30)
            if result[0] is not None: return result
        return None, None, None

    # ==================================================================
    # Scene0 samplers
    # ==================================================================
    def _register_scene0_samplers(self):
        r = self._scene_radius  # 3.20

        # 2-robot eval: same geometry as Scene1b but at Scene0 radius
        def parallel_2r():
            self._clear_obstacle_eval_context()
            tmpls = []
            for direction in ["left", "right", "bottom", "top"]:
                opp = _Geom.opposite(direction)
                for lane_sign in [1.0, -1.0]:
                    rb = [_Geom.side_point(direction, -0.78 * lane_sign, r),
                          _Geom.side_point(direction,  0.78 * lane_sign, r)]
                    tb = [_Geom.side_point(opp, -0.45 * lane_sign, r),
                          _Geom.side_point(opp,  0.45 * lane_sign, r)]
                    tmpls.append((rb, tb))
            return self._fixed_template("scene0_parallel", tmpls)

        def cross_2r():
            self._clear_obstacle_eval_context()
            tmpls = []
            for d0 in ["left", "right", "bottom", "top"]:
                for d1 in ["left", "right", "bottom", "top"]:
                    if d1 == d0: continue
                    rb = [_Geom.side_point(d0, -0.55, r), _Geom.side_point(d1, -0.55, r)]
                    tb = [_Geom.side_point(_Geom.opposite(d0), 0.55, r),
                          _Geom.side_point(_Geom.opposite(d1), 0.55, r)]
                    tmpls.append((rb, tb))
            return self._fixed_template("scene0_cross", tmpls)

        self.register_eval_sampler("scene0_2r_parallel", parallel_2r)
        self.register_eval_sampler("scene0_2r_cross", cross_2r)

        # 3-robot eval
        def parallel_nr():
            return self._sample_queue_nr(self.num_robots)  # same geometry as queue
        def cross_nr():
            return self._sample_cross_nr(self.num_robots)
        def mixed_nr():
            return self._sample_random_nr(self.num_robots)

        self.register_eval_sampler("scene0_3r_parallel", parallel_nr, _FIXED_SCENE0_PARALLEL_3R)
        self.register_eval_sampler("scene0_3r_cross", cross_nr, _FIXED_SCENE0_CROSS_3R)
        self.register_eval_sampler("scene0_3r_mixed", mixed_nr)

        # -- generic N-robot eval modes (any N: 3, 4, 5, ...) --
        self.register_eval_sampler("scene0_nr_parallel", parallel_nr)
        self.register_eval_sampler("scene0_nr_cross", cross_nr)
        self.register_eval_sampler("scene0_nr_mixed", mixed_nr)

    # ==================================================================
    # Training layout  (curriculum dispatch)
    # ==================================================================
    def _sample_train_layout(self):
        """Curriculum-adjusted distribution for 3-robot training.

        Original social-v1: 15% open + 40% center + 35% cross + 10% random.
        Adjusted to reduce policy drift on cross/center and add queue mode
        (the only mode that improved in 25-dim baseline training):

          25% open   — simple parallel, anchors 2-robot navigation skills
          25% center — dense central interaction
          20% cross   — reduced from 35% to limit social-v1 strategy drift
          20% queue   — same-side sequential entry (was implicit, now explicit)
          10% random  — mixed templates
        """
        r = np.random.random()
        if r < 0.25:
            return self._sample_open_train()
        elif r < 0.50:
            return self._sample_center_nr(self.num_robots)
        elif r < 0.70:
            return self._sample_cross_nr(self.num_robots)
        elif r < 0.90:
            return self._sample_queue_nr(self.num_robots)
        else:
            return self._sample_random_nr(self.num_robots)

    def _sample_open_train(self):
        """Simple parallel navigation: robots same side → opposite, spread out.

        At higher N (5+) tries progressively narrower spacing so the spread
        stays clear of corner obstacles in smaller arenas (e.g. scene0).
        """
        self._clear_obstacle_eval_context()
        r = self._scene_radius
        n = self.num_robots
        spacings = [1.2] if n <= 3 else [1.2, 1.0]
        for spacing in spacings:
            for _ in range(40):
                d = ["left", "right", "bottom", "top"][np.random.randint(4)]
                opp = _Geom.opposite(d)
                offsets = _Geom.distribute_offsets(n, spacing=spacing)
                np.random.shuffle(offsets)
                rb = [_Geom.side_point(d, o, r) for o in offsets]
                tb = [_Geom.side_point(opp, o + np.random.uniform(-0.15, 0.15), r) for o in offsets]
                result = self._instantiate_template(rb, tb, robot_jitter=(0.06, 0.06),
                                                     target_jitter=(0.06, 0.06),
                                                     near_obstacle_targets=False, max_trials=20)
                if result[0] is not None:
                    return result
        return None, None, None

    # ------------------------------------------------------------------
    # Layout dispatch
    # ------------------------------------------------------------------
    def _sample_episode_layout(self):
        mode = self.reset_mode
        if mode == "train":
            return self._train_sampler_fn()
        if mode in self._eval_samplers:
            sampler = self._eval_samplers[mode]
            n = self.num_robots
            # Try random sampler (attempts scale with robot count)
            attempts = 120 * max(1, n - 2)
            for _ in range(attempts):
                result = sampler()
                # Length filter: never pass a layout whose robot count != N
                # (guards against 2r-only samplers being invoked at N=4/5).
                if result[0] is not None and len(result[0]) == n:
                    self._hit["custom"] += 1
                    return result
            # Fallback to fixed layouts that match this robot count
            if mode in self._fixed_layouts:
                matching = [l for l in self._fixed_layouts[mode] if len(l[0]) == n]
                if matching:
                    self._hit["fallback"] += 1
                    layout = matching[np.random.randint(len(matching))]
                    starts = [list(s) for s in layout[0]]
                    targets = [list(t) for t in layout[1]]
                    headings = [None] * n
                    return starts, targets, headings
            # Generic N-robot fallback (any valid layout)
            self._hit["fallback"] += 1
            return self._generic_fallback_layout()
        raise ValueError(f"Unknown reset_mode: {mode}")

    def _generic_fallback_layout(self):
        """Return any valid N-robot layout, or (None, None, None) if exhausted.

        Used when a mode's random sampler fails and no fixed template matches
        the current robot count. Tries a spread queue then random, both of which
        are safe for any N.
        """
        n = self.num_robots
        for _ in range(30):
            starts, targets, headings = self._sample_queue_nr(n)
            if starts is not None and self._layout_valid(starts, targets):
                return starts, targets, headings
        for _ in range(30):
            starts, targets, headings = self._sample_random_nr(n)
            if starts is not None and self._layout_valid(starts, targets):
                return starts, targets, headings
        return None, None, None

    # ------------------------------------------------------------------
    # Layout validation  (parameterized — uses self.num_robots throughout)
    # ------------------------------------------------------------------
    def _layout_valid(self, robot_positions, target_positions):
        wall_limit = 4.0 - self.robot_half_size - max(0.35, self.spawn_wall_margin - 0.15)
        for i in range(len(robot_positions)):
            rx, ry = robot_positions[i]; tx, ty = target_positions[i]
            if np.linalg.norm([rx - tx, ry - ty]) < self.reset_target_min_dist:
                return False
            if self._inside_fixed_obstacle(rx, ry, margin_x=self.robot_half_size, margin_y=self.robot_half_size):
                return False
            if abs(rx) > wall_limit or abs(ry) > wall_limit:
                return False
        for x, y in target_positions:
            if abs(x) > 3.75 or abs(y) > 3.75:
                return False
            if self._inside_fixed_obstacle(x, y, margin_x=0.10, margin_y=0.10):
                return False
        for i in range(len(target_positions)):
            for j in range(i + 1, len(target_positions)):
                if np.linalg.norm(np.asarray(target_positions[i]) - np.asarray(target_positions[j])) < 0.80:
                    return False
        for i in range(len(robot_positions)):
            for j in range(i + 1, len(robot_positions)):
                if np.linalg.norm(np.asarray(robot_positions[i]) - np.asarray(robot_positions[j])) < max(0.90, 2.0 * self.robot_collision_radius + 0.10):
                    return False
        return True

    def _robot_spawn_valid_eval(self, robot_positions, idx: int) -> bool:
        x, y = robot_positions[idx]
        wall_limit = 4.0 - self.robot_half_size - max(0.35, self.spawn_wall_margin - 0.15)
        if abs(x) > wall_limit or abs(y) > wall_limit:
            return False
        if self._inside_fixed_obstacle(x, y, margin_x=self.robot_half_size, margin_y=self.robot_half_size):
            return False
        for j, other in enumerate(robot_positions):
            if j == idx: continue
            if np.linalg.norm(np.asarray([x, y]) - np.asarray(other)) < max(0.90, 2.0 * self.robot_collision_radius + 0.10):
                return False
        return True

    def _target_valid_eval(self, target, robot_positions, existing_targets, near_obstacle: bool = False) -> bool:
        x, y = target
        wall_margin = 0.28 if near_obstacle else 0.25
        if abs(x) > 4.0 - wall_margin or abs(y) > 4.0 - wall_margin:
            return False
        obstacle_margin = 0.18 if near_obstacle else 0.10
        if self._inside_fixed_obstacle(x, y, margin_x=obstacle_margin, margin_y=obstacle_margin):
            return False
        tgt_idx = len(existing_targets)
        # For 2 robots: strict check (old env behaviour).  For 3+ robots:
        # relax target→other-start distance so cross layouts are not rejected.
        other_start_min = self.reset_target_min_dist if self.num_robots <= 2 else 0.50
        for j, rp in enumerate(robot_positions):
            d = float(np.linalg.norm(np.asarray([x, y]) - np.asarray(rp)))
            if j == tgt_idx:
                if d < self.reset_target_min_dist:
                    return False
            else:
                if d < other_start_min:
                    return False
        for tp in existing_targets:
            if np.linalg.norm(np.asarray([x, y]) - np.asarray(tp)) < 0.80:
                return False
        return True

    def _instantiate_template(self, robot_bases, target_bases, *, robot_jitter=(0.10, 0.10),
                               target_jitter=(0.10, 0.10), near_obstacle_targets=False, max_trials=80):
        rb = [np.asarray(p, dtype=np.float32) for p in robot_bases]
        tb = [np.asarray(p, dtype=np.float32) for p in target_bases]
        for _ in range(max_trials):
            robot_positions = []
            for base in rb:
                x = float(base[0] + np.random.uniform(-robot_jitter[0], robot_jitter[0]))
                y = float(base[1] + np.random.uniform(-robot_jitter[1], robot_jitter[1]))
                robot_positions.append([x, y])
            if not all(self._robot_spawn_valid_eval(robot_positions, i) for i in range(len(robot_positions))):
                continue
            target_positions = []
            ok = True
            for base in tb:
                x = float(base[0] + np.random.uniform(-target_jitter[0], target_jitter[0]))
                y = float(base[1] + np.random.uniform(-target_jitter[1], target_jitter[1]))
                if not self._target_valid_eval([x, y], robot_positions, target_positions, near_obstacle=near_obstacle_targets):
                    ok = False; break
                target_positions.append([x, y])
            if not ok: continue
            headings = [self._heading_to_target(robot_positions[i], target_positions[i]) for i in range(len(robot_positions))]
            return robot_positions, target_positions, headings
        return None, None, None

    # ------------------------------------------------------------------
    # Reset flow
    # ------------------------------------------------------------------
    def set_reset_mode(self, mode: str = "train"):
        if mode != self.reset_mode and mode in self.eval_template_counters:
            self.eval_template_counters[mode] = 0
        self.reset_mode = mode

    def set_forced_layout(self, starts, targets, headings=None):
        """Replay a specific (pre-sampled) layout on the next reset.

        Used by eval to test fixed "questions" so A/B comparisons see identical
        initial conditions. The layout is consumed by the next reset() call.
        """
        self._forced_layout = (
            [list(s) for s in starts],
            [list(t) for t in targets],
            headings,
        )

    def reset(self):
        for pub in self.cmd_publishers:
            pub.publish_cmd_vel(0.0, 0.0)
        try:
            self.world_reset.reset_world()
        except Exception as exc:
            print(f"MultiRobotEnv: reset_world failed ({exc}), continuing")
        time.sleep(0.05)

        # Pre-sampled "fixed question" takes priority over random sampling.
        if self._forced_layout is not None:
            robot_positions, target_positions, headings = self._forced_layout
            self._forced_layout = None  # consume (one-shot)
            if self._layout_valid(robot_positions, target_positions):
                obs_list = self._apply_layout_and_probe(
                    robot_positions, target_positions, headings, attempt_label="forced")
                if obs_list is not None:
                    return obs_list
            print(f"MultiRobotEnv: forced layout rejected (mode={self.reset_mode}), "
                  "falling back to random sampling")

        max_attempts = min(20 * self.num_robots, 60)  # scale with robot count
        for attempt in range(max_attempts):
            result = self._sample_episode_layout()
            if result[0] is None or len(result[0]) != self.num_robots:
                continue
            robot_positions, target_positions, headings = result
            if not self._layout_valid(robot_positions, target_positions):
                continue
            obs_list = self._apply_layout_and_probe(robot_positions, target_positions, headings,
                                                     attempt_label=f"attempt={attempt+1}")
            if obs_list is not None:
                return obs_list

        # Exhausted — return safe fallback
        print(f"MultiRobotEnv: reset exhausted (mode={self.reset_mode}), returning safe obs")
        self.prev_goal_distance = [5.0 for _ in range(self.num_robots)]
        self.no_progress_steps = [0 for _ in range(self.num_robots)]
        self._clear_obstacle_eval_context()
        safe_obs = []
        extra_nf = [0.0] * (2 * (self.num_robots - 1))
        for _ in range(self.num_robots):
            safe_obs.append((np.asarray([7.0] * 180, dtype=np.float32), 5.0, 1.0, 0.0,
                             False, False, [0.0, 0.0], 0.0, extra_nf))
        return safe_obs

    def _apply_layout_and_probe(self, robot_positions, target_positions, headings, *, attempt_label=""):
        n = self.num_robots
        if len(robot_positions) != n:
            raise ValueError(f"robot_positions len {len(robot_positions)} != {n}")
        if len(target_positions) != n:
            raise ValueError(f"target_positions len {len(target_positions)} != {n}")
        if headings is None:
            headings = [None] * n
        elif len(headings) < n:
            print(f"MultiRobotEnv: headings len {len(headings)} < {n}, padding")
            headings = list(headings) + [None] * (n - len(headings))

        for i, robot_name in enumerate(self.robot_names):
            angle = headings[i] if headings[i] is not None else np.random.uniform(-np.pi, np.pi)
            self._set_entity_pose(robot_name, robot_positions[i][0], robot_positions[i][1], angle)
            self.current_robot_positions[i] = robot_positions[i]

        for i in range(n):
            self.targets[i] = target_positions[i]
            self.current_target_positions[i] = target_positions[i]
            self.last_reset_distances[i] = float(np.linalg.norm(
                np.asarray(robot_positions[i]) - np.asarray(target_positions[i])))
            self.prev_goal_distance[i] = self.last_reset_distances[i]
            self.no_progress_steps[i] = 0
            self.marker_publishers[i].publish(target_positions[i][0], target_positions[i][1])

        obs_list = self._probe_stable_state()
        for i, obs in enumerate(obs_list):
            latest_scan, distance, cos, sin, collision, goal, action, reward = obs[:8]
            min_scan = float(np.min(latest_scan)) if latest_scan is not None else 0.0
            if collision or goal or (min_scan < self.reset_probe_min_scan):
                return None
        return obs_list

    def _probe_stable_state(self):
        self._clear_runtime_cache()
        time.sleep(0.1)
        _ = self.step([(0.0, 0.0)] * self.num_robots,
                      active_mask=[False] * self.num_robots, debug_context="warmup")
        self._clear_runtime_cache()
        time.sleep(0.05)
        return self.step([(0.0, 0.0)] * self.num_robots,
                         active_mask=[True] * self.num_robots, debug_context="probe")

    # ------------------------------------------------------------------
    # Step  (with ALL 3-robot sensor fixes, parameterized for N robots)
    # ------------------------------------------------------------------
    def step(self, actions: List[Tuple[float, float]], active_mask: List[bool],
             debug_context: str = "step"):
        for i, pub in enumerate(self.cmd_publishers):
            lin, ang = actions[i] if i < len(actions) else (0.0, 0.0)
            pub.publish_cmd_vel(lin, ang)

        self.physics_client.unpause_physics()

        laser_hit_count = [0] * self.num_robots
        robot_hit_count = [0] * self.num_robots
        obstacle_hit_count = [0] * self.num_robots
        wall_hit_count = [0] * self.num_robots
        latest_scan = [None] * self.num_robots
        world_positions = [None] * self.num_robots
        world_orientations = [None] * self.num_robots
        min_scan_global = [7.0] * self.num_robots

        for _ in range(3):
            time.sleep(0.05)
            self._spin_all_once(timeout_sec=0.05)

            _ = self.model_state_subscriber.consume_model_updated()
            current_world_positions = []
            for i, robot_name in enumerate(self.robot_names):
                pos, orient = self.model_state_subscriber.get_model_pose(robot_name)
                current_world_positions.append(pos)
                if pos is not None:
                    world_positions[i] = pos
                    world_orientations[i] = orient
                    self._last_world_position[i] = pos
                    self._last_world_orientation[i] = orient

            # Fill model_state gaps from odometry (3-robot fix)
            for i in range(self.num_robots):
                if current_world_positions[i] is None:
                    current_world_positions[i] = self._last_world_position[i]

            robot_contacts = self._compute_robot_contacts(current_world_positions)
            for i in range(self.num_robots):
                if robot_contacts[i]:
                    robot_hit_count[i] += 1
                static_or_wall = self._check_static_or_wall_collision(current_world_positions[i])
                if static_or_wall == "obstacle":
                    obstacle_hit_count[i] += 1
                elif static_or_wall == "wall":
                    wall_hit_count[i] += 1

            for i, sub in enumerate(self.sensor_subscribers):
                # Dedicated spin for subscribers that miss callbacks (parameterized).
                # In the old code this was hardcoded for index 1 only.
                # New logic: last N-1 subscribers (i >= 1) get extra spin time;
                # with more than 3 robots everyone (incl. robot0) gets the boost,
                # since callback starvation can hit any subscriber at higher N.
                if self.num_robots > 3 or i >= 1:
                    for _ in range(5):
                        rclpy.spin_once(sub, timeout_sec=0.05)
                scan, pos, orient = sub.get_latest_sensor()
                if scan is not None:
                    scan = np.asarray(scan, dtype=np.float32)
                    scan = np.nan_to_num(scan, nan=7.0, posinf=7.0, neginf=0.0)
                    latest_scan[i] = scan
                    self._last_scan[i] = scan
                    min_scan_global[i] = min(min_scan_global[i], float(np.min(scan)))
                    if self._check_laser_collision(scan):
                        laser_hit_count[i] += 1
                if pos is not None and orient is not None:
                    self._last_position[i] = pos
                    self._last_orientation[i] = orient
                    # ALSO update world position from odometry (66 Hz) — critical for 3-robot
                    self._last_world_position[i] = pos
                    self._last_world_orientation[i] = orient

        self.physics_client.pause_physics()

        # ---- Build observations ----
        obs_list = []
        for i in range(self.num_robots):
            scan = latest_scan[i] if latest_scan[i] is not None else (
                self._last_scan[i] if self._last_scan[i] is not None
                else np.asarray([7.0] * 180, dtype=np.float32))

            # Prefer odometry (66 Hz) over model_state (1 Hz, may be stale/missing)
            wp = self._last_world_position[i] if self._last_world_position[i] is not None else world_positions[i]
            wo = self._last_world_orientation[i] if self._last_world_orientation[i] is not None else world_orientations[i]

            if wp is not None and wo is not None:
                distance, cos, sin, _ = self._get_dist_sincos(i, wp, wo)
            else:
                distance, cos, sin = 5.0, 1.0, 0.0

            if min_scan_global[i] == 7.0 and scan is not None:
                min_scan_global[i] = float(np.min(scan))

            laser_collision = laser_hit_count[i] >= 2
            robot_collision = robot_hit_count[i] >= 2
            obstacle_collision = obstacle_hit_count[i] >= 2
            wall_collision = (wall_hit_count[i] >= 2) and (min_scan_global[i] < self.collision_delta + 0.07)
            geometry_collision = robot_collision or obstacle_collision or wall_collision
            collision = laser_collision or geometry_collision

            # event source tracking
            if robot_collision:      self.last_event_sources[i] = "robot"
            elif obstacle_collision: self.last_event_sources[i] = "obstacle"
            elif wall_collision:     self.last_event_sources[i] = "wall"
            elif laser_collision:    self.last_event_sources[i] = "laser"
            else:                    self.last_event_sources[i] = "none"

            goal = active_mask[i] and (wp is not None and wo is not None) and self._check_goal(i, distance, collision)
            action_vec = [float(actions[i][0]), float(actions[i][1])]

            # ---- Reward (composition-based) ----
            reward = self._source_reward(goal, collision, action_vec, scan)
            reward += self._anti_stall_penalty(i, distance, action_vec, active=active_mask[i],
                                                goal=goal, collision=collision, debug_context=debug_context)
            reward += self._social_norm_reward(i, world_positions, world_orientations, actions,
                                                active_mask, goal=goal, collision=collision,
                                                debug_context=debug_context)
            # stage4 extras (K-step, progressive yielding) will be appended here in future

            # ---- Neighbor features ----
            neighbor_features = []
            if self.num_robots > 1:
                yaw_i = self._yaw_from_orientation(wo) if wo is not None else 0.0
                cos_h = math.cos(-yaw_i)
                sin_h = math.sin(-yaw_i)
                for j in range(self.num_robots):
                    if j == i: continue
                    wp_j = world_positions[j] if world_positions[j] is not None else self._last_world_position[j]
                    if wp is None or wp_j is None:
                        neighbor_features.extend([0.0, 0.0])
                        continue
                    dx = float(wp_j.x - wp.x)
                    dy = float(wp_j.y - wp.y)
                    ego_dx = (dx * cos_h - dy * sin_h) / 5.0
                    ego_dy = (dx * sin_h + dy * cos_h) / 5.0
                    neighbor_features.extend([ego_dx, ego_dy])

            obs_list.append((scan, distance, cos, sin, collision, goal, action_vec,
                             reward, neighbor_features))

        return obs_list

    # ------------------------------------------------------------------
    # Reward: source + anti-stall
    # ------------------------------------------------------------------
    def _source_reward(self, goal, collision, action, laser_scan):
        if goal:       return 100.0
        if collision:  return -100.0
        if laser_scan is None or len(laser_scan) == 0:
            min_scan = 7.0
        else:
            scan = np.asarray(laser_scan, dtype=np.float32)
            scan = np.nan_to_num(scan, nan=7.0, posinf=7.0, neginf=0.0)
            min_scan = float(np.min(scan))
        r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
        return float(action[0]) - abs(float(action[1])) / 2.0 - r3(min_scan) / 2.0

    def _anti_stall_penalty(self, robot_idx, distance, action, *, active, goal, collision, debug_context):
        if debug_context in ("warmup", "probe"):
            self.prev_goal_distance[robot_idx] = distance
            self.no_progress_steps[robot_idx] = 0
            return 0.0
        if (not active) or goal or collision:
            self.prev_goal_distance[robot_idx] = distance
            self.no_progress_steps[robot_idx] = 0
            return 0.0
        prev_distance = self.prev_goal_distance[robot_idx]
        self.prev_goal_distance[robot_idx] = distance
        if prev_distance is None:
            return 0.0
        progress = float(prev_distance - distance)
        linear_cmd = float(action[0])
        angular_cmd = abs(float(action[1]))
        stalled = (progress < self.progress_epsilon) or \
                  (progress < 0.05 and linear_cmd < 0.12 and angular_cmd > 0.45)
        if stalled:
            self.no_progress_steps[robot_idx] += 1
        else:
            self.no_progress_steps[robot_idx] = 0
        if self.no_progress_steps[robot_idx] < self.stall_step_threshold:
            return 0.0
        extra_steps = self.no_progress_steps[robot_idx] - self.stall_step_threshold + 1
        penalty = min(self.max_stall_penalty, self.stall_penalty_per_step * extra_steps)
        return -float(penalty)

    # ------------------------------------------------------------------
    # Reward: social_norm (base proximity + front-speed) + conflict bias
    # ------------------------------------------------------------------
    def _social_norm_reward(self, robot_idx, world_positions, world_orientations,
                            actions, active_mask, *, goal, collision, debug_context):
        """social_v1 reward = base proximity penalty + conflict bias."""
        base = self._social_norm_proximity(robot_idx, world_positions, world_orientations,
                                           actions, active_mask, goal=goal, collision=collision,
                                           debug_context=debug_context)
        extra = self._continuous_bias_conflict_reward(
            robot_idx, world_positions, world_orientations, actions, active_mask,
            goal=goal, collision=collision, debug_context=debug_context)
        return float(base + extra)

    def _social_norm_proximity(self, robot_idx, world_positions, world_orientations,
                               actions, active_mask, *, goal, collision, debug_context):
        if (not self.social_norm_enabled) or debug_context in ("warmup", "probe"):
            return 0.0
        if (not active_mask[robot_idx]) or goal or collision:
            return 0.0
        pi = world_positions[robot_idx]
        oi = world_orientations[robot_idx]
        if pi is None or oi is None:
            return 0.0
        yaw = self._yaw_from_orientation(oi)
        heading = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        linear_cmd = float(actions[robot_idx][0]) if robot_idx < len(actions) else 0.0
        total_penalty = 0.0
        for j in range(self.num_robots):
            if j == robot_idx or not active_mask[j]:
                continue
            pj = world_positions[j]
            if pj is None: continue
            rel = np.asarray([pj.x - pi.x, pj.y - pi.y], dtype=np.float32)
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6 or dist >= self.social_warning_dist:
                continue
            closeness = (self.social_warning_dist - dist) / max(self.social_warning_dist - self.social_safe_dist, 1e-6)
            closeness = float(np.clip(closeness, 0.0, 1.0))
            penalty = self.social_penalty_scale * (closeness ** 2)
            if dist < self.social_safe_dist:
                close_ratio = (self.social_safe_dist - dist) / max(self.social_safe_dist, 1e-6)
                penalty += self.social_close_penalty_scale * float(np.clip(close_ratio, 0.0, 1.0))
            rel_unit = rel / max(dist, 1e-6)
            front_score = float(np.dot(heading, rel_unit))
            if front_score > 0.25 and linear_cmd > 0.10:
                penalty += self.social_front_speed_penalty * front_score * linear_cmd * closeness
            total_penalty += penalty
        return -float(min(self.social_max_penalty, total_penalty))

    def _continuous_bias_conflict_reward(self, robot_idx, world_positions, world_orientations,
                                          actions, active_mask, *, goal, collision, debug_context):
        if (not self.conflict_social_enabled) or debug_context in ("warmup", "probe"):
            return 0.0
        if (not active_mask[robot_idx]) or goal or collision:
            return 0.0
        pi = world_positions[robot_idx]
        oi = world_orientations[robot_idx]
        if pi is None or oi is None:
            return 0.0
        yaw_i = self._yaw_from_orientation(oi)
        heading_i = np.asarray([np.cos(yaw_i), np.sin(yaw_i)], dtype=np.float32)
        lin_i = float(actions[robot_idx][0]) if robot_idx < len(actions) else 0.0
        min_scan_i = self._min_scan_for_reward(robot_idx)
        total_penalty = 0.0
        for j in range(self.num_robots):
            if j == robot_idx or not active_mask[j]:
                continue
            pj = world_positions[j]
            oj = world_orientations[j]
            if pj is None or oj is None: continue
            rel = np.asarray([float(pj.x - pi.x), float(pj.y - pi.y)], dtype=np.float32)
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6 or dist >= self.conflict_warning_dist:
                continue
            if self.conflict_reward_use_bottleneck_gate and self.bottleneck_zone is not None:
                bz = self.bottleneck_zone
                in_bn = ((bz["x_min"] <= pi.x <= bz["x_max"] and bz["y_min"] <= pi.y <= bz["y_max"]) or
                         (bz["x_min"] <= pj.x <= bz["x_max"] and bz["y_min"] <= pj.y <= bz["y_max"]))
                if not in_bn: continue
            yaw_j = self._yaw_from_orientation(oj)
            heading_j = np.asarray([np.cos(yaw_j), np.sin(yaw_j)], dtype=np.float32)
            rel_unit = rel / max(dist, 1e-6)
            heading_cos = float(np.dot(heading_i, heading_j))
            front_i = float(np.dot(heading_i, rel_unit))
            front_j = float(np.dot(heading_j, -rel_unit))
            head_on_or_crossing = (heading_cos < self.conflict_head_on_cos_threshold) or \
                (front_i > self.conflict_front_score_threshold and front_j > self.conflict_front_score_threshold)
            if not head_on_or_crossing: continue
            min_scan_j = self._min_scan_for_reward(j)
            locally_constrained = min(min_scan_i, min_scan_j) < self.conflict_local_clearance_threshold
            if not locally_constrained: continue
            closeness = (self.conflict_warning_dist - dist) / max(self.conflict_warning_dist - self.conflict_safe_dist, 1e-6)
            closeness = float(np.clip(closeness, 0.0, 1.0))
            local_y = float(heading_i[0] * rel_unit[1] - heading_i[1] * rel_unit[0])
            z = float(np.clip((local_y + 1.0) / 2.0, 0.0, 1.0))
            if self.preferred_pass_side == "right":
                side_scale = 1.0 - self.conflict_preferred_side_discount * (1.0 - z) + self.conflict_opposite_side_extra * z
            else:
                side_scale = 1.0 - self.conflict_preferred_side_discount * z + self.conflict_opposite_side_extra * (1.0 - z)
            side_scale = float(np.clip(side_scale, 0.50, 1.30))
            penalty = self.conflict_bias_penalty_scale * side_scale * (closeness ** 2)
            if dist < self.conflict_safe_dist:
                close_ratio = (self.conflict_safe_dist - dist) / max(self.conflict_safe_dist, 1e-6)
                penalty += self.conflict_close_penalty_scale * float(np.clip(close_ratio, 0.0, 1.0))
            lin_j = float(actions[j][0]) if j < len(actions) else 0.0
            if lin_i < self.conflict_deadlock_speed and lin_j < self.conflict_deadlock_speed:
                penalty += self.conflict_deadlock_penalty * closeness
            total_penalty += penalty
        return -float(min(self.conflict_max_extra_penalty, total_penalty))

    def _min_scan_for_reward(self, robot_idx):
        scan = self._last_scan[robot_idx] if robot_idx < len(self._last_scan) else None
        if scan is None or len(scan) == 0:
            return 7.0
        arr = np.asarray(scan, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=7.0, posinf=7.0, neginf=0.0)
        return float(np.min(arr))

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def _check_laser_collision(self, laser_scan):
        if laser_scan is None or len(laser_scan) == 0:
            return False
        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=7.0, posinf=7.0, neginf=0.0)
        return float(np.min(scan)) < self.collision_delta

    def _check_static_or_wall_collision(self, pos):
        if pos is None: return None
        robot_x, robot_y = pos.x, pos.y
        for (ox, oy), (hx, hy) in zip(self.fixed_obstacle_positions, self.fixed_obstacle_half_sizes):
            if abs(robot_x - ox) < hx + self.robot_half_size and abs(robot_y - oy) < hy + self.robot_half_size:
                return "obstacle"
        if abs(robot_x) > self.wall_collision_limit or abs(robot_y) > self.wall_collision_limit:
            return "wall"
        return None

    def _compute_robot_contacts(self, robot_positions_world):
        contacts = [False] * self.num_robots
        for i in range(self.num_robots):
            pi = robot_positions_world[i]
            if pi is None: continue
            for j in range(i + 1, self.num_robots):
                pj = robot_positions_world[j]
                if pj is None: continue
                if np.linalg.norm([pi.x - pj.x, pi.y - pj.y]) < 2.0 * self.robot_collision_radius:
                    contacts[i] = True
                    contacts[j] = True
        return contacts

    def _check_goal(self, robot_idx, distance, collision):
        if collision: return False
        if distance < self.target_reached_delta:
            self.target_dist[robot_idx] += self.target_dist_increase
            if self.target_dist[robot_idx] > self.max_target_dist:
                self.target_dist[robot_idx] = self.max_target_dist
            return True
        return False

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------
    def _wait_for_sensors(self, timeout: float = 5.0) -> bool:
        """Spin until every robot has scan + position + orientation."""
        start = time.time()
        ready = [False] * self.num_robots
        while time.time() - start < timeout:
            self._spin_all_once(timeout_sec=0.05)
            for i, sub in enumerate(self.sensor_subscribers):
                scan, pos, orient = sub.get_latest_sensor()
                if scan is not None and pos is not None and orient is not None:
                    self._last_scan[i] = scan
                    self._last_position[i] = pos
                    self._last_orientation[i] = orient
                    ready[i] = True
            if all(ready):
                return True
            time.sleep(0.05)
        print("⚠️ 部分机器人传感器首帧未就绪，将使用回退策略继续")
        return False

    def _spin_all_once(self, timeout_sec=0.02):
        if self._headless: return
        for sub in self.sensor_subscribers:
            try:
                rclpy.spin_once(sub, timeout_sec=timeout_sec)
            except Exception:
                pass
        try:
            rclpy.spin_once(self.model_state_subscriber, timeout_sec=timeout_sec)
        except Exception:
            pass

    def _clear_runtime_cache(self):
        self._last_scan = [None] * self.num_robots
        self._last_position = [None] * self.num_robots
        self._last_orientation = [None] * self.num_robots
        self._last_world_position = [None] * self.num_robots
        self._last_world_orientation = [None] * self.num_robots
        if self._headless: return
        for sub in self.sensor_subscribers:
            sub.odom_updated = False
        self.model_state_subscriber.latest_names = []
        self.model_state_subscriber.latest_poses = []
        self.model_state_subscriber.latest_twists = []
        self.model_state_subscriber.model_updated = False

    def _sync_world_from_odom(self):
        """Pull latest sensor data from each subscriber into _last_* caches.

        Called by eval loop between reset/step and metrics collection, so
        _update_metrics sees the most recent odometry positions without
        needing to re-spin during step.
        """
        for i, sub in enumerate(self.sensor_subscribers):
            scan, pos, orient = sub.get_latest_sensor()
            if scan is not None:
                self._last_scan[i] = np.asarray(scan, dtype=np.float32)
            if pos is not None:
                self._last_position[i] = pos
                self._last_world_position[i] = pos
            if orient is not None:
                self._last_orientation[i] = orient
                self._last_world_orientation[i] = orient

    def _clear_obstacle_eval_context(self):
        self._eval_obstacle_positions = None

    # ------------------------------------------------------------------
    # Geometry / pose helpers
    # ------------------------------------------------------------------
    def _set_entity_pose(self, name, x, y, angle):
        if self._headless: return
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        quaternion = Quaternion.from_euler(0, 0, angle)
        pose.orientation.w = float(quaternion.w)
        pose.orientation.x = float(quaternion.x)
        pose.orientation.y = float(quaternion.y)
        pose.orientation.z = float(quaternion.z)
        self.robot_state_client.set_state(name, pose)

    def _heading_to_target(self, pos, target):
        return float(np.arctan2(target[1] - pos[1], target[0] - pos[0]))

    def _inside_fixed_obstacle(self, x, y, margin_x=0.0, margin_y=0.0):
        for (ox, oy), (hx, hy) in zip(self.fixed_obstacle_positions, self.fixed_obstacle_half_sizes):
            if abs(x - ox) < hx + margin_x and abs(y - oy) < hy + margin_y:
                return True
        return False

    def _yaw_from_orientation(self, orientation):
        if orientation is None: return 0.0
        quaternion = Quaternion(orientation.w, orientation.x, orientation.y, orientation.z)
        return float(quaternion.to_euler(degrees=False)[2])

    @staticmethod
    def _cossin(vec1, vec2):
        v1_norm = vec1 / max(np.linalg.norm(vec1), 1e-8)
        v2_norm = vec2 / max(np.linalg.norm(vec2), 1e-8)
        cos = np.dot(v1_norm, v2_norm)
        cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
        return float(cos), float(cross) / max(abs(cross), 1e-8) if abs(cross) < 1e-8 else float(cross)

    def _get_dist_sincos(self, robot_idx, world_position, world_orientation):
        odom_x, odom_y = world_position.x, world_position.y
        quaternion = Quaternion(world_orientation.w, world_orientation.x,
                                world_orientation.y, world_orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.targets[robot_idx][0] - odom_x,
                       self.targets[robot_idx][1] - odom_y]
        distance = np.linalg.norm(goal_vector)
        cos, sin = self._cossin(pose_vector, goal_vector)
        return distance, cos, sin, angle
