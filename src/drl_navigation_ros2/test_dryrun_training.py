#!/usr/bin/env python3
"""Dry-run training loop verification (no Gazebo).

Simulates the full training loop for N episodes with a mock environment
that returns realistic observations.  Catches interface errors, dimension
mismatches, and logic bugs before launching real training.

Now parameterized (num_robots / state_dim / extra_state_dim / seq_len /
shared_buffer) and extended with new checks: reset_mode switching (headless
real env), prepare_state embedding of neighbor features, and seq_len
frame-stacking shapes.

Usage:
  source ~/mapf/bin/activate
  python3 src/drl_navigation_ros2/test_dryrun_training.py
  python3 src/drl_navigation_ros2/test_dryrun_training.py --num-robots 4 --seq-len 3
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch

_BASE = Path(__file__).parent
sys.path.append(str(_BASE))

from SAC.SAC import SAC
from SAC.frame_stack import FrameStacker
from replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Mock environment — simulates Gazebo physics with simple kinematics
# ---------------------------------------------------------------------------
class MockGazeboEnv:
    """Returns realistic observations; robots drift toward targets each step."""

    def __init__(self, num_robots=3, arena_size=4.0):
        self.num_robots = num_robots
        self.arena_size = arena_size
        self.reset_mode = "train"
        self.targets = None
        self.positions = None
        self.headings = None
        self._step_counter = 0
        self._episode_counter = 0
        self._last_actions = [(0.0, 0.0)] * num_robots

    def set_reset_mode(self, mode):
        self.reset_mode = mode

    def reset(self):
        self._episode_counter += 1
        self._step_counter = 0
        self.positions = []
        self.headings = []
        for _ in range(self.num_robots):
            side = np.random.randint(4)
            if side == 0:
                x, y = -3.2, np.random.uniform(-2.0, 2.0)
                h = 0.0
            elif side == 1:
                x, y = 3.2, np.random.uniform(-2.0, 2.0)
                h = np.pi
            elif side == 2:
                x, y = np.random.uniform(-2.0, 2.0), -3.2
                h = np.pi / 2
            else:
                x, y = np.random.uniform(-2.0, 2.0), 3.2
                h = -np.pi / 2
            self.positions.append([x, y])
            self.headings.append(h)

        self.targets = []
        for p in self.positions:
            tx = -p[0] + np.random.uniform(-0.5, 0.5)
            ty = -p[1] + np.random.uniform(-0.5, 0.5)
            tx = np.clip(tx, -3.5, 3.5)
            ty = np.clip(ty, -3.5, 3.5)
            self.targets.append([tx, ty])

        self._last_actions = [(0.0, 0.0)] * self.num_robots
        return self._build_obs_list()

    def step(self, actions, active_mask, debug_context="step"):
        self._step_counter += 1
        self._last_actions = [(float(a[0]), float(a[1])) for a in actions]

        dt = 0.15
        for i in range(self.num_robots):
            if not active_mask[i]:
                continue
            lin = (actions[i][0] + 1.0) / 2.0
            ang = actions[i][1]
            self.headings[i] += ang * dt * 2.0
            speed = lin * 0.3
            self.positions[i][0] += speed * np.cos(self.headings[i]) * dt
            self.positions[i][1] += speed * np.sin(self.headings[i]) * dt

        return self._build_obs_list()

    def _build_obs_list(self):
        obs_list = []
        for i in range(self.num_robots):
            px, py = self.positions[i]
            tx, ty = self.targets[i]
            h = self.headings[i]

            dx, dy = tx - px, ty - py
            dist = float(np.hypot(dx, dy))
            goal_angle = float(np.arctan2(dy, dx))
            cos = float(np.cos(goal_angle - h))
            sin = float(np.sin(goal_angle - h))

            scan = np.full(180, 3.0, dtype=np.float32)
            scan += np.random.uniform(-0.3, 0.3, 180).astype(np.float32)
            scan = np.clip(scan, 0.15, 7.0)

            collision = dist < 0.25 or float(np.min(scan)) < 0.25
            goal = dist < 0.30 and not collision

            # Neighbor features (ego frame, normalized by /5.0)
            nf = []
            for j in range(self.num_robots):
                if j == i:
                    continue
                pjx, pjy = self.positions[j]
                dx_n = pjx - px
                dy_n = pjy - py
                cos_h = np.cos(-h)
                sin_h = np.sin(-h)
                ego_dx = (dx_n * cos_h - dy_n * sin_h) / 5.0
                ego_dy = (dx_n * sin_h + dy_n * cos_h) / 5.0
                nf.extend([float(ego_dx), float(ego_dy)])

            prev_action = self._last_actions[i] if hasattr(self, '_last_actions') and i < len(self._last_actions) else [0.0, 0.0]
            reward = 100.0 if goal else (-100.0 if collision else 0.0)

            obs_list.append((scan, dist, cos, sin, collision, goal,
                             list(prev_action), reward, nf))

        return obs_list


# ---------------------------------------------------------------------------
# Dry-run: mirrors the exact training loop logic
# ---------------------------------------------------------------------------
def dryrun(num_robots=3, state_dim=25, extra_state_dim=None, seq_len=1,
           shared_buffer=True, num_episodes=10, max_steps=60,
           train_every=2, training_iterations=10, batch_size=20):
    """Run a shortened training loop with mock env.

    Returns True if everything passed, False if something failed.
    Checks:
      1. Model construction + warm-start (parameterized dims / seq_len)
      2. Buffer construction
      3. env.reset() / env.step() return correct shapes
      4. prepare_state (frame_dim) + stacking + get_action (noisy + det)
      5-6. Full training loop (with frame stacking when seq_len > 1)
      7. Model save
      8. reset_mode switching (headless real env)
      9. prepare_state embeds neighbor features at the tail
      10. seq_len → net_input_dim == seq_len * frame_dim (FrameStacker)
    """
    if extra_state_dim is None:
        extra_state_dim = 2 * (num_robots - 1)
    frame_dim = state_dim + extra_state_dim
    net_input_dim = frame_dim * seq_len

    print("=" * 60)
    print(f"Dry-run: {num_robots} robots, state={state_dim}+{extra_state_dim}"
          f"={frame_dim}d, seq_len={seq_len}, net_input={net_input_dim}d")
    print(f"eps={num_episodes}, steps/ep={max_steps}, shared_buffer={shared_buffer}")
    print("=" * 60)

    # ------ Check 1: Model construction ------
    print("\n📦 Check 1: Model construction + warm-start")
    social_v1_dir = Path("src/drl_navigation_ros2/models/"
                         "SAC_multi_robot_scene1b_bottleneck_social_v1_0")
    load_name = "SAC_multi_robot_scene1b_bottleneck_social_v1_0"
    models = []
    try:
        for i in range(num_robots):
            model = SAC(
                state_dim=state_dim, action_dim=2, max_action=1,
                device=torch.device("cpu"),
                save_every=0, load_model=True,
                extra_state_dim=extra_state_dim, old_state_dim=state_dim,
                seq_len=seq_len,
                save_directory=Path(tempfile.mkdtemp()),
                model_name=f"dryrun_r{i}",
                load_directory=social_v1_dir, load_name=load_name,
                log_dir=None,
            )
            models.append(model)
        assert models[0].net_input_dim == net_input_dim
        print(f"   ✅ {len(models)} models loaded, net_input_dim={models[0].net_input_dim}")
    except Exception as e:
        print(f"   ❌ Model construction failed: {e}")
        return False

    # ------ Check 2: Buffer construction ------
    print("\n📦 Check 2: Buffer construction")
    try:
        if shared_buffer:
            buffers = [ReplayBuffer(buffer_size=1000, random_seed=4242)]
        else:
            buffers = [ReplayBuffer(buffer_size=1000, random_seed=42 + i)
                       for i in range(num_robots)]
        print(f"   ✅ {len(buffers)} buffer(s), capacity=1000")
    except Exception as e:
        print(f"   ❌ Buffer construction failed: {e}")
        return False

    # ------ Check 3: Environment interface ------
    print("\n📦 Check 3: Environment interface (mock Gazebo)")
    env = MockGazeboEnv(num_robots=num_robots)
    try:
        obs_list = env.reset()
        assert len(obs_list) == num_robots, f"reset returned {len(obs_list)} obs"
        for i, obs in enumerate(obs_list):
            assert len(obs) == 9, f"obs[{i}] has {len(obs)} elements, expected 9"
            scan, dist, cos, sin, col, goal, a, reward, nf = obs
            assert scan.shape == (180,), f"scan shape {scan.shape}"
            assert len(nf) == extra_state_dim, f"nf len {len(nf)}, expected {extra_state_dim}"
        print(f"   ✅ reset() returns {num_robots}x 9-element obs, nf={extra_state_dim}d")

        cmds = [(0.5, 0.0)] * num_robots
        next_obs = env.step(cmds, active_mask=[True] * num_robots)
        assert len(next_obs) == num_robots
        print("   ✅ step() returns correct format")
    except Exception as e:
        print(f"   ❌ Environment interface failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # ------ Check 4: prepare_state + stacking + get_action ------
    print("\n📦 Check 4: prepare_state (frame) → stack → get_action")
    try:
        for i in range(num_robots):
            scan, dist, cos, sin, col, goal, a, _, nf = obs_list[i]
            state, terminal = models[i].prepare_state(scan, dist, cos, sin, col, goal, a, nf)
            assert len(state) == frame_dim, f"prepare_state → {len(state)} != {frame_dim}"
            fs = FrameStacker(frame_dim, seq_len)
            for _ in range(seq_len):
                stacked = fs.push(state)
            assert stacked is not None and len(stacked) == net_input_dim

            action_det = models[i].get_action(stacked, add_noise=False)
            action_noisy = models[i].get_action(stacked, add_noise=True)
            assert len(action_det) == 2 and len(action_noisy) == 2
        print(f"   ✅ prepare_state→{frame_dim}d, stack→{net_input_dim}d, get_action→2d")
    except Exception as e:
        print(f"   ❌ prepare_state/get_action failed: {e}")
        return False

    # ------ Checks 5-6: Full training loop (with frame stacking) ------
    print(f"\n📦 Check 5-6: Training loop ({num_episodes} episodes)")
    global_ep = 0
    total_goals = 0
    total_transitions = 0

    def _get_buffer():
        return buffers[0] if shared_buffer else None

    try:
        for ep in range(num_episodes):
            obs_list = env.reset()
            active = [True] * num_robots
            # Frame stackers (seq_len=1 → identity)
            stackers = [FrameStacker(frame_dim, seq_len) for _ in range(num_robots)]
            for i in range(num_robots):
                frame0, _ = models[i].prepare_state(*obs_list[i][:7],
                                                    (obs_list[i][8] if extra_state_dim > 0 else None))
                for _ in range(seq_len):
                    stackers[i].push(frame0)
            gs = 0
            ep_goals = 0

            while gs < max_steps and any(active):
                states = [None] * num_robots
                cmds = []
                for i in range(num_robots):
                    if active[i]:
                        frame, _ = models[i].prepare_state(
                            *obs_list[i][:7],
                            (obs_list[i][8] if extra_state_dim > 0 else None))
                        states[i] = stackers[i].push(frame)
                        action = models[i].get_action(states[i], add_noise=True)
                        linear = (float(action[0]) + 1.0) / 2.0
                        angular = float(action[1])
                        cmds.append((linear, angular))
                    else:
                        cmds.append((0.0, 0.0))

                next_obs = env.step(cmds, active_mask=active)

                for i in range(num_robots):
                    if not active[i]:
                        continue
                    frame_next, terminal = models[i].prepare_state(
                        *next_obs[i][:7],
                        (next_obs[i][8] if extra_state_dim > 0 else None))
                    next_state = stackers[i].push(frame_next)
                    if terminal:
                        active[i] = False
                        if next_obs[i][5]:
                            ep_goals += 1

                    if states[i] is not None:
                        reward = next_obs[i][7]
                        buf = _get_buffer() or buffers[i]
                        buf.add(states[i], cmds[i], reward, terminal, next_state)
                        total_transitions += 1

                obs_list = next_obs
                gs += 1

            total_goals += ep_goals
            global_ep += 1

            if global_ep % train_every == 0:
                for i in range(num_robots):
                    buf = _get_buffer() or buffers[i]
                    if len(buf) >= batch_size:
                        models[i].train(buf, training_iterations, batch_size)

        print(f"   ✅ {num_episodes} episodes complete")
        print(f"   ✅ {total_transitions} transitions stored")
        print(f"   ✅ goals={total_goals}")
        print(f"   ✅ buffer sizes: {[len(b) for b in buffers]}")

        if total_transitions == 0:
            print("   ⚠️  No transitions stored — episodes too short?")
            return False

    except Exception as e:
        print(f"   ❌ Training loop failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # ------ Check 7: Model save ------
    print("\n📦 Check 7: Model save")
    tmpdir = tempfile.mkdtemp()
    try:
        for i in range(num_robots):
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            models[i].save(f"dryrun_final_r{i}", tmpdir)
            assert (Path(tmpdir) / f"dryrun_final_r{i}_actor.pth").exists()
        print(f"   ✅ All models saved to {tmpdir}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        for m in models:
            shutil.rmtree(m.save_directory, ignore_errors=True)

    # ------ Check 8: reset_mode switching (headless real env) ------
    print("\n📦 Check 8: reset_mode switching (headless env, no Gazebo)")
    try:
        from multi_robot_env import MultiRobotEnv
        env2 = MultiRobotEnv(num_robots=num_robots, scene="scene1b", headless=True)
        env2.set_reset_mode("train")
        starts, targets, _ = env2._sample_episode_layout()
        assert starts is not None and len(starts) == num_robots, "train layout invalid"
        env2.set_reset_mode("scene1b_nr_queue")
        starts2, targets2, _ = env2._sample_episode_layout()
        assert starts2 is not None and len(starts2) == num_robots, "eval layout invalid"
        print(f"   ✅ train + scene1b_nr_queue → valid {num_robots}-robot layouts")
    except Exception as e:
        print(f"   ❌ reset_mode check failed: {e}")
        return False

    # ------ Check 9: prepare_state embeds neighbor features ------
    print("\n📦 Check 9: prepare_state embeds neighbor features")
    try:
        if extra_state_dim > 0:
            nf_known = [0.1 * (i + 1) for i in range(extra_state_dim)]
            scan = np.full(180, 2.0, dtype=np.float32)
            state, _ = models[0].prepare_state(scan, 3.0, 0.9, 0.4, False, False,
                                               [0.0, 0.0], nf_known)
            assert len(state) == frame_dim
            assert np.allclose(state[-extra_state_dim:], nf_known), "nf not at tail"
        else:
            print("   (extra_state_dim=0, skipping)")
        print(f"   ✅ prepare_state embeds {extra_state_dim}d nf at tail")
    except Exception as e:
        print(f"   ❌ nf embed check failed: {e}")
        return False

    # ------ Check 10: seq_len → net_input_dim + FrameStacker ------
    print("\n📦 Check 10: seq_len frame-stacking shape")
    try:
        assert models[0].net_input_dim == net_input_dim
        fs = FrameStacker(frame_dim, seq_len)
        stacked = None
        for _ in range(seq_len):
            stacked = fs.push(np.zeros(frame_dim, dtype=np.float32))
        assert stacked is not None and stacked.shape[0] == net_input_dim
        print(f"   ✅ net_input_dim={net_input_dim} == seq_len({seq_len}) × frame_dim({frame_dim})")
    except Exception as e:
        print(f"   ❌ seq_len check failed: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry-run training verification")
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=25)
    parser.add_argument("--extra-state-dim", type=int, default=None,
                        help="Default: 2*(num_robots-1)")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--shared-buffer", action="store_true", default=True)
    args = parser.parse_args()

    ok = dryrun(num_robots=args.num_robots, state_dim=args.state_dim,
                extra_state_dim=args.extra_state_dim, seq_len=args.seq_len,
                shared_buffer=args.shared_buffer)
    print("\n" + "=" * 60)
    if ok:
        print("✅ DRY-RUN PASSED — training script is ready to launch")
    else:
        print("❌ DRY-RUN FAILED — fix issues above before launching training")
    print("=" * 60)
    sys.exit(0 if ok else 1)
