from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
import SAC.SAC_utils as utils
from SAC.SAC_critic import DoubleQCritic as critic_model
from SAC.SAC_actor import DiagGaussianActor as actor_model
from torch.utils.tensorboard import SummaryWriter


class SAC(object):
    """SAC algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        learnable_temperature=True,
        save_every=0,
        load_model=False,
        log_dist_and_hist=False,
        extra_state_dim=0,
        save_directory=Path("src/drl_navigation_ros2/models/SAC"),
        model_name="SAC",
        load_directory=Path("src/drl_navigation_ros2/models/SAC"),
        load_name=None,
        old_state_dim=None,
        actor_only=False,
        log_dir=None,
        # ---- network injection (defaults = current MLP classes, no behavior change) ----
        actor_cls=None,
        critic_cls=None,
        hidden_dim=1024,
        hidden_depth=2,
        log_std_bounds=(-5, 2),
        # ---- frame stacking (seq_len>1 → network input = seq_len × frame_dim) ----
        seq_len=1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        actual_state_dim = state_dim + extra_state_dim
        self.seq_len = max(1, int(seq_len))
        self.actual_state_dim = actual_state_dim            # single-frame dim (prepare_state output)
        self.net_input_dim = actual_state_dim * self.seq_len  # network input dim (stacked)
        self.extra_state_dim = extra_state_dim
        self.action_range = (-max_action, max_action)
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.log_dist_and_hist = log_dist_and_hist

        self.train_metrics_dict = { "train_critic/loss_av": [],
                                    "train_actor/loss_av": [],
                                    "train_actor/target_entropy_av": [],
                                    "train_actor/entropy_av": [],
                                    "train_alpha/loss_av": [],
                                    "train_alpha/value_av": [],
                                    "train/batch_reward_av": []
        }

        # Injectable network classes — default to the current MLP classes so
        # existing experiments are bit-for-bit unchanged. A custom actor (e.g.
        # RTP-Net with attention + GRU) is plugged in via actor_cls=... .
        critic_model = critic_cls or globals()["critic_model"]
        actor_model = actor_cls or globals()["actor_model"]

        self.critic = critic_model(
            obs_dim=self.net_input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.critic_target = critic_model(
            obs_dim=self.net_input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = actor_model(
            obs_dim=self.net_input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            log_std_bounds=list(log_std_bounds),
        ).to(self.device)

        if load_model:
            self.load(filename=load_name or model_name, directory=load_directory,
                      old_state_dim=old_state_dim, actor_only=actor_only)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.critic_target.train()

        self.actor.train(True)
        self.critic.train(True)
        self.step = 0
        # 如果指定了log_dir，使用它；否则使用默认的runs目录
        if log_dir:
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = SummaryWriter()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory, old_state_dim=None, actor_only=False):
        actor_sd = torch.load("%s/%s_actor.pth" % (directory, filename),
                              map_location=self.device)

        # --- Actor loading ---
        if old_state_dim is not None and old_state_dim < self.net_input_dim:
            utils.pad_input_weights(
                self.actor.trunk, old_state_dim, self.net_input_dim,
                actor_sd, "trunk",
            )
            for k in list(actor_sd.keys()):
                if k.startswith("trunk.0."):
                    del actor_sd[k]
            self.actor.load_state_dict(actor_sd, strict=False)
        else:
            self.actor.load_state_dict(actor_sd)

        if actor_only:
            # Critic stays randomly initialised; sync target
            self.critic_target.load_state_dict(self.critic.state_dict())
            print(f"   🎯 Actor warm-start, critic random init")
        else:
            critic_sd = torch.load("%s/%s_critic.pth" % (directory, filename),
                                   map_location=self.device)
            critic_target_sd = torch.load("%s/%s_critic_target.pth" % (directory, filename),
                                          map_location=self.device)

            if old_state_dim is not None and old_state_dim < self.net_input_dim:
                old_critic_dim = old_state_dim + self.action_dim
                new_critic_dim = self.net_input_dim + self.action_dim
                for q_name, sd in [("Q1", critic_sd), ("Q2", critic_sd),
                                   ("Q1", critic_target_sd), ("Q2", critic_target_sd)]:
                    trunk = getattr(self.critic if sd is critic_sd else self.critic_target,
                                    q_name)
                    utils.pad_input_weights(
                        trunk, old_critic_dim, new_critic_dim, sd, q_name,
                        suffix_cols=self.action_dim,
                    )
                    for k in list(sd.keys()):
                        if k.startswith(f"{q_name}.0."):
                            del sd[k]
                self.critic.load_state_dict(critic_sd, strict=False)
                self.critic_target.load_state_dict(critic_target_sd, strict=False)
            else:
                self.critic.load_state_dict(critic_sd)
                self.critic_target.load_state_dict(critic_target_sd)

        print(f"Loaded weights from: {directory}")

    def train(self, replay_buffer, iterations, batch_size):
        for _ in range(iterations):
            self.update(
                replay_buffer=replay_buffer, step=self.step, batch_size=batch_size
            )

        for key, value in self.train_metrics_dict.items():
            if len(value):
                self.writer.add_scalar(key, mean(value), self.step)
            self.train_metrics_dict[key] = []
        self.step += 1

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs, add_noise):
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(self.action_range[0], self.action_range[1])
        else:
            return self.act(obs)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, done, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + ((1 - done) * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.train_metrics_dict["train_critic/loss_av"].append(critic_loss.item())
        self.writer.add_scalar("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.log_dist_and_hist:
            self.critic.log(self.writer, step)

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # Auxiliary-loss extension point (e.g. RTP-Net's intent-prediction head).
        # Default actor has no aux_loss() → behavior unchanged.
        if hasattr(self.actor, "aux_loss"):
            actor_loss = actor_loss + self.actor.aux_loss()

        self.train_metrics_dict["train_actor/loss_av"].append(actor_loss.item())
        self.train_metrics_dict["train_actor/target_entropy_av"].append(self.target_entropy)
        self.train_metrics_dict["train_actor/entropy_av"].append(-log_prob.mean().item())
        self.writer.add_scalar("train_actor/loss", actor_loss, step)
        self.writer.add_scalar("train_actor/target_entropy", self.target_entropy, step)
        self.writer.add_scalar("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.log_dist_and_hist:
            self.actor.log(self.writer, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            self.train_metrics_dict["train_alpha/loss_av"].append(alpha_loss.item())
            self.train_metrics_dict["train_alpha/value_av"].append(self.alpha.item())
            self.writer.add_scalar("train_alpha/loss", alpha_loss, step)
            self.writer.add_scalar("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step, batch_size):
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_states,
        ) = replay_buffer.sample_batch(batch_size)

        state = torch.Tensor(batch_states).to(self.device)
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device)
        done = torch.Tensor(batch_dones).to(self.device)
        self.train_metrics_dict["train/batch_reward_av"].append(batch_rewards.mean().item())
        self.writer.add_scalar("train/batch_reward", batch_rewards.mean(), step)

        self.update_critic(state, action, reward, next_state, done, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action,
                      neighbor_features=None):
        # update the returned data from ROS into a form used for learning in the current model
        latest_scan = np.array(latest_scan, dtype=np.float32)

        if latest_scan.size == 0:
            latest_scan = np.full(180, 7.0, dtype=np.float32)

        latest_scan = np.nan_to_num(
            latest_scan,
            nan=7.0,
            posinf=7.0,
            neginf=0.0,
        )

        base_dim = 25  # 20 LiDAR bins + dist + cos + sin + 2 prev actions
        max_bins = base_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin))
        state = min_values + [distance, cos, sin] + [action[0], action[1]]

        if neighbor_features is not None:
            if len(neighbor_features) != self.extra_state_dim:
                raise ValueError(
                    f"neighbor_features length {len(neighbor_features)} != "
                    f"extra_state_dim {self.extra_state_dim}"
                )
            state += list(neighbor_features)
        elif self.extra_state_dim > 0:
            state += [0.0] * self.extra_state_dim

        assert len(state) == self.actual_state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal
