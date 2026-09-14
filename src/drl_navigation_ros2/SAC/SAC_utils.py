import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import os
from collections import deque
import random
import math


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def _first_linear(sequential):
    """Return the first nn.Linear layer in a Sequential module."""
    for child in sequential.children():
        if isinstance(child, nn.Linear):
            return child
    raise ValueError("No nn.Linear layer found in Sequential")


def _first_input_layer(sequential):
    """Return (first_layer, kind) where kind in {'linear', 'recurrent', 'other'}."""
    for child in sequential.children():
        if isinstance(child, nn.Linear):
            return child, "linear"
        if isinstance(child, (nn.LSTM, nn.GRUCell, nn.GRU)):
            return child, "recurrent"
    return None, "other"


def pad_input_weights(trunk, old_input_dim, new_input_dim, old_state_dict, prefix,
                      suffix_cols=0):
    """Zero-pad / transfer first-layer weights when expanding input dimension.

    Dispatches on the first layer type so a new network architecture
    (e.g. RTP-Net with attention + GRU) never crashes the warm-start:

      linear    → existing prefix-copy / state+action-suffix logic (unchanged)
      recurrent → reserved: input-weight transfer not implemented yet → warn,
                  leave this layer random (cold start for the input projection)
      other     → warn, leave random (cold start); SAC.load's strict=False then
                  ignores the old MLP's keys, so the new architecture stays
                  randomly initialised — the safe "actor_only" fallback.

    For actor (suffix_cols=0): state-only, simple prefix copy.
      new[:, :old] ← old[:, :old],  new[:, old:] ← 0

    For critic (suffix_cols=action_dim): input = [state, action].
      old layout: [state(25), action(2)]  → 27 cols
      new layout: [state(29), action(2)]  → 31 cols
      Copy: new[:, :old_state] ← old[:, :old_state]        (state prefix)
            new[:, old_state:new_state] ← 0                  (extra state)
            new[:, new_state:] ← old[:, old_state:]          (action suffix)
    """
    first, kind = _first_input_layer(trunk)
    if kind != "linear":
        print(f"   ⚠️ [pad_input_weights] first layer is '{kind}' "
              f"(not nn.Linear) — input-weight transfer not applied; "
              f"this layer stays random (cold start).")
        return
    old_weight = old_state_dict[f"{prefix}.0.weight"]
    if old_weight.shape[1] != old_input_dim:
        raise ValueError(
            f"Expected old_input_dim={old_input_dim}, "
            f"got weight shape[1]={old_weight.shape[1]}"
        )
    if suffix_cols == 0:
        # Simple prefix expansion (actor)
        first.weight.data[:, :old_input_dim] = old_weight
        first.weight.data[:, old_input_dim:] = 0.0
    else:
        # Input has suffix columns that shift position (critic: state + action)
        old_prefix = old_input_dim - suffix_cols
        new_prefix = new_input_dim - suffix_cols
        # State prefix
        first.weight.data[:, :old_prefix] = old_weight[:, :old_prefix]
        # Extra state dims → zero
        first.weight.data[:, old_prefix:new_prefix] = 0.0
        # Action suffix (shifted)
        first.weight.data[:, new_prefix:] = old_weight[:, old_prefix:]
    old_bias_key = f"{prefix}.0.bias"
    if old_bias_key in old_state_dict:
        first.bias.data.copy_(old_state_dict[old_bias_key])


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
