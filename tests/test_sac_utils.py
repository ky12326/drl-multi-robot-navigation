"""Unit tests for SAC_utils.pad_input_weights layer-type dispatch (pure torch)."""

import torch
from torch import nn

from SAC.SAC_utils import pad_input_weights


def _mlp_trunk(in_dim, out_dim=4):
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))


def test_pad_linear_prefix_copy():
    old = _mlp_trunk(25)
    new = _mlp_trunk(29)
    old_sd = {"trunk.0.weight": old[0].weight.detach().clone(),
              "trunk.0.bias": old[0].bias.detach().clone()}
    pad_input_weights(new, 25, 29, old_sd, "trunk")
    assert torch.allclose(new[0].weight[:, :25], old[0].weight)
    assert torch.allclose(new[0].weight[:, 25:], torch.zeros_like(new[0].weight[:, 25:]))
    assert torch.allclose(new[0].bias, old[0].bias)


def test_pad_critic_suffix_shift():
    # critic input = [state(25), action(2)] = 27; new = [state(29), action(2)] = 31
    old = _mlp_trunk(27)
    new = _mlp_trunk(31)
    old_sd = {"Q1.0.weight": old[0].weight.detach().clone(),
              "Q1.0.bias": old[0].bias.detach().clone()}
    pad_input_weights(new, 27, 31, old_sd, "Q1", suffix_cols=2)
    # state prefix (cols 0..24) copied; extra state (25..28) zero; action suffix (29..30) copied
    assert torch.allclose(new[0].weight[:, :25], old[0].weight[:, :25])
    assert torch.allclose(new[0].weight[:, 25:29], torch.zeros_like(new[0].weight[:, 25:29]))
    assert torch.allclose(new[0].weight[:, 29:31], old[0].weight[:, 25:27])
    assert torch.allclose(new[0].bias, old[0].bias)


def test_pad_recurrent_warns_but_does_not_crash():
    # LSTM first layer → pad_input_weights must warn + return, not raise
    trunk = nn.Sequential(nn.LSTM(input_size=25, hidden_size=64, batch_first=True))
    old_sd = {"trunk.0.weight": torch.randn(64, 25)}
    # Should not raise (warm-start falls back to cold start for this layer)
    pad_input_weights(trunk, 25, 33, old_sd, "trunk")


def test_pad_other_warns_but_does_not_crash():
    # Non-Linear, non-recurrent first layer (e.g. a custom encoder) → cold start
    trunk = nn.Sequential(nn.Conv1d(1, 8, 3))
    old_sd = {"trunk.0.weight": torch.randn(8, 1, 3)}
    pad_input_weights(trunk, 25, 33, old_sd, "trunk")
