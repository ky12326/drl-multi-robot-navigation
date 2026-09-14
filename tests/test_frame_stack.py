"""Unit tests for SAC.frame_stack.FrameStacker (pure Python, no ROS)."""

import numpy as np

from SAC.frame_stack import FrameStacker


def test_seq_len_1_is_identity():
    fs = FrameStacker(4, 1)
    s = fs.push(np.array([1, 2, 3, 4], dtype=np.float32))
    assert s.tolist() == [1, 2, 3, 4]


def test_seq_len_3_rolling_stack():
    fs = FrameStacker(4, 3)
    # Not full yet → None
    assert fs.push(np.array([1, 0, 0, 0], dtype=np.float32)) is None
    assert fs.push(np.array([2, 0, 0, 0], dtype=np.float32)) is None
    # Full → flattened [f1, f2, f3]
    s = fs.push(np.array([3, 0, 0, 0], dtype=np.float32))
    assert s.tolist() == [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
    # Rolling: [f2, f3, f4]
    s2 = fs.push(np.array([4, 0, 0, 0], dtype=np.float32))
    assert s2.tolist() == [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]


def test_stacked_dim():
    fs = FrameStacker(25, 5)
    assert fs.stacked_dim == 125


def test_reset_clears_history():
    fs = FrameStacker(4, 3)
    fs.push(np.array([1, 0, 0, 0], dtype=np.float32))
    fs.reset()
    assert len(fs) == 0
