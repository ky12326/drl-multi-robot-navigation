"""Frame stacking for temporal context (GRU/RTP-Net ready).

Maintains a per-robot history of single-frame states and returns a flattened
[seq_len * D] vector once enough frames are buffered. seq_len=1 → identity
(current behavior). A recurrent actor (e.g. RTP-Net's GRU) can reshape the
flat [T*D] back to [T, D] inside its forward().
"""

from collections import deque

import numpy as np


class FrameStacker:
    """Buffers the last seq_len single-frame states and stacks them flat."""

    def __init__(self, state_dim, seq_len=1):
        self.state_dim = int(state_dim)
        self.seq_len = max(1, int(seq_len))
        self._buf = deque(maxlen=self.seq_len)
        self.reset()

    def reset(self):
        """Clear history (call at episode start)."""
        self._buf.clear()

    def push(self, state):
        """Push one frame; return flat [seq_len*state_dim] or None if not full yet."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"frame shape {state.shape[0]} != state_dim {self.state_dim}")
        self._buf.append(state)
        if len(self._buf) < self.seq_len:
            return None
        return np.concatenate(list(self._buf)).astype(np.float32)

    @property
    def stacked_dim(self):
        return self.seq_len * self.state_dim

    def __len__(self):
        return len(self._buf)
