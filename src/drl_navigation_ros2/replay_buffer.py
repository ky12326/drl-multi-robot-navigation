import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.priorities = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2, priority=1.0):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.priorities.append(float(priority))
            self.count += 1
        else:
            self.buffer.popleft()
            self.priorities.popleft()
            self.buffer.append(experience)
            self.priorities.append(float(priority))

    def __len__(self):
        return self.count

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            n = self.count
        else:
            n = batch_size

        probs = np.array(self.priorities, dtype=np.float64)
        probs /= probs.sum()
        indices = np.random.choice(self.count, size=n, replace=False, p=probs)
        batch = [self.buffer[i] for i in indices]

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()
        self.count = 0
