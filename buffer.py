import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_nx, state_pyg, action, reward, next_state_nx, next_state_pyg, done):
        # 【提速优化】：直接存储 PyG 数据，避免训练时重复转换
        self.buffer.append((
            state_nx.copy(),
            state_pyg.clone(),
            action,
            reward,
            next_state_nx.copy(),
            next_state_pyg.clone(),
            done
        ))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
