# buffer.py
import random
from collections import deque
import networkx as nx
import copy


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_graph, action, reward, next_state_graph, done):
        # 为了节省内存，存储图的快照（边列表和节点特征简表，这里简单存整个图）
        # 注意：大型图会占用大量内存，可优化为只存删除节点序号 + 原图索引，此处简化。
        self.buffer.append((
            copy.deepcopy(state_graph),
            action,
            reward,
            copy.deepcopy(next_state_graph),
            done
        ))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)