import networkx as nx
import numpy as np

class GraphDismantleEnv:
    """
    状态：节点特征（归一化度）+ 全局特征（当前GCC/初始GCC）
    动作：移除节点（整数索引）
    奖励：gcc下降率 + 命中最大度节点奖励
    """
    def __init__(self, graph, max_steps_ratio=0.5):
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.num_nodes = graph.number_of_nodes()
        self.max_steps = int(self.num_nodes * max_steps_ratio)
        self.steps = 0
        self.removed_set = set()
        self.initial_gcc_size = self._get_gcc_size()
        self.reset()

    def reset(self):
        self.graph = self.original_graph.copy()
        self.steps = 0
        self.removed_set = set()
        return self._get_state()

    def _get_gcc_size(self):
        if self.graph.number_of_nodes() == 0:
            return 0
        largest_cc = max(nx.connected_components(self.graph), key=len)
        return len(largest_cc)

    def _get_max_degree_node(self):
        if self.graph.number_of_nodes() == 0:
            return None
        degrees = dict(self.graph.degree())
        max_deg = max(degrees.values())
        candidates = [n for n, d in degrees.items() if d == max_deg]
        return np.random.choice(candidates)

    def _get_state(self):
        # 节点特征: 归一化的度 (1维)
        degrees = np.zeros(self.num_nodes)
        for n in self.graph.nodes:
            degrees[n] = self.graph.degree(n)
        max_deg = max(degrees.max(), 1)
        node_feat = (degrees / max_deg).reshape(-1, 1)   # (N,1)
        # 全局特征: 当前GCC占比
        gcc_ratio = self._get_gcc_size() / max(1, self.initial_gcc_size)
        global_feat = np.array([gcc_ratio])
        return node_feat, global_feat

    def step(self, action):
        if action in self.removed_set or action >= self.num_nodes:
            # 无效动作：负奖励，不改变环境
            return self._get_state(), -0.5, False, {'valid': False}

        prev_gcc = self._get_gcc_size()
        prev_max_node = self._get_max_degree_node()

        self.graph.remove_node(action)
        self.removed_set.add(action)
        self.steps += 1

        curr_gcc = self._get_gcc_size()
        gcc_reward = (prev_gcc - curr_gcc) / max(1, prev_gcc)   # GCC下降比例
        hit_bonus = 0.3 if action == prev_max_node else 0.0
        reward = gcc_reward + hit_bonus

        done = (curr_gcc < 0.1 * self.initial_gcc_size) or (self.steps >= self.max_steps)

        next_state = self._get_state()
        info = {'valid': True, 'gcc_ratio': curr_gcc / self.initial_gcc_size}
        return next_state, reward, done, info