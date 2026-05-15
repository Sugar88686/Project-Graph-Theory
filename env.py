# env.py
import networkx as nx
import copy


class NetworkDismantlingEnv:
    """
    环境：给定一个初始图，智能体逐步删除节点。
    状态：当前图（由外部编码器处理）
    动作：当前 GCC 中的节点
    奖励：基于论文公式(1)的score变化
    """

    def __init__(self, graph: nx.Graph):
        self.original_graph = graph.copy()
        self.reset()

    def reset(self):
        """重置环境到初始状态"""
        self.current_graph = self.original_graph.copy()
        self.prev_score = self.compute_score(self.current_graph)
        return self._get_state()

    def _get_state(self):
        """返回当前图（用于编码器）"""
        return self.current_graph

    def get_gcc(self):
        """返回当前图的最大连通分量子图"""
        if self.current_graph.number_of_nodes() == 0:
            return nx.Graph()
        components = list(nx.connected_components(self.current_graph))
        largest = max(components, key=len)
        return self.current_graph.subgraph(largest).copy()

    def compute_score(self, graph):
        """计算公式(1)中的score"""
        gcc = self.get_gcc() if graph is self.current_graph else self._get_gcc_of(graph)
        n_total = graph.number_of_nodes()
        if n_total == 0 or gcc.number_of_nodes() == 0:
            return 0.0
        gcc_size = gcc.number_of_nodes()
        max_deg = max(dict(gcc.degree).values()) if gcc.nodes else 0
        score = (gcc_size / n_total) * (max_deg / n_total)
        return score

    def _get_gcc_of(self, graph):
        components = list(nx.connected_components(graph))
        if not components:
            return nx.Graph()
        largest = max(components, key=len)
        return graph.subgraph(largest).copy()

    def step(self, node):
        """执行删除节点动作"""
        if node not in self.current_graph.nodes:
            raise ValueError(f"Node {node} not in graph")
        self.current_graph.remove_node(node)
        new_score = self.compute_score(self.current_graph)
        # 奖励 = -(score变化)  因为RL最大化奖励，我们想最小化score
        reward = -(new_score - self.prev_score)
        self.prev_score = new_score
        done = (self.get_gcc().number_of_nodes() == 0)
        return self._get_state(), reward, done

    def get_valid_actions(self):
        """返回当前可动作的节点列表（GCC中的节点）"""
        gcc = self.get_gcc()
        return list(gcc.nodes)