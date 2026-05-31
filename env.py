import networkx as nx
from models import get_graph_data_from_nx

class NetworkDismantlingEnv:
    def __init__(self, graph: nx.Graph):
        self.original_graph = graph.copy()
        self.reset()

    def reset(self):
        self.current_graph = self.original_graph.copy()
        # 【提速优化】：在环境重置时直接生成 PyG 数据
        self.current_pyg = get_graph_data_from_nx(self.current_graph)
        return self.current_graph.copy(), self.current_pyg.clone()

    def get_gcc(self):
        if self.current_graph.number_of_nodes() == 0:
            return nx.Graph()
        components = list(nx.connected_components(self.current_graph))
        if not components:
            return nx.Graph()
        largest = max(components, key=len)
        return self.current_graph.subgraph(largest).copy()

    def step(self, node):
        if node not in self.current_graph.nodes:
            raise ValueError(f"Node {node} not in graph")

        self.current_graph.remove_node(node)

        # 1. 获取当前 GCC
        gcc = self.get_gcc()
        gcc_size = gcc.number_of_nodes()
        n_total = self.original_graph.number_of_nodes()

        # 2. 计算 GCC 中的最大度数 (论文的核心创新点 Dual Metric)
        if gcc_size > 0:
            max_deg = max(dict(gcc.degree()).values())
        else:
            max_deg = 0

        # 3. 计算论文 Equation (1) 中的 Score
        if n_total > 0:
            score = (gcc_size / n_total) * (max_deg / n_total)
        else:
            score = 0.0

        # ==========================================
        # 🌟 终极奖励函数：密集惩罚形式
        # ==========================================
        reward = -score

        done = (gcc_size == 0)

        self.current_pyg = get_graph_data_from_nx(self.current_graph)
        return (self.current_graph.copy(), self.current_pyg.clone()), reward, done

    def get_valid_actions(self):
        gcc = self.get_gcc()
        return list(gcc.nodes)
