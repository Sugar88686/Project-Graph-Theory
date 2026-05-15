# utils.py
import networkx as nx
import numpy as np
from tqdm import tqdm

def generate_ba_graphs(sizes, m=4, num_per_size=100):
    """生成 BA 图集合，sizes: list of (min_nodes, max_nodes) 或固定节点数"""
    graphs = []
    for size_range in sizes:
        if isinstance(size_range, tuple):
            low, high = size_range
            nodes_list = np.random.randint(low, high+1, num_per_size)
        else:
            nodes_list = [size_range] * nusm_per_size
        for n in nodes_list:
            G = nx.barabasi_albert_graph(n, m)
            graphs.append(G)
    return graphs

def generate_sbm_graphs(community_sizes, p_intra=0.1, p_inter=0.02, num_per_size=100):
    """生成 SBM 图，community_sizes: 每个社区的节点数列表"""
    graphs = []
    for _ in range(num_per_size):
        sizes = community_sizes.copy()
        probs = [[p_intra if i==j else p_inter for j in range(len(sizes))] for i in range(len(sizes))]
        G = nx.stochastic_block_model(sizes, probs, seed=None)[0]
        graphs.append(G)
    return graphs

def evaluate_agent(agent, test_graphs, env_class=NetworkDismantlingEnv):
    """评估智能体在测试图上的累积 GCC 大小和累积最大度数曲线下面积"""
    from env import NetworkDismantlingEnv
    results = []
    for G in tqdm(test_graphs, desc="Evaluating"):
        env = env_class(G.copy())
        state = env.reset()
        done = False
        gcc_sizes = []
        max_degs_in_gcc = []
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.act(env.current_graph, valid, eval_mode=True)
            _, _, done = env.step(action)
            gcc = env.get_gcc()
            gcc_sizes.append(gcc.number_of_nodes())
            max_degs_in_gcc.append(max(dict(gcc.degree).values()) if gcc.nodes else 0)
        # 计算累积指标（曲线下面积，归一化）
        total_nodes = G.number_of_nodes()
        acc_gcc = np.sum(gcc_sizes) / total_nodes
        acc_maxdeg = np.sum(max_degs_in_gcc) / total_nodes
        results.append((acc_gcc, acc_maxdeg))
    mean_gcc = np.mean([r[0] for r in results])
    mean_maxdeg = np.mean([r[1] for r in results])
    return mean_gcc, mean_maxdeg

# 基线算法实现（供对比）
class BaselineHDA:
    @staticmethod
    def dismantle(graph):
        """返回删除节点序列"""
        G = graph.copy()
        removed = []
        while True:
            gcc = max(nx.connected_components(G), key=len)
            gcc_sub = G.subgraph(gcc)
            if len(gcc_sub) == 0:
                break
            deg = dict(gcc_sub.degree)
            if not deg:
                break
            node = max(deg, key=deg.get)
            G.remove_node(node)
            removed.append(node)
        return removed

# 类似可添加 HBA, HPRA 等