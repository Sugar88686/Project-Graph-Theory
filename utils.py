import networkx as nx
import numpy as np
from tqdm import tqdm
import torch
import concurrent.futures


def generate_ba_graphs(sizes, m=4, num_per_size=100):
    graphs = []
    for size_range in sizes:
        if isinstance(size_range, tuple):
            low, high = size_range
            nodes_list = np.random.randint(low, high + 1, num_per_size)
        else:
            nodes_list = [size_range] * num_per_size
        for n in nodes_list:
            G = nx.barabasi_albert_graph(n, m)
            graphs.append(G)
    return graphs


def generate_sbm_graphs(community_sizes, p_intra=0.1, p_inter=0.02, num_per_size=100):
    graphs = []
    for _ in range(num_per_size):
        sizes = community_sizes.copy()
        probs = [[p_intra if i == j else p_inter for j in range(len(sizes))] for i in range(len(sizes))]
        G = nx.stochastic_block_model(sizes, probs, seed=None)
        graphs.append(G)
    return graphs


# ==========================================
# 🌟 多进程 Worker 函数
# ==========================================
def _eval_worker(args):
    G, cfg, encoder_state, decoder_state = args

    # 局部导入，避免多进程序列化问题
    from agent import MaxShotAgent
    from env import NetworkDismantlingEnv

    # 强制在 CPU 上运行评估，避免 CUDA 多进程冲突
    device = torch.device('cpu')
    agent = MaxShotAgent(cfg, device)
    agent.encoder.load_state_dict(encoder_state)
    agent.decoder.load_state_dict(decoder_state)
    agent.encoder.eval()
    agent.decoder.eval()

    env = NetworkDismantlingEnv(G.copy())
    state_nx, state_pyg = env.reset()
    done = False
    gcc_sizes = []
    max_degs_in_gcc = []

    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        action = agent.act(env.current_graph, env.current_pyg, valid, eval_mode=True)
        _, _, done = env.step(action)
        gcc = env.get_gcc()
        gcc_sizes.append(gcc.number_of_nodes())
        max_degs_in_gcc.append(max(dict(gcc.degree).values()) if gcc.nodes else 0)

    total_nodes = G.number_of_nodes()
    acc_gcc = np.sum(gcc_sizes) / total_nodes
    acc_maxdeg = np.sum(max_degs_in_gcc) / total_nodes
    return acc_gcc, acc_maxdeg


# ==========================================
# 🌟 多进程评估主函数
# ==========================================
def evaluate_agent(agent, test_graphs, cfg):
    # 1. 将当前 GPU 模型的权重拷贝到 CPU 内存中
    encoder_state = {k: v.cpu() for k, v in agent.encoder.state_dict().items()}
    decoder_state = {k: v.cpu() for k, v in agent.decoder.state_dict().items()}

    # 2. 准备多进程参数
    args_list = [(G, cfg, encoder_state, decoder_state) for G in test_graphs]
    results = []

    # 3. 启动进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.eval_num_workers) as executor:
        # 使用 tqdm 显示多进程进度条
        for res in tqdm(executor.map(_eval_worker, args_list), total=len(test_graphs),
                        desc="Evaluating (Multi-Process)"):
            results.append(res)

    mean_gcc = np.mean([r[0] for r in results])
    mean_maxdeg = np.mean([r[1] for r in results])
    return mean_gcc, mean_maxdeg


class BaselineHDA:
    @staticmethod
    def dismantle(graph):
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
