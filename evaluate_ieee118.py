import os
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 导入电力系统标准库以获取 IEEE 118 真实拓扑
try:
    import pandapower.networks as pn
except ImportError:
    print("[!] 缺少 pandapower 库。请运行: pip install pandapower")
    exit()

from config import Config
from env import NetworkDismantlingEnv
from agent import MaxShotAgent


# ==========================================
# 1. 定义论文中的 Baseline 算法 (自适应版本)
# ==========================================

def get_hda_sequence(graph):
    """HDA (High-Degree Algorithm): 每次重新计算度数并移除最大度节点"""
    G = graph.copy()
    sequence = []
    while G.number_of_nodes() > 0:
        deg = dict(G.degree())
        best_node = max(deg, key=deg.get)
        sequence.append(best_node)
        G.remove_node(best_node)
    return sequence


def get_hba_sequence(graph):
    """HBA (High-Betweenness Algorithm): 每次重新计算介数中心性并移除"""
    G = graph.copy()
    sequence = []
    while G.number_of_nodes() > 0:
        bet = nx.betweenness_centrality(G)
        best_node = max(bet, key=bet.get)
        sequence.append(best_node)
        G.remove_node(best_node)
    return sequence


def get_hca_sequence(graph):
    """HCA (High-Closeness Algorithm): 每次重新计算接近中心性并移除"""
    G = graph.copy()
    sequence = []
    while G.number_of_nodes() > 0:
        clo = nx.closeness_centrality(G)
        best_node = max(clo, key=clo.get)
        sequence.append(best_node)
        G.remove_node(best_node)
    return sequence


def get_hpra_sequence(graph):
    """HPRA (High PageRank Removal Algorithm): 每次重新计算 PageRank 并移除"""
    G = graph.copy()
    sequence = []
    while G.number_of_nodes() > 0:
        try:
            pr = nx.pagerank(G)
        except:
            pr = {n: 1.0 for n in G.nodes()}
        best_node = max(pr, key=pr.get)
        sequence.append(best_node)
        G.remove_node(best_node)
    return sequence


def get_ai_sequence(agent, graph, env):
    """MaxShot (Ours): 使用训练好的强化学习模型"""
    state_nx, state_pyg = env.reset()
    sequence = []
    done = False

    while not done:
        valid_nodes = env.get_valid_actions()
        if not valid_nodes:
            break
        # eval_mode=True 确保模型不进行随机探索，只输出最优解
        action = agent.act(state_nx, state_pyg, valid_nodes, eval_mode=True)
        if action is None:
            break
        sequence.append(action)
        (state_nx, state_pyg), _, done = env.step(action)

    return sequence


# ==========================================
# 2. 评估序列并计算双重指标 (Dual Metric)
# ==========================================

def evaluate_sequence(graph, sequence):
    """返回每一步拆解后的 GCC 相对大小 和 GCC 中的最大度数相对大小"""
    G = graph.copy()
    N = G.number_of_nodes()

    gcc_sizes = []
    gcc_max_degs = []

    def get_metrics(current_G):
        components = list(nx.connected_components(current_G))
        if not components:
            return 0.0, 0.0
        gcc_nodes = max(components, key=len)
        gcc_sub = current_G.subgraph(gcc_nodes)

        gcc_size = len(gcc_nodes) / N
        if len(gcc_sub.edges) > 0:
            max_deg = max(dict(gcc_sub.degree).values()) / N
        else:
            max_deg = 0.0
        return gcc_size, max_deg

    # 记录初始状态
    s, d = get_metrics(G)
    gcc_sizes.append(s)
    gcc_max_degs.append(d)

    # 逐步拆解并记录
    for node in sequence:
        if node in G:
            G.remove_node(node)
        s, d = get_metrics(G)
        gcc_sizes.append(s)
        gcc_max_degs.append(d)

    # 补齐长度到 N+1 (防止网络提前完全碎裂)
    while len(gcc_sizes) <= N:
        gcc_sizes.append(0.0)
        gcc_max_degs.append(0.0)

    return np.array(gcc_sizes), np.array(gcc_max_degs)


# ==========================================
# 3. 主函数
# ==========================================

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 评估使用设备: {device}")

    # 1. 加载模型 (已包含 weights_only=False 修复)
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"[!] 找不到模型文件 {model_path}，请先完成训练！")
        return

    agent = MaxShotAgent(cfg, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    agent.encoder.load_state_dict(checkpoint['encoder'])
    agent.decoder.load_state_dict(checkpoint['decoder'])
    agent.encoder.eval()
    agent.decoder.eval()
    print(f"[*] 成功加载模型 (来自 Episode {checkpoint.get('episode', 'Unknown')})")

    # 2. 获取 IEEE 118 节点电网图
    print("\n[*] 正在加载 IEEE 118-bus 电网真实数据...")
    net = pn.case118()
    G = nx.Graph()
    # 添加边 (pandapower 的 line 数据包含 from_bus 和 to_bus)
    edges = list(zip(net.line.from_bus, net.line.to_bus))
    G.add_edges_from(edges)

    # 确保节点编号是连续的整数 (RL 环境需要)
    G = nx.convert_node_labels_to_integers(G)
    print(f"[*] IEEE 118 图加载完成: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")

    # 3. 准备记录结果的字典
    methods = ['HDA', 'HBA', 'HCA', 'HPRA', 'MaxShot (Ours)']
    results_size = {}
    results_deg = {}

    # 4. 开始评估
    print("\n[*] 开始在 IEEE 118 上运行各算法 (中心性算法可能需要十几秒，请稍候)...")
    env = NetworkDismantlingEnv(G.copy())

    seqs = {
        'HDA': get_hda_sequence(G),
        'HBA': get_hba_sequence(G),
        'HCA': get_hca_sequence(G),
        'HPRA': get_hpra_sequence(G),
        'MaxShot (Ours)': get_ai_sequence(agent, G, env)
    }

    for method, seq in seqs.items():
        print(f"    - 正在评估 {method} 的拆解序列...")
        sizes, degs = evaluate_sequence(G, seq)
        results_size[method] = sizes
        results_deg[method] = degs

    # 5. 计算 AUC (曲线下面积)
    print("\n" + "=" * 55)
    print("📊 IEEE 118 评估结果 (AUC 越小越好):")
    print(f"{'Method':<18} | {'GCC Size AUC':<15} | {'Max Degree AUC':<15}")
    print("-" * 55)

    for method in methods:
        auc_size = np.sum(results_size[method]) / len(results_size[method])
        auc_deg = np.sum(results_deg[method]) / len(results_deg[method])

        marker = "🚀" if method == 'MaxShot (Ours)' else "  "
        print(f"{marker} {method:<15} | {auc_size:<15.4f} | {auc_deg:<15.4f}")
    print("=" * 55)

    # 6. 绘制对比图 (双子图：完全对齐论文的 Table 2 和 Table 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x_axis = np.linspace(0, 1, G.number_of_nodes() + 1)

    styles = {
        'HDA': {'color': 'orange', 'linestyle': '-'},
        'HBA': {'color': 'blue', 'linestyle': '--'},
        'HCA': {'color': 'green', 'linestyle': '-.'},
        'HPRA': {'color': 'purple', 'linestyle': ':'},
        'MaxShot (Ours)': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5}
    }

    for method in methods:
        # 左图：GCC Size
        ax1.plot(x_axis, results_size[method], label=method,
                 color=styles[method]['color'], linestyle=styles[method]['linestyle'],
                 linewidth=styles.get(method, {}).get('linewidth', 1.5))

        # 右图：Max Degree
        ax2.plot(x_axis, results_deg[method], label=method,
                 color=styles[method]['color'], linestyle=styles[method]['linestyle'],
                 linewidth=styles.get(method, {}).get('linewidth', 1.5))

    ax1.set_title('IEEE 118: GCC Size vs Nodes Removed', fontsize=14)
    ax1.set_xlabel('Fraction of nodes removed', fontsize=12)
    ax1.set_ylabel('Size of GCC', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    ax2.set_title('IEEE 118: Max Degree in GCC vs Nodes Removed', fontsize=14)
    ax2.set_xlabel('Fraction of nodes removed', fontsize=12)
    ax2.set_ylabel('Max Degree of GCC', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    save_path = "ieee118_evaluation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 绘图完成！已保存至: {save_path}")


if __name__ == "__main__":
    main()