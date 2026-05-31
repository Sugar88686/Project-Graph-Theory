import os
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from env import NetworkDismantlingEnv
from agent import MaxShotAgent
from utils import generate_ba_graphs


# ==========================================
# 1. 定义各种 Baseline 算法
# ==========================================

def get_hda_sequence(graph):
    """HDA (High Degree Adaptive): 每次重新计算当前网络的最大度节点并移除"""
    G = graph.copy()
    sequence = []
    while G.number_of_nodes() > 0:
        # 只在最大的连通块中找
        components = list(nx.connected_components(G))
        if not components:
            break
        gcc = max(components, key=len)
        gcc_sub = G.subgraph(gcc)

        if len(gcc_sub) == 0:
            break

        deg = dict(gcc_sub.degree)
        best_node = max(deg, key=deg.get)
        sequence.append(best_node)
        G.remove_node(best_node)
    return sequence


def get_nda_sequence(graph):
    """NDA (Non-adaptive Degree): 仅在初始状态计算一次度数，按度数从大到小移除"""
    deg = dict(graph.degree)
    # 按度数降序排序
    sequence = sorted(deg.keys(), key=lambda x: deg[x], reverse=True)
    return sequence


def get_pagerank_sequence(graph):
    """PageRank: 根据 PageRank 中心性从大到小移除 (静态)"""
    pr = nx.pagerank(graph)
    sequence = sorted(pr.keys(), key=lambda x: pr[x], reverse=True)
    return sequence


def get_random_sequence(graph):
    """Random: 随机移除节点 (作为性能下界)"""
    nodes = list(graph.nodes)
    np.random.shuffle(nodes)
    return nodes


def get_ai_sequence(agent, graph, env):
    """AI (MaxShot): 使用我们训练好的强化学习模型进行拆解"""
    state_nx, state_pyg = env.reset()
    sequence = []
    done = False

    while not done:
        valid_nodes = env.get_valid_actions()
        if not valid_nodes:
            break
        # eval_mode=True 关闭 epsilon-greedy 探索，完全使用模型预测
        action = agent.act(state_nx, state_pyg, valid_nodes, eval_mode=True)
        if action is None:
            break
        sequence.append(action)
        (state_nx, state_pyg), _, done = env.step(action)

    return sequence


# ==========================================
# 2. 评估序列并计算 GCC 变化
# ==========================================

def evaluate_sequence(graph, sequence):
    """给定一个拆解序列，返回每一步拆解后的 GCC 相对大小"""
    G = graph.copy()
    N = G.number_of_nodes()

    # 初始状态的 GCC 大小 (通常是 1.0)
    gcc_sizes = [max((len(c) for c in nx.connected_components(G)), default=0) / N]

    for node in sequence:
        if node in G:
            G.remove_node(node)
        gcc_sizes.append(max((len(c) for c in nx.connected_components(G)), default=0) / N)

    # 如果序列长度不足 N，说明网络已经提前碎成孤立节点了，后面全补 0
    while len(gcc_sizes) <= N:
        gcc_sizes.append(0.0)

    return np.array(gcc_sizes)


# ==========================================
# 3. 主评估与绘图函数
# ==========================================

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 评估使用设备: {device}")

    # 1. 加载模型
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

    # 2. 生成测试图 (使用 100 个节点的 BA 图进行测试)
    test_graph_size = 100
    num_test_graphs = 50
    print(f"[*] 正在生成 {num_test_graphs} 张测试图 (节点数: {test_graph_size})...")
    test_graphs = generate_ba_graphs([test_graph_size], m=cfg.ba_m, num_per_size=num_test_graphs)

    # 3. 准备记录结果的字典
    methods = ['Random', 'PageRank', 'NDA', 'HDA', 'MaxShot (Ours)']
    results = {method: [] for method in methods}

    # 4. 开始评估
    print("[*] 开始对比评估...")
    for G in tqdm(test_graphs, desc="Evaluating Graphs"):
        env = NetworkDismantlingEnv(G.copy())

        # 获取各算法的拆解序列
        seqs = {
            'Random': get_random_sequence(G),
            'PageRank': get_pagerank_sequence(G),
            'NDA': get_nda_sequence(G),
            'HDA': get_hda_sequence(G),
            'MaxShot (Ours)': get_ai_sequence(agent, G, env)
        }

        # 计算每一步的 GCC 大小
        for method, seq in seqs.items():
            gcc_curve = evaluate_sequence(G, seq)
            results[method].append(gcc_curve)

    # 5. 计算平均值和 AUC
    print("\n" + "=" * 40)
    print("📊 最终评估结果 (AUC 越小越好):")
    print("=" * 40)

    mean_curves = {}
    for method in methods:
        # 将所有图的曲线堆叠并求平均
        mean_curve = np.mean(np.vstack(results[method]), axis=0)
        mean_curves[method] = mean_curve

        # 计算 AUC (曲线下面积，即所有步的 GCC 相对大小之和除以节点数)
        auc = np.sum(mean_curve) / len(mean_curve)

        if method == 'MaxShot (Ours)':
            print(f"🚀 {method:15s} : AUC = {auc:.4f}  <-- 你的 AI")
        else:
            print(f"   {method:15s} : AUC = {auc:.4f}")

    # 6. 绘制对比图
    plt.figure(figsize=(10, 7))

    # 设置 X 轴为移除节点的比例 (0 到 1)
    x_axis = np.linspace(0, 1, test_graph_size + 1)

    # 定义颜色和线型
    styles = {
        'Random': {'color': 'gray', 'linestyle': '--'},
        'PageRank': {'color': 'green', 'linestyle': '-.'},
        'NDA': {'color': 'blue', 'linestyle': ':'},
        'HDA': {'color': 'orange', 'linestyle': '-'},
        'MaxShot (Ours)': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5}
    }

    for method in methods:
        plt.plot(x_axis, mean_curves[method],
                 label=f"{method} (AUC: {np.sum(mean_curves[method]) / len(mean_curves[method]):.3f})",
                 color=styles[method]['color'],
                 linestyle=styles[method]['linestyle'],
                 linewidth=styles.get(method, {}).get('linewidth', 1.5))

    plt.title('Network Dismantling Performance Comparison', fontsize=16)
    plt.xlabel('Fraction of nodes removed', fontsize=14)
    plt.ylabel('Size of Giant Connected Component (GCC)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 保存图片
    save_path = "evaluation_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 绘图完成！已保存至: {save_path}")

    # 如果在带界面的系统中，可以取消注释下面这行直接显示图片
    # plt.show()


if __name__ == "__main__":
    main()
