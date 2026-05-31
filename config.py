import torch
import os


class Config:
    # ---------- 环境与回合控制 ----------
    max_steps_per_episode = 100  # 训练图最大节点数为100，最多拆解100步

    # ---------- 训练图生成参数 ----------
    train_graph_sizes = [(30, 50), (50, 100)]
    train_graphs_per_size = 100
    ba_m = 4

    # ---------- 编码器参数（GraphSAGE）----------
    node_input_dim = 1
    hidden_dim = 64
    embedding_dim = 64
    graphsage_layers = 5  # 保持5层，确保充分捕获全局结构以超越HDA

    # ---------- Dueling DQN 解码器参数 ----------
    decoder_hidden_dim = 64
    dueling_hidden_dim = 64

    # ---------- 强化学习参数 ----------
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 8000  # 缩短探索期，尽早利用模型经验
    lr = 2e-4  # 提升学习率，加速收敛
    batch_size = 64
    buffer_capacity = 20000
    target_update_freq = 100

    # 【提速优化】：每走 4 步更新一次网络，大幅减少 CPU-GPU 切换开销
    train_freq = 4

    # ---------- 训练与评估参数 ----------
    total_episodes = 30000  # 缩减总回合数，确保在5-7小时内完赛
    eval_freq = 500  # 降低评估频率，大幅节省 CPU 算力
    eval_episodes = 100

    # 【多进程提速】：评估阶段使用的 CPU 进程数（预留2个核心给系统）
    eval_num_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 1

    # ---------- 其他 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # ---------- 混合精度训练 ----------
    use_amp = True  # 开启混合精度，RTX 5070 完美支持
