# config.py
class Config:
    # --- 环境参数 ---
    max_steps_per_episode = 200  # 单回合最大步数（防止无限循环）

    # --- 图生成参数 (BA 模型) ---
    train_graph_sizes = [(30, 50), (50, 100)]  # 训练时图的节点范围
    train_graphs_per_size = 100  # 每种规模生成图数量
    ba_m = 4  # BA 模型每个新节点连边数

    # --- 编码器 (GraphSAGE) 参数 ---
    node_input_dim = 1  # 节点初始特征维度 (此处用度)
    hidden_dim = 64  # 隐藏层维度
    embedding_dim = 32  # 输出嵌入维度
    graphsage_layers = 2  # GraphSAGE 层数

    # --- 解码器 (MLP) 参数 ---
    decoder_hidden_dim = 64

    # --- 强化学习参数 ---
    gamma = 0.99  # 折扣因子
    epsilon_start = 1.0  # 初始探索率
    epsilon_end = 0.01  # 最终探索率
    epsilon_decay_steps = 5000  # 探索率衰减步数
    lr = 1e-4  # 学习率
    batch_size = 64
    buffer_capacity = 20000
    target_update_freq = 100  # 目标网络更新频率 (C)

    # --- 训练参数 ---
    total_episodes = 50000
    eval_freq = 300  # 每多少 episode 评估一次
    eval_episodes = 100  # 评估时使用的图数量

    # --- 其他 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42