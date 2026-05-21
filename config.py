class Config:
    # ---------- 环境与回合控制 ----------
    max_steps_per_episode = 100          # 原论文未明确，但训练图节点最大50，100步足够

    # ---------- 训练图生成参数（BA 模型）----------
    train_graph_sizes = [(25, 50)]       # 缩小节点范围，兼顾速度与结构多样性
    train_graphs_per_size = 20           # 减少预生成图数量，大幅降低采样开销
    ba_m = 4                             # 保持论文设置

    # ---------- 编码器参数（GraphSAGE）----------
    node_input_dim = 1                   # 归一化度（论文设计）
    hidden_dim = 32                      # 从64降低，减少参数量
    embedding_dim = 32                   # 从64降低，平衡效果与速度
    graphsage_layers = 2                 # 从5降低，显著减少计算量（原论文5层过深）

    # ---------- 解码器参数（Q解码器 MLP）----------
    decoder_hidden_dim = 32              # 与 hidden_dim 保持一致

    # ---------- 强化学习参数 ----------
    gamma = 0.99                         # 折扣因子（标准值）
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 2000           # 因总回合数减少，加速探索衰减
    lr = 1e-4                            # 论文表5固定值
    batch_size = 64                      # 保持论文batch size，RTX 5070显存足够
    buffer_capacity = 10000              # 从20000降低，节约内存
    target_update_freq = 100             # 论文C=100，保持不变

    # ---------- 训练与评估参数 ----------
    total_episodes = 5000                # 从50000大幅缩减，确保1小时内完成
    eval_freq = 200                      # 每200回合评估一次，观察收敛趋势
    eval_episodes = 20                   # 评估时用20张图（原100张，减少开销）

    # ---------- 其他 ----------
    device = "cuda"                      # 强制使用GPU（RTX 5070）
    seed = 42                            # 固定随机种子

    # ---------- 可选：混合精度训练（大幅加速）----------
    use_amp = True                       # 若代码支持，开启自动混合精度
