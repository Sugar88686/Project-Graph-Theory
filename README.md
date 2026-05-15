# Project-Graph-theory
下面，我将对 **MaxShot 复现框架**中的每一个文件进行**详细完整的介绍**，包括其**具体内容、核心函数/类的说明、设计意图、与论文的对应关系**，以及**学习研究时如何理解与扩展**。

---

## 1. `config.py` – 配置管理

### 用途
集中管理所有超参数和运行配置，避免在代码中硬编码，方便调参和实验复现。

### 主要内容
```python
class Config:
    max_steps_per_episode = 200
    train_graph_sizes = [(30, 50), (50, 100)]
    train_graphs_per_size = 100
    ba_m = 4
    node_input_dim = 1
    hidden_dim = 64
    embedding_dim = 32
    graphsage_layers = 2
    decoder_hidden_dim = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 5000
    lr = 1e-4
    batch_size = 64
    buffer_capacity = 20000
    target_update_freq = 100
    total_episodes = 50000
    eval_freq = 300
    eval_episodes = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
```

### 详细解释
- **环境参数**：`max_steps_per_episode` 防止单回合无限循环（极端情况可能难以完全瓦解）。
- **图生成参数**：BA 图节点范围 (30-50, 50-100)，每种规模 100 张图，`m=4` 是 BA 模型的平均度控制。论文中训练时使用 BA 图。
- **编码器参数**：`node_input_dim=1` 因为初始特征仅用度（归一化后）。`embedding_dim` 是最终节点嵌入维度。`graphsage_layers=2` 表示聚合两跳邻居信息。
- **解码器参数**：MLP 的隐藏层维度。
- **RL 参数**：`gamma` 折扣因子（重视长期奖励）；`epsilon` 从 1.0 衰减到 0.01；`buffer_capacity` 经验回放大小；`target_update_freq` 对应论文中的 `C` 步更新一次目标网络。
- **训练参数**：`total_episodes=50000` 与论文一致；`eval_freq=300` 每 300 回合评估一次；`eval_episodes=100` 评估时使用 100 张图。
- **其他**：自动选择 GPU/CPU，固定随机种子保证可重复性。

### 学习研究要点
- 理解每个超参数对训练的影响（例如折扣因子 `gamma` 越大，智能体越考虑远期收益）。
- 可以添加更多配置（如 SBM 图参数、不同编码器选择、奖励缩放等）。

---

## 2. `env.py` – 网络瓦解环境 (MDP)

### 用途
实现网络瓦解任务的**马尔可夫决策过程**，包括状态表示、动作空间、奖励计算、终止条件。它封装了图的操作，为智能体提供交互接口。

### 主要内容

#### `NetworkDismantlingEnv` 类
- **初始化** `__init__(self, graph)`：存储原始图，并调用 `reset()`。
- **`reset()`**：重置当前图为原始图，计算初始 score 并存储，返回初始状态（当前图对象）。
- **`_get_state()`**：返回当前图（将被编码器处理）。
- **`get_gcc()`**：计算当前图的最大连通分量子图（使用 `networkx.connected_components`）。
- **`compute_score(graph)`**：实现论文公式 (1)  
  \[
  \text{score}(G) = \frac{|GCC(G)|}{|G|} \times \frac{\max \deg(GCC(G))}{|G|}
  \]  
  注意处理空图边界条件（score = 0）。
- **`step(self, node)`**：
  - 从当前图中移除节点 `node`。
  - 计算新的 score。
  - 奖励 `r = -(new_score - prev_score)`（负号因为 RL 最大化奖励，我们要最小化 score）。
  - 更新 prev_score。
  - 判断是否终止（GCC 为空）。
  - 返回 `(next_state, reward, done)`。
- **`get_valid_actions()`**：返回当前 GCC 中的所有节点（动作空间）。

### 与论文的对应
- 状态：论文中状态是“剩余 GCC 的大小”，但实际编码器需要整个图的结构信息。因此 `state` 返回整个图对象，由后续 GNN 自动提取 GCC 相关信息。这是一种合理的扩展。
- 奖励：完全按照论文公式 (1)。
- 终止：GCC 完全消失。

### 学习研究要点
- 验证奖励函数：手动删除一个高度数节点，观察 score 是否下降，奖励是否为正。
- 注意 `compute_score` 的效率：每次调用都重新计算 GCC 和最大度数，对于大型图可能较慢，可考虑增量更新（但初期不必）。
- 可以考虑加入“提前终止”条件，如 GCC 大小小于某个阈值。

---

## 3. `models.py` – 图编码器与 Q 解码器

### 用途
定义 **GraphSAGE 编码器**（将图结构映射为节点嵌入）和 **Q 值解码器**（将节点嵌入+图嵌入映射为标量 Q 值）。这是 MaxShot 框架的核心神经网络部分。

### 主要内容

#### `GraphSAGEEncoder` 类
- 继承 `nn.Module`。
- 构造函数：接收 `in_dim, hidden_dim, out_dim, num_layers`。
- 内部使用 `torch_geometric.nn.SAGEConv` 堆叠多层。
- `forward(x, edge_index)`：逐层应用 SAGEConv + ReLU，最后通过线性层得到 `[num_nodes, out_dim]` 的节点嵌入。

#### `QDecoder` 类
- 构造函数：接收 `node_emb_dim, graph_emb_dim, hidden_dim`。
- `forward(node_emb, graph_emb)`：
  - 将图嵌入 `graph_emb` 扩展为与节点数相同的矩阵。
  - 拼接节点嵌入和图嵌入。
  - 通过两层 MLP（隐藏层 `hidden_dim`，输出 1 维）得到每个节点的 Q 值。
  - 返回 `[num_nodes]` 张量。

#### `get_graph_data_from_nx(graph, node_feat_func)` 函数
- 将 NetworkX 图转换为 PyTorch Geometric 的 `Data` 对象。
- 默认节点特征：归一化的度（度/最大度）。也可以传入自定义特征函数。
- 使用 `from_networkx` 自动构建 `edge_index`（COO 格式）。

### 与论文的对应
- 编码器：论文使用 GraphSAGE，支持归纳学习，适用于动态变化的图。这里实现了基础版本。
- 虚拟节点：论文提到引入虚拟节点来捕获全局特征，但本框架简化了，直接用节点嵌入的均值作为图嵌入（在 `agent.py` 中实现）。若要严格复现，可以在图中添加一个虚拟节点并与所有真实节点连接，然后取该虚拟节点的嵌入作为图嵌入。
- 解码器：论文说“线性解码器”，但实际可以用 MLP 提升表达能力。

### 学习研究要点
- 尝试不同的 GNN 编码器（GCN, GIN, GAT）对比效果。
- 尝试不同的图嵌入聚合方式（max, sum, 虚拟节点）。
- 增加节点特征：度、聚类系数、PageRank 等，观察性能变化。

---

## 4. `buffer.py` – 经验回放缓冲区

### 用途
存储智能体与环境交互产生的经验元组 `(state, action, reward, next_state, done)`，并在训练时随机采样小批量，打破时间相关性。

### 主要内容

#### `ReplayBuffer` 类
- 构造函数：`__init__(self, capacity)`，使用 `deque(maxlen=capacity)` 作为存储容器。
- `push(self, state_graph, action, reward, next_state_graph, done)`：
  - 为了存储图对象，使用 `copy.deepcopy` 保存当前图和下一图的快照。
  - **注意**：直接存储整个图会占用大量内存（尤其是大规模真实数据集）。原型实现可以接受，但优化时只需存储图的差异（如删除的节点序列）或存储节点集/边集的轻量表示。
- `sample(self, batch_size)`：随机返回 `batch_size` 个元组。
- `__len__(self)`：返回当前缓冲区大小。

### 设计考虑
- 为什么需要 `deepcopy`？因为图对象在后续步骤中会被修改，如果不拷贝，回放时状态会发生变化。
- 内存优化提示：对于大型图（数万节点），存储完整图不现实。可以改为存储**原始图索引 + 已删除节点序列**，然后根据这些信息重构状态。但这会增加计算开销。论文中未明确实现细节，但这是工程常见权衡。

### 学习研究要点
- 理解经验回放对 DQN 稳定性的重要性。
- 实现优先经验回放（PER）可能提升学习效率。
- 对于大型图，探索更高效的状态存储方式（如只存储图的边列表和当前 GCC 节点集）。

---

## 5. `agent.py` – MaxShot 智能体 (Double DQN)

### 用途
整合编码器、解码器、目标网络、epsilon-greedy 策略，实现 **Double DQN 算法**。负责动作选择、经验存储、训练更新（损失计算、反向传播、目标网络同步）。

### 主要内容

#### `MaxShotAgent` 类

**初始化**
- 创建当前网络（encoder + decoder）和目标网络（结构相同，参数克隆）。
- 优化器（Adam）。
- 超参数：`epsilon`, `gamma`, `update_counter`, `target_update_freq` 等。

**`get_q_values(self, graph, valid_nodes)`**
- 将图转为 PyG Data，通过编码器得到节点嵌入，计算图嵌入（均值），再通过解码器得到所有节点的 Q 值。
- 只返回 `valid_nodes` 中节点的 Q 值和它们在原始节点列表中的索引。
- 注意：`valid_nodes` 通常是当前 GCC 中的节点，但编码器输入整个图，这是合理的。

**`act(self, graph, valid_nodes, eval_mode=False)`**
- 根据 epsilon-greedy 选择动作。
- 如果 `eval_mode=True`，则不随机探索，始终选最大 Q 值（用于评估）。
- 返回选中的节点 ID。

**`update_epsilon(self)`**
- 线性衰减 epsilon。

**`train_step(self, batch)`**
- 输入：一批经验元组 `(state, action, reward, next_state, done)`。
- 步骤：
  1. 对每个经验，计算当前 Q(s, a)（通过当前网络）。
  2. 对每个经验，计算目标 Q 值：
     - 如果 `done=True`，目标 = reward。
     - 否则，使用**目标网络**计算 `max_{a'} Q_target(s', a')`，然后 `reward + gamma * max_next_q`。
  3. 计算均方误差损失。
  4. 梯度清零，反向传播，优化器更新。
  5. 增加 `update_counter`，若达到 `target_update_freq` 则同步目标网络参数。
- 返回损失值（用于监控）。

### 与论文的对应
- Double DQN：论文第 4.2 节明确使用 Double DQN，这里实现了标准 Double DQN 更新（动作选择用当前网络，价值评估用目标网络）。严格实现可参考 Hasselt 2010 论文。
- 目标网络更新频率 `C` 对应论文中的 `C` 步。
- 损失函数对应公式 (2)。

### 学习研究要点
- 实现标准的 Double DQN：注意在计算目标值时，应先使用当前网络选择最佳动作，再用目标网络评估该动作的价值。本代码简化了（直接取 max 来自目标网络），实际应分离。可以进一步改进。
- 探索使用 Dueling DQN 架构提升性能。
- 监控 Q 值是否发散或过估计。

---

## 6. `utils.py` – 工具函数

### 用途
提供图生成、评估指标、基线算法实现、可视化等辅助功能。

### 主要内容

#### 图生成函数
- **`generate_ba_graphs(sizes, m=4, num_per_size=100)`**：
  - 支持 `sizes` 为 `[(30,50), (50,100)]` 等范围，每个范围内随机选择节点数生成 BA 图。
  - 返回 `list` 的 NetworkX 图对象。
- **`generate_sbm_graphs(community_sizes, p_intra, p_inter, num_per_size)`**：
  - 生成随机块模型图，用于论文中的 SBM 实验。

#### 评估函数
- **`evaluate_agent(agent, test_graphs, env_class)`**：
  - 对每个测试图，运行智能体（`eval_mode=True`）直到终止。
  - 记录每一步的 GCC 大小和 GCC 内最大度数。
  - 计算累积指标：`sum(gcc_sizes) / total_nodes` 和 `sum(max_degs) / total_nodes`（近似曲线下面积）。
  - 返回所有图的平均值。

#### 基线算法示例
- **`BaselineHDA` 类**：
  - `dismantle(graph)` 方法：模拟最高度算法，返回删除节点序列。
  - 可扩展 `BaselineHBA`, `BaselineCI` 等。

#### 其他可能函数
- `plot_curves()`：绘制累积 GCC 大小曲线（类似论文图 2、3）。
- `compute_pareto_frontier()`：计算帕累托前沿（论文图 4）。

### 学习研究要点
- 确保评估指标与论文一致：论文中“accumulated GCC size”是曲线下面积，注意归一化方式。
- 基线算法的实现可参考 NetworkX 内置中心性函数（`betweenness_centrality`, `pagerank` 等），但注意计算效率。
- 对于大型真实数据集，评估时需注意内存和时间。

---

## 7. `train.py` – 训练主脚本

### 用途
整合所有组件，运行完整的训练循环，包括图生成、智能体与环境交互、经验存储、训练更新、定期评估和保存最佳模型。

### 主要内容

**`set_seed(seed)`**：固定随机种子。

**`train()` 函数**
1. **配置与设备**：加载 `Config`，设置设备，固定种子。
2. **生成训练图**：调用 `generate_ba_graphs` 生成训练图集（论文中使用 BA 图训练）。
3. **生成验证图**：用于定期评估（例如固定 50 节点的 BA 图）。
4. **初始化智能体和缓冲区**。
5. **循环 episodes**（共 `total_episodes` 次）：
   - 随机选择一个训练图，创建环境。
   - 重置环境，获得初始状态。
   - 在每一步内：
     - 获取有效动作（GCC 节点）。
     - 智能体选择动作。
     - 执行 `step`，获得奖励、下一状态、终止标志。
     - 存储经验到缓冲区。
     - 如果缓冲区足够大，采样一个 batch 并调用 `agent.train_step`。
     - 更新状态。
   - 每步后调用 `agent.update_epsilon` 衰减探索率。
   - 每 `eval_freq` 个 episode：使用验证图评估智能体，打印 GCC AUC 和 MaxDeg AUC，若优于最佳则保存模型。
6. 训练结束。

### 与论文的对应
- 训练 episode 数：50,000。
- 评估频率：每 300 episode。
- 验证集：100 个 BA 图（与论文一致）。
- 模型保存：根据累积 GCC 大小（论文表 2/3 中的指标）选择最佳。

### 学习研究要点
- 训练时注意时间：50,000 episodes 在普通 GPU 上可能需要数天。可先减少 episodes 测试代码正确性。
- 增加日志记录（使用 TensorBoard 或 wandb）监控损失、平均奖励、Q 值等。
- 尝试在多进程环境中并行生成环境交互（例如使用 Ray 或 Python multiprocessing）加速训练。

---

## 8. `evaluate.py` – 评估与对比

### 用途
用于加载训练好的模型，在测试集（合成图或真实数据集）上进行评估，并与多种基线算法对比，生成论文中的表格数据。

### 主要内容

- **`load_model(agent, checkpoint_path)`**：加载保存的模型参数。
- **`compare_baselines(test_graphs)`**：运行 HDA、HBA、CI 等基线，测量累积指标和运行时间。
- **`evaluate_agent_baseline(algorithm_class, graphs)`**：针对基线的通用评估函数。
- **`evaluate_real_datasets()`**：加载真实数据集（如 Crime, Enron）并评估 MaxShot 和基线。
- **主程序**：
  - 加载配置和测试图（例如 BA 50-100 节点，或 SBM 图）。
  - 加载训练好的 MaxShot 模型。
  - 计算 MaxShot 的累积指标和运行时间。
  - 计算各基线的对应指标。
  - 打印或保存结果（类似论文表 2、3、6、7、8、9）。

### 与论文的对应
- 表 2、3：SBM 图上的累积最大度数和累积 GCC 大小。
- 表 6、7：真实数据集上的相同指标。
- 表 8、9：运行时间对比。
- 图 4：帕累托前沿（需要多目标优化分析）。

### 学习研究要点
- 实现论文中的全部基线（HBA, CI, MinSum, CoreHD, BPD, GND, FINDER 等）。部分算法有公开代码可参考。
- 运行时间测量：在相同硬件条件下多次运行取平均值（论文中给出了标准差）。
- 真实数据集可从 SNAP 下载（如 Digg, Enron, Epinions），注意数据预处理（转为无向简单图）。
- 帕累托前沿：需要同时考虑 GCC 面积和最大度数面积两个目标，可使用 NSGA-II 或简单网格搜索。

---

## 总结：文件依赖关系图

```
config.py
    ↓
env.py  ←  utils.py (图生成/评估)
    ↓
models.py (encoder/decoder)
    ↓
buffer.py
    ↓
agent.py (依赖 models, config)
    ↓
train.py (依赖 agent, env, buffer, utils)
    ↓
evaluate.py (依赖 agent, utils, 基线)
```

## 建议的学习路径

1. **先阅读论文**，理解每个模块的数学原理。
2. **运行最小示例**：将 `total_episodes` 改为 100，图大小 20 节点，验证代码无报错。
3. **单元测试**：单独测试 `env.py` 的奖励函数，单独测试 `models.py` 的前向传播。
4. **调试训练**：打印每回合的总奖励，观察是否上升；打印 Q 值是否合理。
5. **调参**：从简单的参数开始（小图、小 episode），逐步扩大。
6. **对比实验**：实现 HDA 基线，确保 MaxShot 能明显优于 HDA。
7. **复现论文结果**：使用论文中的超参数和数据集，跑出类似表格。

希望这份详细的介绍能帮助你顺利复现并深入理解 MaxShot！如果有任何具体文件或函数需要更深入的说明，欢迎继续提问。






# 三人分工
基于我之前给出的 MaxShot 复现框架（8个核心文件），下面仿照你提供的图片格式，制定一份三人分工计划。每个角色负责一组文件，接口约定明确，按周推进，包含数据集推荐和风险预案。

---

## 三人角色与文件对应

| 角色 | 职责 | 核心文件 |
|------|------|----------|
| **A号 – 环境与数据工程师** | 网络瓦解环境、图生成、评估指标、基线辅助 | `env.py`, `utils.py`（图生成/评估部分）, `data_loader.py`（真实数据集） |
| **B号 – GNN与模型工程师** | 图编码器、Q解码器、节点特征设计 | `models.py`, `gnn_utils.py`（虚拟节点等扩展） |
| **C号 – RL与训练工程师** | 智能体（Double DQN）、经验回放、训练主控、评估脚本 | `agent.py`, `buffer.py`, `config.py`, `train.py`, `evaluate.py`, `baselines.py` |

---

## 第一周：各自构建核心模块，互不等待

### Day1（共同）
- 配置环境：`pip install torch torch-geometric networkx numpy tqdm matplotlib`
- 读懂 MaxShot 论文核心：双指标奖励（公式1）、Double DQN（公式2）、编码器-解码器架构
- **接口约定**：
  - 环境 → 智能体：`state = env.reset()` 返回 `nx.Graph`；`valid_actions = env.get_valid_actions()` 返回 `List[node_id]`；`next_state, reward, done = env.step(action)`
  - 模型（编码器）→ 智能体：`node_emb = encoder(data.x, data.edge_index)` 返回 `[N, emb_dim]`
  - 智能体 → 环境：`action = agent.act(state, valid_actions)`

### Day2-4（并行开发）

#### A号 – 环境与数据
- 实现 `env.py`：`NetworkDismantlingEnv` 类（GCC计算、score、奖励）
- 实现 `utils.py` 中的图生成：`generate_ba_graphs()`, `generate_sbm_graphs()`
- 加载第一个小数据集：Karate Club（34节点）用于调试
- **验证**：手动删除节点，打印 reward 符号正确（score下降时reward为正）

#### B号 – GNN模型
- 安装 PyTorch Geometric，实现 `models.py`
- 编码器：`GraphSAGEEncoder`（输入节点度特征，输出64维嵌入）
- 解码器：`QDecoder`（节点嵌入 + 图嵌入 → Q值）
- 实现 `get_graph_data_from_nx()`：将nx图转为PyG Data，节点特征=归一化度
- **验证**：单图前向传播，输出节点嵌入形状正确，Q值形状正确

#### C号 – RL框架
- 实现 `buffer.py`：`ReplayBuffer`（容量20000）
- 实现 `agent.py` 骨架：`__init__`, `act`（ε-greedy）, `remember`, `update_epsilon`
- 实现 `config.py` 基础超参
- 写一个**随机环境测试**：用随机策略跑一个episode，确保无报错

### Day5-7（完善与基线）

#### A号 – 基线算法
- 实现 `utils.py` 中的简单基线：随机移除（Random）、度贪心（Degree）、介数贪心（Betweenness）
- 实现评估函数：`evaluate_agent()` 返回累积GCC大小和累积最大度数
- 准备真实数据集加载函数（Crime, Enron等），存为 `nx.Graph`

#### B号 – 特征设计
- 增加节点特征：度、k-core值、局部聚类系数
- 实现“虚拟节点”概念（可选）：添加一个与所有节点相连的虚拟节点，取其嵌入作为图嵌入
- 支持动态更新：每次移除节点后，编码器重新计算剩余图的嵌入

#### C号 – 奖励设计与DQN
- 实现 `agent.train_step()`：Double DQN 损失（公式2），目标网络同步
- 在随机环境中跑通完整训练循环（单图，少量episode）
- 调参实验：奖励中的权重（若想引入 `α×degree` 可修改环境奖励函数）
- 实现掩码机制：确保智能体不会选择已移除节点

### 第7天晚：三模块联调
- 用 Karate Club 图，跑一个完整 episode：
  - A 提供环境 → B 提供编码器 → C 提供智能体 → 循环 step
- 检查：每一步 action 合法，reward 计算正确，loss 能下降
- 保存第一个 checkpoint

---

## 第二周：整合训练 · 实验对比 · 报告

### Day8-10（并行优化）

#### A号 – 多图训练与数据增强
- 生成多种规模的 BA 图（30-50, 50-100）用于训练
- 实现 SBM 图生成，供消融实验
- 绘制 LCC 缩减曲线（类似论文图2/3）

#### B号 – GNN端到端微调
- 将编码器与智能体联合训练（已在 agent 中实现）
- 消融实验：有无 GNN（直接用度作为 Q 值）的对比
- t-SNE 可视化节点嵌入，观察移除前后结构变化

#### C号 – 超参调优与训练稳定性
- 调优学习率、ε衰减、折扣因子、目标网络更新频率
- 训练完整 50000 episodes，每 300 步评估
- 记录训练曲线（损失、平均奖励、评估指标）
- 对比 MaxShot 论文中的表格数据

### Day11-14：三人合作收尾
- **A号**：撰写“图论与环境建模”章节（环境设计、GCC、奖励函数）
- **B号**：撰写“GNN模型设计”章节（GraphSAGE、虚拟节点、特征工程）
- **C号**：撰写“实验结果与分析”章节（表格、曲线、与基线对比）
- 共同制作可视化图表（帕累托前沿、训练收敛图）和答辩PPT

---

## 代码仓库结构（对应分工）

```
maxshot/
├── config.py                 # C号
├── env.py                    # A号
├── models.py                 # B号
├── agent.py                  # C号
├── buffer.py                 # C号
├── utils.py                  # A号 + 部分B号（特征）
├── train.py                  # C号
├── evaluate.py               # C号
├── baselines.py              # C号（或A号）
├── data/                     # A号：存放真实数据集
│   ├── karate.gml
│   ├── ieee118.gml
│   └── enron.gpickle
├── notebooks/                # 三人共享：可视化分析
└── reports/                  # 三人共享：实验报告和PPT
```

---

## 数据集推荐（由易到难）

| 顺序 | 数据集 | 节点数 | 用途 | 负责人 |
|------|--------|--------|------|--------|
| ① | Karate Club | 34 | 调试、单图训练 | A号准备 |
| ② | IEEE 118-Bus (电网) | 118 | 验证中等规模 | A号 |
| ③ | Email-Enron | 36,692 | 真实社交网络 | A号 |
| ④ | Digg / Epinions | 数万 | 论文对比实验 | A号 |

先在①上调通整个流程，再逐步上②③④验证效果。

---

## 最大风险与预案

| 风险 | 概率 | 影响 | 预案 |
|------|------|------|------|
| **DQN 训练不收敛**（Q值不下降） | 高 | 高 | 第10天仍不收敛，改用 **GNN贪心评分**：直接用编码器输出的节点嵌入加上度信息，计算一个启发式分数（如嵌入与全局图嵌入的余弦相似度 × 度），按分数排序选择节点。该方法不依赖RL，仍能体现GNN优势，可作为论文的对比基线。 |
| 真实数据集加载时间过长 | 中 | 中 | 预先将图序列化存为 `.gpickle`，评估时直接加载；使用稀疏邻接矩阵 |
| 多图训练内存溢出 | 低 | 中 | 使用生成器动态生成 BA 图，不一次性全部加载；限制最大图尺寸 |

---

## 核心评估指标

- **LCC/N 曲线**：移除 k% 节点后，剩余最大连通分量大小除以总节点数。曲线越低，瓦解能力越强。
- **累积 GCC 大小**（论文表3/7）：移除过程中 GCC 曲线下面积。
- **累积最大度数**（论文表2/6）：GCC 内最大度数的曲线下面积。
- **运行时间**（论文表8/9）：毫秒/秒级对比。
- **帕累托前沿**（论文图4）：同时考虑两个累积指标的多目标优化。

最终对比基线：Random, Degree, Betweenness, CI, CoreHD, BPD, FINDER 等。

---

如果需要每个角色更细致的每日任务清单或具体的接口伪代码，我可以进一步补充。
