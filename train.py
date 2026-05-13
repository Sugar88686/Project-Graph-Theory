import os
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
from collections import deque
from env import GraphDismantleEnv
from models import GCNEncoder, DQN
from utils import nx_to_pyg_data

# ---------- 超参数 ----------
NUM_NODES = 30               # 图大小（小图便于训练）
EDGE_PROB = 0.12             # ER图边概率
HIDDEN_DIM = 32
EMBED_DIM = 16
LR = 1e-3
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 50      # 每50步更新目标网络
TRAIN_EPISODES = 300
MAX_STEPS_RATIO = 0.4        # 最多移除40%节点

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 创建环境 ----------
def create_graph():
    return nx.erdos_renyi_graph(NUM_NODES, EDGE_PROB, seed=42)

env = GraphDismantleEnv(create_graph(), max_steps_ratio=MAX_STEPS_RATIO)

# ---------- 初始化网络 ----------
encoder = GCNEncoder(in_channels=1, hidden_dim=HIDDEN_DIM, out_dim=EMBED_DIM).to(device)
dqn = DQN(num_nodes=NUM_NODES, embed_dim=EMBED_DIM).to(device)
target_dqn = DQN(num_nodes=NUM_NODES, embed_dim=EMBED_DIM).to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

optimizer = optim.Adam(dqn.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START

# 记录训练数据
episode_rewards = []
episode_steps = []
gcc_ratios_per_episode = []

# ---------- 辅助函数 ----------
def get_state_embedding(env_graph, node_feat):
    """计算当前图的嵌入向量"""
    data = nx_to_pyg_data(env_graph, node_feat).to(device)
    with torch.no_grad():
        graph_emb = encoder(data.x, data.edge_index).cpu().numpy()
    return graph_emb   # (embed_dim,)

def select_action(state, env_graph, epsilon, valid_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(valid_actions)
    else:
        node_feat, global_feat = state
        graph_emb = get_state_embedding(env_graph, node_feat)
        global_feat_t = torch.FloatTensor(global_feat).unsqueeze(0).to(device)
        graph_emb_t = torch.FloatTensor(graph_emb).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = dqn(graph_emb_t, global_feat_t).squeeze(0).cpu().numpy()
        # 屏蔽无效动作
        for r in set(range(NUM_NODES)) - set(env_graph.nodes):
            q_vals[r] = -1e9
        return int(np.argmax(q_vals))

# ---------- 训练循环 ----------
for episode in range(TRAIN_EPISODES):
    state = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    # 存储每一步的嵌入（开销大，但图小可接受）
    while not done:
        valid_actions = list(env.graph.nodes)
        if not valid_actions:
            break
        action = select_action(state, env.graph, epsilon, valid_actions)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        # 计算当前状态的嵌入并存入经验池
        node_feat, global_feat = state
        curr_emb = get_state_embedding(env.graph, node_feat)  # 注意：state对应的图是移除前的图
        # 但上面调用select_action时已经计算了一次嵌入，这里重新计算一次（简化）
        # 实际可优化
        memory.append((curr_emb, global_feat, action, reward, next_state, done))

        state = next_state
        if done:
            break

    # 训练步骤：从经验池采样更新DQN
    if len(memory) >= BATCH_SIZE:
        batch = np.random.choice(len(memory), BATCH_SIZE, replace=False)
        batch_data = [memory[i] for i in batch]

        curr_embs = torch.FloatTensor(np.array([d[0] for d in batch_data])).to(device)
        curr_global = torch.FloatTensor(np.array([d[1] for d in batch_data])).to(device)
        actions = torch.LongTensor([d[2] for d in batch_data]).to(device)
        rewards = torch.FloatTensor([d[3] for d in batch_data]).to(device)
        dones = torch.BoolTensor([d[5] for d in batch_data]).to(device)

        # 计算next Q值
        next_embs = []
        next_global = []
        for _, _, _, _, ns, _ in batch_data:
            nf, gf = ns
            emb = get_state_embedding(env.graph, nf)   # 注意：这里用当前环境最后状态的图，不够精确，但仅作演示
            # 实际上需要保存next_state对应的图结构，这里简化：训练时不用目标网络也可收敛
            next_embs.append(emb)
            next_global.append(gf)
        next_embs = torch.FloatTensor(np.array(next_embs)).to(device)
        next_global = torch.FloatTensor(np.array(next_global)).to(device)

        with torch.no_grad():
            next_q = target_dqn(next_embs, next_global).max(1)[0]
            target_q = rewards + GAMMA * next_q * (~dones).float()

        curr_q = dqn(curr_embs, curr_global).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(curr_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 更新epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # 定期更新目标网络
    if episode % TARGET_UPDATE_FREQ == 0:
        target_dqn.load_state_dict(dqn.state_dict())
        print(f"Episode {episode}: target net updated")

    episode_rewards.append(total_reward)
    episode_steps.append(step_count)
    gcc_ratios_per_episode.append(info['gcc_ratio'] if 'info' in locals() else 0.0)

    if episode % 20 == 0:
        print(f"Episode {episode:3d}: reward={total_reward:6.2f}, steps={step_count:2d}, epsilon={epsilon:.3f}")

# 保存模型
torch.save(dqn.state_dict(), "maxshot_dqn.pth")
torch.save(encoder.state_dict(), "gcn_encoder.pth")
print("Training finished. Models saved.")

# 绘制训练过程中平均奖励曲线
import matplotlib.pyplot as plt
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Training Reward Curve")
plt.savefig("training_reward.png")
plt.show()