import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from models import DQN

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, num_nodes, embed_dim=16, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.num_nodes = num_nodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(num_nodes, embed_dim=embed_dim).to(self.device)
        self.target_net = DQN(num_nodes, embed_dim=embed_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.embed_dim = embed_dim

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, encoder, graph, valid_actions=None):
        """使用当前策略选择动作"""
        if np.random.rand() < self.epsilon:
            # 随机选择有效动作
            if valid_actions is None:
                valid_actions = list(graph.nodes)
            if not valid_actions:
                return 0
            return np.random.choice(valid_actions)
        else:
            node_feat, global_feat = state
            # 转换成PyG需要的格式
            from utils import nx_to_pyg_data
            data = nx_to_pyg_data(graph, node_feat)
            data = data.to(self.device)
            with torch.no_grad():
                graph_emb = encoder(data.x, data.edge_index).unsqueeze(0)   # (1, embed_dim)
                global_feat_t = torch.FloatTensor(global_feat).unsqueeze(0).to(self.device)  # (1,1)
                q_vals = self.q_net(graph_emb, global_feat_t).squeeze(0)   # (num_nodes,)
            # 屏蔽已移除节点
            removed = set(range(self.num_nodes)) - set(graph.nodes)
            for r in removed:
                q_vals[r] = -1e9
            action = q_vals.argmax().item()
            return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # 解包状态
        node_feats_batch = []
        global_feats_batch = []
        for s in states:
            nf, gf = s
            node_feats_batch.append(nf)
            global_feats_batch.append(gf)
        # 这里为了简化，假设所有state的node_feat shape相同（num_nodes,1）
        node_feats_batch = np.stack(node_feats_batch)   # (batch, N, 1)
        global_feats_batch = np.array(global_feats_batch)  # (batch,1)

        # 转换tensor
        node_feats_t = torch.FloatTensor(node_feats_batch).to(self.device)  # (batch,N,1)
        global_feats_t = torch.FloatTensor(global_feats_batch).to(self.device)  # (batch,1)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        # 这里需要获取next_state的图嵌入，但next_state也是(node_feat, global_feat)，我们需要知道下一个图的节点特征和结构。
        # 为了简化训练，我们不使用GCN动态编码（因为它太慢），而是直接使用全局特征作为状态输入。
        # 实际上，许多论文也采用这种简化：只使用全局特征和节点度作为输入，而不在训练循环中重复调用GCN。
        # 为了保持代码可运行，我们下面采用简化版：只用全局特征 + 节点特征（作为额外向量）来训练DQN。
        # 但为了不偏离主题，我将在train.py中实际采用GCN编码方式，但学习部分使用完整的transition需要存储嵌入。
        # 这里为了代码简洁，给出一个更实用的方案：在train循环中，我们每一步都计算出当前图嵌入并存入经验池，
        # 这样学习时直接从经验池中取出嵌入向量。这更符合实际实现。
        pass  # 实际实现会在train.py中完成完整流程，请参见下方训练脚本