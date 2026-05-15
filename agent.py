# agent.py
import torch
import torch.optim as optim
import random
import numpy as np
from models import GraphSAGEEncoder, QDecoder, get_graph_data_from_nx


class MaxShotAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.encoder = GraphSAGEEncoder(
            config.node_input_dim, config.hidden_dim, config.embedding_dim,
            num_layers=config.graphsage_layers
        ).to(device)
        self.decoder = QDecoder(
            config.embedding_dim, config.embedding_dim, config.decoder_hidden_dim
        ).to(device)

        self.target_encoder = GraphSAGEEncoder(
            config.node_input_dim, config.hidden_dim, config.embedding_dim,
            num_layers=config.graphsage_layers
        ).to(device)
        self.target_decoder = QDecoder(
            config.embedding_dim, config.embedding_dim, config.decoder_hidden_dim
        ).to(device)

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_decoder.load_state_dict(self.decoder.state_dict())

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.lr
        )

        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps
        self.gamma = config.gamma
        self.update_counter = 0
        self.target_update_freq = config.target_update_freq

    def get_q_values(self, graph, valid_nodes):
        """返回当前图中 valid_nodes 对应的 Q 值"""
        data = get_graph_data_from_nx(graph).to(self.device)
        node_emb = self.encoder(data.x, data.edge_index)  # [N, emb_dim]
        graph_emb = node_emb.mean(dim=0)  # [emb_dim]
        q_all = self.decoder(node_emb, graph_emb)  # [N]
        # 将节点ID映射到顺序索引
        node_list = list(graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        valid_indices = [node_to_idx[n] for n in valid_nodes if n in node_to_idx]
        if not valid_indices:
            return torch.tensor([]).to(self.device), []
        q_valid = q_all[valid_indices]
        return q_valid, valid_indices

    def act(self, graph, valid_nodes, eval_mode=False):
        """epsilon-greedy 动作选择"""
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(valid_nodes)
        with torch.no_grad():
            q_vals, idx_list = self.get_q_values(graph, valid_nodes)
            if len(q_vals) == 0:
                return None
            best_local_idx = torch.argmax(q_vals).item()
            return valid_nodes[best_local_idx]

    def update_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train_step(self, batch):
        """
        对一批经验进行 Double DQN 更新。
        batch: list of (state_graph, action, reward, next_state_graph, done)
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        # 计算当前 Q(s, a)
        q_current_list = []
        for state, action in zip(states, actions):
            data = get_graph_data_from_nx(state).to(self.device)
            node_emb = self.encoder(data.x, data.edge_index)
            graph_emb = node_emb.mean(dim=0)
            q_all = self.decoder(node_emb, graph_emb)
            node_list = list(state.nodes)
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            if action in node_to_idx:
                q_current_list.append(q_all[node_to_idx[action]])
            else:
                q_current_list.append(torch.tensor(0.0, device=self.device))
        q_current = torch.stack(q_current_list)

        # 计算目标 Q 值
        q_target_list = []
        for reward, next_state, done in zip(rewards, next_states, dones):
            if done:
                q_target_list.append(torch.tensor(reward, dtype=torch.float, device=self.device))
            else:
                # 使用目标网络计算 max_{a'} Q_target(s', a')
                data_next = get_graph_data_from_nx(next_state).to(self.device)
                with torch.no_grad():
                    next_node_emb = self.target_encoder(data_next.x, data_next.edge_index)
                    next_graph_emb = next_node_emb.mean(dim=0)
                    next_q_all = self.target_decoder(next_node_emb, next_graph_emb)
                max_next_q = next_q_all.max() if len(next_q_all) > 0 else torch.tensor(0.0, device=self.device)
                q_target_list.append(reward + self.gamma * max_next_q)
        q_target = torch.stack(q_target_list)

        loss = torch.mean((q_current - q_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            self.target_decoder.load_state_dict(self.decoder.state_dict())

        return loss.item()