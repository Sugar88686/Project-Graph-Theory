import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import networkx as nx
from models import GraphSAGEEncoder, DuelingQDecoder
from torch_geometric.data import Batch

class MaxShotAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.encoder = GraphSAGEEncoder(
            config.node_input_dim, config.hidden_dim, config.embedding_dim,
            num_layers=config.graphsage_layers
        ).to(device)
        self.decoder = DuelingQDecoder(
            config.embedding_dim, config.embedding_dim, config.dueling_hidden_dim
        ).to(device)

        self.target_encoder = GraphSAGEEncoder(
            config.node_input_dim, config.hidden_dim, config.embedding_dim,
            num_layers=config.graphsage_layers
        ).to(device)
        self.target_decoder = DuelingQDecoder(
            config.embedding_dim, config.embedding_dim, config.dueling_hidden_dim
        ).to(device)

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_decoder.load_state_dict(self.decoder.state_dict())

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.lr
        )

        self.use_amp = getattr(config, 'use_amp', False) and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps
        self.gamma = config.gamma

        self.update_counter = 0
        self.target_update_freq = config.target_update_freq

    def get_q_values(self, graph, pyg_data, valid_nodes):
        data = pyg_data.clone().to(self.device)

        node_emb, virtual_emb = self.encoder(data.x, data.edge_index)
        graph_emb = virtual_emb.squeeze(0)

        node_list = list(graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        valid_set = set(valid_nodes)
        valid_mask = torch.tensor(
            [n in valid_set for n in node_list], dtype=torch.bool, device=self.device
        )

        q_all = self.decoder(node_emb, graph_emb, valid_mask=valid_mask)

        valid_indices = [node_to_idx[n] for n in valid_nodes if n in node_to_idx]
        if not valid_indices:
            return torch.tensor([]).to(self.device), []
        q_valid = q_all[valid_indices]
        return q_valid, valid_indices

    def act(self, graph, pyg_data, valid_nodes, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(valid_nodes)
        with torch.no_grad():
            q_vals, _ = self.get_q_values(graph, pyg_data, valid_nodes)
            if len(q_vals) == 0:
                return None
            best_local_idx = torch.argmax(q_vals).item()
            return valid_nodes[best_local_idx]

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train_step(self, batch):
        states_nx, states_pyg, actions, rewards, next_states_nx, next_states_pyg, dones = zip(*batch)

        state_batch = Batch.from_data_list(states_pyg).to(self.device)
        next_state_batch = Batch.from_data_list(next_states_pyg).to(self.device)

        def compute_current_q():
            node_emb, virtual_emb = self.encoder(
                state_batch.x, state_batch.edge_index, batch=state_batch.batch
            )
            graph_emb = virtual_emb
            graph_emb_expanded = graph_emb[state_batch.batch]
            q_all = self.decoder.forward_batch(
                node_emb, graph_emb_expanded, graph_emb, state_batch.batch
            )
            return q_all

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                q_all = compute_current_q()
        else:
            q_all = compute_current_q()

        q_current_list = []
        node_offset = 0
        for state, action in zip(states_nx, actions):
            node_list = list(state.nodes)
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            if action in node_to_idx:
                abs_idx = node_offset + node_to_idx[action]
                q_current_list.append(q_all[abs_idx])
            else:
                q_current_list.append(torch.tensor(0.0, device=self.device))
            node_offset += state.number_of_nodes()
        q_current = torch.stack(q_current_list)

        with torch.no_grad():
            def compute_online_next_q():
                node_emb, virtual_emb = self.encoder(
                    next_state_batch.x, next_state_batch.edge_index, batch=next_state_batch.batch
                )
                graph_emb = virtual_emb
                graph_emb_expanded = graph_emb[next_state_batch.batch]
                return self.decoder.forward_batch(
                    node_emb, graph_emb_expanded, graph_emb, next_state_batch.batch
                ), node_emb, virtual_emb

            def compute_target_next_q():
                node_emb, virtual_emb = self.target_encoder(
                    next_state_batch.x, next_state_batch.edge_index, batch=next_state_batch.batch
                )
                graph_emb = virtual_emb
                graph_emb_expanded = graph_emb[next_state_batch.batch]
                return self.target_decoder.forward_batch(
                    node_emb, graph_emb_expanded, graph_emb, next_state_batch.batch
                )

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    online_q_all, _, _ = compute_online_next_q()
                    target_q_all = compute_target_next_q()
            else:
                online_q_all, _, _ = compute_online_next_q()
                target_q_all = compute_target_next_q()

        q_target_list = []
        node_offset = 0
        for next_state, reward, done in zip(next_states_nx, rewards, dones):
            num_nodes = next_state.number_of_nodes()
            if done or num_nodes == 0:
                q_target_list.append(
                    torch.tensor(reward, dtype=torch.float, device=self.device)
                )
            else:
                start_idx = node_offset
                end_idx = node_offset + num_nodes

                graph_online_q = online_q_all[start_idx:end_idx]
                graph_target_q = target_q_all[start_idx:end_idx]

                components = list(nx.connected_components(next_state))
                gcc_nodes = set(max(components, key=len)) if components else set()
                node_list = list(next_state.nodes)
                valid_mask = torch.tensor(
                    [n in gcc_nodes for n in node_list],
                    dtype=torch.bool, device=self.device
                )

                masked_online_q = graph_online_q.clone()
                if valid_mask.any():
                    masked_online_q[~valid_mask] = -float('inf')
                best_action_idx = masked_online_q.argmax()
                max_next_q = graph_target_q[best_action_idx]

                q_target_list.append(
                    torch.tensor(reward, dtype=torch.float, device=self.device)
                    + self.gamma * max_next_q
                )
            node_offset += num_nodes

        q_target = torch.stack(q_target_list)

        # ==========================================
        # 🌟 核心修复：使用 Huber Loss (Smooth L1 Loss) 替代 MSE
        # ==========================================
        loss = F.smooth_l1_loss(q_current, q_target.detach())

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            self.target_decoder.load_state_dict(self.decoder.state_dict())

        return loss.item()
