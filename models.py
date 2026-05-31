import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        virtual_emb = torch.zeros(num_graphs, x.size(-1), dtype=x.dtype, device=x.device)

        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            virtual_emb = self._scatter_mean(x, batch, num_graphs)
            virtual_emb = self.virtual_node_mlp(virtual_emb)
            x = x + virtual_emb[batch]

        node_emb = self.lin(x)
        virtual_emb = self.lin(virtual_emb)
        return node_emb, virtual_emb

    @staticmethod
    def _scatter_mean(x, batch, num_graphs):
        out = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        count = torch.zeros(num_graphs, 1, dtype=x.dtype, device=x.device)
        count.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(x[:, :1]))
        return out / count.clamp(min=1)

class DuelingQDecoder(nn.Module):
    def __init__(self, node_emb_dim, graph_emb_dim, hidden_dim=64):
        super().__init__()
        self.advantage_stream = nn.Sequential(
            nn.Linear(node_emb_dim + graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_emb, graph_emb, valid_mask=None):
        v = self.value_stream(graph_emb)
        graph_emb_expanded = graph_emb.unsqueeze(0).expand(node_emb.size(0), -1)
        combined = torch.cat([node_emb, graph_emb_expanded], dim=-1)
        advantage = self.advantage_stream(combined).squeeze(-1)

        if valid_mask is not None and valid_mask.any():
            adv_mean = advantage[valid_mask].mean()
        else:
            adv_mean = advantage.mean()

        q_vals = v.squeeze(-1) + advantage - adv_mean
        return q_vals

    def forward_batch(self, node_emb, graph_emb_expanded, graph_emb_per_graph, batch_idx):
        combined = torch.cat([node_emb, graph_emb_expanded], dim=-1)
        advantage = self.advantage_stream(combined).squeeze(-1)

        v_per_graph = self.value_stream(graph_emb_per_graph).squeeze(-1)
        v_expanded = v_per_graph[batch_idx]

        num_graphs = graph_emb_per_graph.size(0)
        adv_mean = torch.zeros(num_graphs, dtype=advantage.dtype, device=node_emb.device)
        count = torch.zeros(num_graphs, dtype=advantage.dtype, device=node_emb.device)

        adv_mean.scatter_add_(0, batch_idx, advantage)
        count.scatter_add_(0, batch_idx, torch.ones_like(advantage))
        count = count.clamp(min=1)
        adv_mean = adv_mean / count
        adv_mean_expanded = adv_mean[batch_idx]

        q_vals = v_expanded + advantage - adv_mean_expanded
        return q_vals

def get_graph_data_from_nx(graph):
    data = from_networkx(graph)
    if not hasattr(data, 'edge_index') or data.edge_index is None:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)

    degrees = torch.tensor(
        [graph.degree(n) for n in graph.nodes], dtype=torch.float
    )
    if degrees.numel() > 0 and degrees.max() > 0:
        degrees = degrees / degrees.max()
    data.x = degrees.unsqueeze(1)
    return data
