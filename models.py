# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx


class GraphSAGEEncoder(nn.Module):
    """图编码器：将节点特征映射到嵌入向量"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.lin(x)
        return x  # [num_nodes, out_dim]


class QDecoder(nn.Module):
    """Q值解码器：结合节点嵌入和图嵌入，输出每个节点的Q值"""

    def __init__(self, node_emb_dim, graph_emb_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_emb_dim + graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_emb, graph_emb):
        """
        node_emb: [num_nodes, node_emb_dim]
        graph_emb: [graph_emb_dim]
        returns: [num_nodes] Q values
        """
        graph_emb_expanded = graph_emb.unsqueeze(0).expand(node_emb.size(0), -1)
        combined = torch.cat([node_emb, graph_emb_expanded], dim=-1)
        q_vals = self.mlp(combined).squeeze(-1)
        return q_vals


def get_graph_data_from_nx(graph, node_feat_func=None):
    """
    将 networkx 图转换为 PyG Data 对象，并生成节点初始特征。
    node_feat_func: 可调用，输入图，输出每个节点的特征向量。
    默认：使用归一化的度作为特征。
    """
    data = from_networkx(graph)
    if node_feat_func is None:
        # 默认特征：度 / 最大度（归一化）
        degrees = torch.tensor([graph.degree(n) for n in graph.nodes], dtype=torch.float)
        if degrees.max() > 0:
            degrees = degrees / degrees.max()
        data.x = degrees.unsqueeze(1)
    else:
        data.x = node_feat_func(graph)
    return data