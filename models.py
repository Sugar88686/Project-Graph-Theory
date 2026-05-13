import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    """将图结构编码为固定长度向量"""
    def __init__(self, in_channels=1, hidden_dim=32, out_dim=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # 全局平均池化得到图嵌入
        return x.mean(dim=0)   # (out_dim,)

class DQN(nn.Module):
    def __init__(self, num_nodes, embed_dim=16, global_dim=1, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        input_dim = embed_dim + global_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, graph_emb, global_feat):
        # graph_emb: (batch, embed_dim)
        # global_feat: (batch, 1)
        x = torch.cat([graph_emb, global_feat], dim=1)
        q = self.net(x)   # (batch, num_nodes)
        return q