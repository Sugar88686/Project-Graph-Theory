import torch
from torch_geometric.data import Data
import networkx as nx

def nx_to_pyg_data(graph, node_features):
    """
    graph: networkx.Graph
    node_features: numpy array of shape (num_nodes, feat_dim)
    """
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    if edge_index.shape[1] == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    x = torch.FloatTensor(node_features)
    return Data(x=x, edge_index=edge_index)