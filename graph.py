import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from utils.helpers import device


def gumbel_adjacency_matrix(node_embeddings, similarity_threshold, temperature, hard):
    
    node_norm = torch.norm(node_embeddings, p=2, dim=-1, keepdim=True)
    norm_matrix = torch.matmul(node_norm, node_norm.transpose(-2, -1))
    
    similarity_matrix = torch.matmul(node_embeddings, node_embeddings.transpose(-2, -1)) / (norm_matrix + 1e-8)
    similarity_matrix = torch.sigmoid(similarity_matrix)
    sim_matrix_centered = similarity_matrix - similarity_threshold
    
    adjacency_matrix = F.gumbel_softmax(sim_matrix_centered, temperature, hard=hard)
    adjacency_matrix = adjacency_matrix * (1 - torch.eye(node_embeddings.shape[-2], device=device).unsqueeze(0))
    
    return adjacency_matrix


class StateToGraph(nn.Module):

    def __init__(self, state_dim, node_feature_dim, num_nodes):
        super(StateToGraph, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, num_nodes * node_feature_dim)
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim


    def adjacency_to_edge_index(self, adj_matrix):
        
        b, n, _ = adj_matrix.shape
        
        # Create row and column index tensors
        row_indices, col_indices = torch.meshgrid(torch.arange(n), torch.arange(n))
        row_indices = row_indices.to(device).repeat(b, 1, 1)
        col_indices = col_indices.to(device).repeat(b, 1, 1)
        
        # Find non-zero elements in the adjacency matrix
        non_zero_mask = adj_matrix.bool()
        
        # Extract non-zero row and column indices and create edge_index tensor
        edge_index = torch.stack((row_indices[non_zero_mask], col_indices[non_zero_mask]), dim=1)
        
        # Calculate edge count for each (a, b) pair
        edge_counts = non_zero_mask.view(b, n * n).sum(dim=-1)
        edge_indices = torch.split(edge_index.T, edge_counts.tolist(),dim=-1)
        return edge_indices


    def forward(self, x, similarity_threshould, temperature, hard):

        x = torch.relu(self.fc1(x))
        node_emb = x.view(-1, self.num_nodes, self.node_feature_dim)

        adjacency_matrix = gumbel_adjacency_matrix(
            node_emb, 
            similarity_threshold=similarity_threshould, 
            temperature=temperature, 
            hard=hard,
        )
        edge_index = self.adjacency_to_edge_index(adjacency_matrix)
        
        return node_emb, edge_index

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout=0.6):
        super(GAT, self).__init__()
        
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, dropout=dropout)

    def forward(self, g):
        
        x = self.gat1(g.x, g.edge_index)
        x = F.elu(x)
        x = self.gat2(x, g.edge_index)
        
        return x
