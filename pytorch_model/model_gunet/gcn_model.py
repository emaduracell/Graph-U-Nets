import torch.nn as nn
from model_gunet.helpers_models import get_adj_norm_fn
import torch

class GCN(nn.Module):
    """
    Graph Convolutional Network: takes care of message passing (aggregating node neighborhoods).

    self.proj:
        linear layer (trainable)
    self.act:
        activation function
    self.drop:
        dropout
    self.adj_norm_fn:
        adjacency matrix normalization function
    """

    def __init__(self, in_dim, out_dim, act, p, adj_norm):
        super(GCN, self).__init__()
        self.adj_norm_fn = get_adj_norm_fn(adj_norm)
        self.proj = nn.Linear(in_dim, out_dim)  # learnable
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        """
        Forward pass in GCN. Dropout --> Convolution/Aggregation (matmul) --> learnable linear layer --> activation

        :param g:
            adjacency matrix
        :param h:
            embedded matrix until here

        :return: h
            resulting new prediction/embedded matrix
        """
        h = self.drop(h)
        h = torch.matmul(g, h)  # convolution step
        h = self.proj(h)  # learnable
        h = self.act(h)
        return h