import torch.nn as nn
from model_gunet.helpers_models import top_k_graph, get_adj_norm_fn

class Pool(nn.Module):
    """
    Layer that takes care of pooling, that is, a scalar projection.

    self.k:
        pooling ratio (how many nodes to keep)
    self.sigmoid:
        activation function
    self.proj:
        linear projection (trainable)
    self.drop:
        dropout
    self.adj_norm_fn:
        adjacency matrix normalization function
    """

    def __init__(self, k, in_dim, p, adj_norm):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.adj_norm_fn = get_adj_norm_fn(adj_norm)

    def forward(self, g, h):
        """
        Forward pass of Pooling layer: Dropout layer --> Linear layer learnable (out: scalar score for each node, or
        better for each node's feature vector) --> sigmoid (activation)

        Args:
            g:
                adjacency matrix
            h:
                embedded matrix until here

        :return: top_k_graph(scores, g, h, self.k)
            resulting new prediction/embedded matrix
        """
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()  # learnable
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k, adj_norm_fn=self.adj_norm_fn)