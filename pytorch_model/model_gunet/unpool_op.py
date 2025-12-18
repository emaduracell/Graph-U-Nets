import torch.nn as nn

class Unpool(nn.Module):
    """
    Unpooling class, not learnable.
    """

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        """
        Args:
            g:
                original graph adjacency matrix (skip connection)
            h:
                input (encoded) embedded feature matrix
            pre_h:
                NOT IMPLEMENTED
            idx:
                indexes of previously kept nodes by top-k.

        :return: (g, new_h)
            g: original graph adjacency matrix (skip connection)
            new_h: new feature matrix
        """
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h