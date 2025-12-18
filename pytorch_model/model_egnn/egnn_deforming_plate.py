import torch
import torch.nn as nn
from .egnn import EGNN_Network


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EGNN_DefPlate(nn.Module):
    """
    EGNN model with heads MLPs for stress and velocity prediction.
    """

    def __init__(self, in_dim, vel_out_dim, stress_out_dim, model_config_hyperparams, adj_norm):
        """
        Args:
            in_dim: Input node feature dimension.
            vel_out_dim: Output dimension for velocity.
            stress_out_dim: Output dimension for stress.
            model_config_hyperparams: argparse.Namespace (requires .hid_gnn_layer_dim)
        """
        super().__init__()

        # Hyperparams
        hidden_dim = model_config_hyperparams.hid_gnn_layer_dim

        if hasattr(model_config_hyperparams, 'num_layers'):
            depth = model_config_hyperparams.num_layers
        elif hasattr(model_config_hyperparams, 'depth'):
            depth = model_config_hyperparams.depth
        elif hasattr(model_config_hyperparams, 'k_pool_ratios'):
            depth = len(model_config_hyperparams.k_pool_ratios) + 1
        else:
            depth = 4

        # Cap neighbors to prevent dense N^2 blowups in this egnn implementation
        self.max_neighbors = int(getattr(model_config_hyperparams, "max_neighbors", 32))
        # Coordinate slicing heuristic. If in_dim > 12, coords are likely at [3:6] (MeshPos included), else at [0:3]
        self.pos_slice = slice(3, 6) if in_dim > 12 else slice(0, 3)
        feat_dim = in_dim - 3  # remove 3 coordinate channels from features

        # Architecture
        self.input_mlp = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                                       SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       SiLU(),
                                       nn.Linear(hidden_dim, hidden_dim))

        self.egnn_layers = EGNN_Network(num_tokens=None, dim=hidden_dim, depth=depth, edge_dim=0,
                                        only_sparse_neighbors=True, update_coors=False, update_feats=True)

        # Physics Heads
        self.phi_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   SiLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   SiLU(),
                                   nn.Linear(hidden_dim, vel_out_dim))

        edge_input_dim = 2 * hidden_dim + 1
        self.phi_e_proj = nn.Sequential(nn.Linear(edge_input_dim, hidden_dim), SiLU())
        self.phi_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   SiLU(),
                                   nn.Linear(hidden_dim,
                                             1))

        # Learnable parameter for residual connection
        self.C = nn.Parameter(torch.tensor(-1.0))
        self.register_buffer('_c_initialized', torch.tensor(False))

        # Stress Head
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, hidden_dim), SiLU(),
            nn.Linear(hidden_dim, stress_out_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m == self.stress_head[-1]:
                        nn.init.constant_(m.bias, -0.8)
                    else:
                        nn.init.zeros_(m.bias)

    def forward(self, batch_adj_A, batch_feat_X, feat_tp1_mat_list=None, node_type=None, **kwargs):
        """
        Accepts optional args to be safe, but primarily uses A and X.

        Args:
            batch_adj_A:
                adjacency matrix
            batch_feat_X:
                feature matrix
            feat_tp1_mat_list:
                t+1 feature matrix
            node_type:
                node type vector

        :return:
        """
        return self.embed(batch_adj_A, batch_feat_X)

    def rollout_step(self, A, X_t):
        """
        Single-step prediction (no loss), for rollouts.

        Args:
            A: adjacency matrix
            X_t: feature matrix
        :return: prediction
        """
        return self.embed(A, X_t)

    def embed(self, adj_A_list, X_list):
        """
        Process a batch of graphs.

        Args:
            adj_A_list: adjacency matrix list
            X_list: feature matrix list
        :return:
        """
        # Single-graph case: tensors
        if isinstance(adj_A_list, torch.Tensor):
            return self.embed_one(adj_A_list, X_list)
        # Batch case: list of tensors
        return [self.embed_one(A, X) for A, X in zip(adj_A_list, X_list)]

    @torch.no_grad()
    def _sparsify_adj_bool(self, adj, max_neighbors):
        """
        Convert adj to boolean adjacency and cap outgoing neighbors per node.

        Args:
            adj: float/bool adjacency
            max_neighbors: int, cap per row (<=0 means keep all)

        Returns:
            adj_bool_sparse: bool, no self-loops, <= max_neighbors True per row
        """
        N = adj.size(0)
        device = adj.device

        adj_bool = (adj > 0)
        adj_bool.fill_diagonal_(False)

        if max_neighbors is None or max_neighbors <= 0:
            return adj_bool

        out = torch.zeros((N, N), device=device, dtype=torch.bool)
        for i in range(N):
            nbrs = adj_bool[i].nonzero(as_tuple=False).flatten()
            if nbrs.numel() == 0:
                continue
            if nbrs.numel() > max_neighbors:
                nbrs = nbrs[:max_neighbors]  # deterministic cap (can randomize if desired)
            out[i, nbrs] = True
        return out

    def embed_one(self, adj, x):
        """
        Process a single graph.
        This EGNN fork expects adj_mat as a boolean mask and derives neighborhoods internally.
        """
        # Lazy initialization of C: set to 1/(N-1) on first forward if not already initialized
        N = x.shape[0]
        if not self._c_initialized.item():
            # Check if C was loaded from checkpoint (not sentinel value)
            if self.C.item() < 0:
                # Initialize C = 1/(N-1), with fallback for edge cases
                c_init = 1.0 / (N - 1) if N > 1 else 1e-3
                with torch.no_grad():
                    self.C.data.fill_(c_init)
            # Mark as initialized regardless (either from checkpoint or just initialized)
            self._c_initialized.fill_(True)

        pos = x[:, self.pos_slice]

        # Concatenate features excluding coords
        h_in = torch.cat([x[:, :self.pos_slice.start], x[:, self.pos_slice.stop:]], dim=1)

        # Batch dimension required by library [1, N, C]
        h_b = h_in.unsqueeze(0)  # [1, N, feat_dim]
        pos_b = pos.unsqueeze(0)  # [1, N, 3]

        # Initial Embedding
        h_emb = self.input_mlp(h_b)

        # bool adjacency with capped neighbors to prevent dense blowups
        adj_sparse_bool = self._sparsify_adj_bool(adj, self.max_neighbors)
        adj_b = adj_sparse_bool.unsqueeze(0)

        # EGNN Layers
        h_updated, _ = self.egnn_layers(h_emb, pos_b, adj_mat=adj_b)
        v_direct = self.phi_v(h_updated)

        edge_indices = adj_sparse_bool.nonzero(as_tuple=False).t()  # [2, E]

        if edge_indices.numel() > 0:
            i_idx, j_idx = edge_indices[0], edge_indices[1]

            pos0 = pos_b.squeeze(0)
            rel_pos_edges = pos0[i_idx] - pos0[j_idx]
            dist_sq_edges = (rel_pos_edges ** 2).sum(-1, keepdim=True)

            h0 = h_updated.squeeze(0)
            h_i = h0[i_idx]
            h_j = h0[j_idx]

            edge_feat = torch.cat([h_i, h_j, dist_sq_edges], dim=-1)
            weights_edges = self.phi_x(self.phi_e_proj(edge_feat)).squeeze(-1)

            neighbor_term = torch.zeros_like(v_direct.squeeze(0))
            src = rel_pos_edges * weights_edges.unsqueeze(-1)
            neighbor_term.index_add_(0, i_idx, src.to(dtype=neighbor_term.dtype))
        else:
            neighbor_term = torch.zeros_like(v_direct.squeeze(0))

        neighbor_term = neighbor_term.unsqueeze(0)

        # Final predictions
        pred_vel = v_direct + self.C * neighbor_term
        pred_stress = self.stress_head(h_updated)

        return torch.cat([pred_vel, pred_stress], dim=-1).squeeze(0)
