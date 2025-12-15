import torch
import torch_geometric.transforms as T
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import coalesce

# Constants (Must match your data)
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1

class InjectKinematicVelocity(T.BaseTransform):
    """
    SPARSE OPERATION: Node-wise Parallel
    Appends the FUTURE velocity of SPHERE nodes to the input features.
    This gives the model 'clairvoyance' about where the tool is moving.
    """
    def __init__(self, velocity_idxs=slice(0, 3)):
        self.velocity_idxs = velocity_idxs

    def forward(self, data):
        # 1. Identify Sphere Nodes (Boolean Mask)
        # We assume node_type is shape [N] or [N, 1]
        sphere_mask = (data.node_type == SPHERE_NODE).squeeze()
        
        # 2. Extract Target Velocity (v_tp1) from data.y
        # data.y is the target for the next step.
        v_tp1 = data.y[:, self.velocity_idxs]
        
        # 3. Create Kinematic Feature Vector
        # Initialize with Zeros (Sparse concept: most nodes have 0 kinematic vel)
        kinematic_vel = torch.zeros_like(v_tp1)
        
        # Only fill in values for the sphere
        if sphere_mask.sum() > 0:
            kinematic_vel[sphere_mask] = v_tp1[sphere_mask]
            
        # 4. Concatenate to Input Features (x)
        # New Feature Dim = Old Dim + 3
        data.x = torch.cat([data.x, kinematic_vel], dim=-1)
        
        return data

class AddDynamicWorldEdges(T.BaseTransform):
    """
    SPARSE OPERATION: Spatial Hashing / K-NN
    Dynamically adds edges between nodes based on spatial proximity.
    """
    def __init__(self, mode='radius', radius=0.05, k=5, world_pos_idxs=slice(0, 3)):
        assert mode in ['radius', 'neighbours', 'None']
        self.mode = mode
        self.radius = radius
        self.k = k
        self.pos_idxs = world_pos_idxs

    def forward(self, data):
        if self.mode == 'None':
            return data
            
        # Extract positions
        pos = data.x[:, self.pos_idxs]
        
        # PyG handles batching automatically via data.batch
        batch = data.batch if 'batch' in data else None
        
        new_edges = None
        
        # --- STRATEGY 1: Radius Graph (The "Sparse on Sparse" Logic) ---
        if self.mode == 'radius':
            # This function DOES NOT compute a distance matrix.
            # It uses grid-cluster search (on CPU) or parallel primitive (on GPU)
            # to find pairs < radius efficiently.
            new_edges = radius_graph(pos, r=self.radius, batch=batch, loop=False)
            
        # --- STRATEGY 2: K-Nearest Neighbors ---
        elif self.mode == 'neighbours':
            # We specifically need Sphere <-> Plate connections.
            node_type = data.node_type.squeeze()
            
            # Compute KNN for the whole graph (Optimized sparse search)
            edge_index_knn = knn_graph(pos, k=self.k, batch=batch, loop=False)
            
            src, dst = edge_index_knn
            src_type = node_type[src]
            dst_type = node_type[dst]
            
            # Filter: Only keep edges connecting Sphere to Non-Sphere
            # This filtering is fast because we are filtering a list of edges (E),
            # not a matrix of nodes (N^2).
            is_sphere_src = (src_type == SPHERE_NODE)
            is_plate_dst  = (dst_type == NORMAL_NODE) | (dst_type == BOUNDARY_NODE)
            
            is_plate_src  = (src_type == NORMAL_NODE) | (src_type == BOUNDARY_NODE)
            is_sphere_dst = (dst_type == SPHERE_NODE)
            
            valid_mask = (is_sphere_src & is_plate_dst) | (is_plate_src & is_sphere_dst)
            new_edges = edge_index_knn[:, valid_mask]

        if new_edges is not None and new_edges.numel() > 0:
            # Add new edges to existing mesh edges
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
            
            # coalese() merges duplicate edges and sorts indices.
            data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
            
        return data