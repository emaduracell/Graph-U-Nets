import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_mesh_pair(
    pos_true, pos_pred, cells,
    stress_true=None, stress_pred=None,
    node_type_true=None, node_type_pred=None,
    title_true="Ground Truth",
    title_pred="Prediction",
    color_mode="stress"
):
    """
    Visualizzazione mesh + heatmap stress o node_type.
    """

    # ======================================================
    # 1) REMOVE BATCH DIMENSION IF PRESENT
    # ======================================================
    if pos_true.ndim == 3:
        pos_true = pos_true[0]
    if pos_pred.ndim == 3:
        pos_pred = pos_pred[0]

    if stress_true is not None and stress_true.ndim == 2:
        stress_true = stress_true[0]
    if stress_pred is not None and stress_pred.ndim == 2:
        stress_pred = stress_pred[0]

    if node_type_true is not None and node_type_true.ndim == 2:
        node_type_true = node_type_true[0]
    if node_type_pred is not None and node_type_pred.ndim == 2:
        node_type_pred = node_type_pred[0]

    # ======================================================
    # 2) TRIANGOLAZIONE CELLE QUADRILATERE
    # ======================================================
    tri_i, tri_j, tri_k = [], [], []
    for (i0, i1, i2, i3) in cells:
        tri_i.extend([i0, i0])
        tri_j.extend([i1, i2])
        tri_k.extend([i2, i3])

    # ======================================================
    # 3) COLORI / HEATMAP
    # ======================================================

    # --- STRESS MODE ---
    if color_mode == "stress":
        if stress_true is None:
            intensity_true = np.zeros(pos_true.shape[0])
        else:
            intensity_true = stress_true.astype(float)

        if stress_pred is None:
            intensity_pred = np.zeros(pos_pred.shape[0])
        else:
            intensity_pred = stress_pred.astype(float)

        colorscale = "Viridis"

    # --- NODE TYPE MODE ---
    elif color_mode == "node_type":
        if node_type_true is None:
            intensity_true = np.zeros(pos_true.shape[0])
        else:
            intensity_true = node_type_true.astype(float)

        if node_type_pred is None:
            intensity_pred = np.zeros(pos_pred.shape[0])
        else:
            intensity_pred = node_type_pred.astype(float)

        # discrete but continuous scale
        colorscale = "Turbo"

    else:
        raise ValueError("color_mode must be 'stress' or 'node_type'")

    # ======================================================
    # 4) FIGURE SETUP
    # ======================================================
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(title_true, title_pred)
    )

    # ---------------- TRUE MESH ----------------
    fig.add_trace(
        go.Mesh3d(
            x=pos_true[:,0], y=pos_true[:,1], z=pos_true[:,2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=intensity_true,
            colorscale=colorscale,
            showscale=True,
            flatshading=True,
            opacity=1.0,
            name="true_mesh"
        ),
        row=1, col=1
    )

    # ---------------- PRED MESH ----------------
    fig.add_trace(
        go.Mesh3d(
            x=pos_pred[:,0], y=pos_pred[:,1], z=pos_pred[:,2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=intensity_pred,
            colorscale=colorscale,
            showscale=True,
            flatshading=True,
            opacity=1.0,
            name="pred_mesh"
        ),
        row=1, col=2
    )

    # ======================================================
    # 5) SETTINGS
    # ======================================================
    fig.update_scenes(aspectmode="data")
    fig.update_layout(
        height=600,
        width=1200,
        title_text="Mesh Comparison",
        showlegend=False
    )

    fig.show()
