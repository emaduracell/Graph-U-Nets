## Graph U-Nets

#### How to run
0. Create a `raw_data` folder in `pytorch_model` with test.tfrecord, train.tfrecord, valid.tfrecord, meta.json files. 
  So for example `pytorch_model/raw_data/train.tfrecord`.

The following commands must be ran with `pytorch_model` as working directory.
1. Run `python main_data.py`: you can set it via `dataconfig.yaml`.
2. Run `python train.py`: you can set it via `config.yaml`.
3. Run `python visualize_simulation.py`: you can set it via the following parameters.
```python
traj_idx = 0 # Trajectory to plot
t_step = 10  # Start time idx
rollout_set = True  # if True, run multi-step rollout
rollout_steps = 10  # maximum number of rollout steps 
render_mode = "all"  # options: "all", "no_border", "no_sphere", "no_border_no_sphere"
# Here assumes the data and the trained model are from the `config.yaml`. Since they are only strings, change it 
# accordingly if you don't want this behaviour
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = load_config(config_path)
preprocessed_path = config['training']['datapath']
add_world_edges = config['training']['add_world_edges']
checkpoint_path = ((config['training']['model_path'] + "model_" + preprocessed_path.rsplit("/", 1)[0]) + "_" +
                   add_world_edges)
```
###### Dataconfig.yaml
NOTE: the data is actually saved in the folder, which is automatatically created:
```python
output_dir = output_dir + "_" + norm_method + "_" + include_mesh_pos"
```
to be able touse different datasets with different options. Now the yaml file:
```python
data:
  include_mesh_pos: True # Include mesh position in the features or not
  normalization_method: "standard" # 'standard' or 'centroid'
  max_trajs: 1500 # (upper threshold for) number of trajectories to load
  tfrecord_path: "raw_data/train.tfrecord" 
  meta_path: "raw_data/meta.json"
  output_dir: "data" # Where you want the processed data.
```
###### Config.yaml
NOTE:
- `datapath: "data_standard_True/preprocessed_train.pt"` specifies the desired dataset to train on.
- `add_world_edges: "radius"` specifies if we should add world edges and with which method or not.
The model is saved in the directory specified in model_path: `"model_out/"` with name of the file being`"model_" + preprocessed_data_path.rsplit("/", 1)[0] + "_" +
add_world_edges)` (see `checkpoint_path` var in `train.py`)
An example of how we could train the model in a normal way
```python
# Model hyperparameters
model:
  activation_gnn: "ReLU"
  activation_mlps_final: "ReLU"
  hid_gnn_layer_dim: 128
  hid_mlp_dim: 256
  k_pool_ratios: [0.9, 0.8, 0.7]
  dropout_gnn: 0.05 # default = 0.1, overfit = 0
  dropout_mlps_final: 0.05 # default = 0.1, overfit = 0

# Training parameters TODO add loss as a hyperparameter
training:
  lr: 0.0001 # default = 1e-4 good value 0.001
  epochs: 500
  batch_size: 8
  shuffle: True # False for overfitting?
  adam_weight_decay: 0.0001 # default: 1e-4 | overfit = 0
  num_train_trajs: 8
  mode: "None" # None or "overfit"
  gamma_lr_scheduler: 0.9998 # lr_final = lr0 * gamma^epochs 0.9995 good value 0.9997
  random_seed: 42 # Seed for train/test split and any random operations
  overfit_traj_id: 0 # For overfit mode: which trajectory to use (null = use first available)
  overfit_time_idx: [0,1,2,3,4,5,6,7] # For overfit mode: which time step t to use (null = use all time steps from trajectory) [13, 202, 302, 10, 11, 90, 321, 43]
  add_world_edges: "radius" # 'neighbours' or 'radius' or 'None'
  datapath: "data_standard_True/preprocessed_train.pt"
  radius_world_edge: 0.03
  k_neighb: 1
  model_path: "model_out/"
```
For seeing if the model overfits, we could use
```python
# Model hyperparameters
model:
  activation_gnn: "ReLU"
  activation_mlps_final: "ReLU"
  hid_gnn_layer_dim: 128
  hid_mlp_dim: 256
  k_pool_ratios: [0.9, 0.8, 0.7]
  dropout_gnn: 0 # default = 0.1, overfit = 0
  dropout_mlps_final: 0 # default = 0.1, overfit = 0

# Training parameters 
training:
  lr: 0.0001 # default = 1e-4 good value 0.001 for starting fast
  epochs: 1000
  batch_size: 8
  shuffle: False # False for overfitting
  adam_weight_decay: 0 # default: 1e-4 | overfit = 0
  num_train_trajs: 8
  mode: "overfit" # None or "overfit"
  gamma_lr_scheduler: 1 # or 1
  random_seed: 42 # Seed for train/test split and any random operations
  overfit_traj_id: 0 # For overfit mode: which trajectory to use (null = use first available)
  overfit_time_idx: [0,1,2,3,4,5,6,7] # For overfit mode: which time step t to use (null = use all time steps from trajectory) [13, 202, 302, 10, 11, 90, 321, 43]
  add_world_edges: "radius" # 'neighbours' or 'radius' or 'None'
  datapath: "data_standard_True/preprocessed_train.pt" # NOTE: DATA YOU WANT TO USE
  radius_world_edge: 0.03
  k_neighb: 1
  model_path: "model_out/"
```
#### Structure
###### Data generation
- `data_loader.py`:
  - generates normalized couples $(t, t+1)$ used for training. 
  - Appending the next step velocity feature for only the sphere (actuator) is NOT done here but
   it's done in the training phase and it's normalized already via mean and stddev of the whole dataset.
  - normalization is done with two techniques:
    - `"standard"`: usual zero mean unit variance normalization with stddev and mean. We have to note, though 
    meshgraphnet paper uses it, they do not feed absolute mesh and world positions.
    - `"centroid"` one that should take better account for translation invariance (not checked).
  - Edges are build via `load_edges_from_cells` and are tetrahedra.
  - NOTE: the data is actually saved in the folder, which is automatatically created:
    `output_dir = output_dir + "_" + norm_method + "_" + include_mesh_pos"`
- `main_data.py`: uses `data_loader.py` to save via function `preprocess_and_save` data and metadata (e.g. mean stddev 
to denormalize) in as torch files.
  - NOTE: all the data are stored in RAM currently (better also for SCITAS) because it apparently is fine for our PCs.
- `data_exploration.py`: script to render the data as is.
###### Train and evaluation
- `train.py`: trains the model
  - `train_gnet`: actual training loop, with lr scheduler, Adam optimizer, huber loss for stress and mse for velocity.
  - `compute_loss`: computes loss of current batch
  - `compute_batch_metrics`: computes MAE
  - `get_grad_norm.py`: computes gradient norm
  - Model is saved here
```python
checkpoint_path = (train_cfg['model_path'] + "model_" + preprocessed_data_path.rsplit("/", 1)[0] + "_" +
                       add_world_edges)
```
- `defplate_datset.py`: Defines the dataset object to use in train and its get item function.
  - Computes world edges via `add_w_edges_neigh` or `add_w_edges_radius`. 
  - Takes care of adding the next step velocity for only the actuator.
- `plots.py`: plots residuals (normalized and not) of pred - ground truth vs ground truth, prediction vs ground truth 
(normalized), gradient norm vs epochs, overall loss vs epochs, loss for stress and for velocity vs epochs.
  - Plots are saved here:
```python
plots_dir = os.path.join(os.path.dirname(__file__), train_cfg['model_path'] + "plots_" +preprocessed_data_path.rsplit("/", 1)[0] + "_" + add_world_edges)
```
###### Model
- `model_entire.py`: creates an instance of graph_unet_layers and then puts two MLP heads at the end, one for stress and
the other one for velocity.
- `graph_unet_layers.py`: graph unet layers e.g. GCNs and Poolings.
###### Rollout
- `visualize_simulation.py`: implements rollout both visually and plotting the error.