import torch
import numpy as np
import json
import os

from NSM.datasets import MultiSurfaceSDFSamples
from NSM.models import TriplanarDecoder
from NSM.train.train_deep_sdf_multi_surface import train_deep_sdf as train_deep_sdf_multi_surface

CACHE = True
USE_WANDB = True
N_TRAIN = 50
N_VAL = 10

PROJECT_NAME = 'ShapeMedKnee'
ENTITY_NAME = 'bone-modeling'
RUN_NAME = 'test'
LOC_SDF_CACHE = 'cache'
LOC_SAVE_NEW_MODELS = 'models'

if (USE_WANDB is True) and ('WANDB_KEY' not in os.environ):
    raise ValueError('WANDB_KEY is not in the environment variables. Please set it or set USE_WANDB to False.')

if CACHE is True:
    if not os.path.exists(LOC_SDF_CACHE):
        os.makedirs(LOC_SDF_CACHE)
    LOC_SDF_CACHE = os.path.abspath(LOC_SDF_CACHE)
    os.environ['LOC_SDF_CACHE'] = LOC_SDF_CACHE


path_config = 'ShapeMedKnee_2024_config.json'
with open(path_config, 'r') as f:
    config = json.load(f)

if USE_WANDB is True:
    config['project_name'] = PROJECT_NAME
    config['entity_name'] = ENTITY_NAME
    config['entity'] = ENTITY_NAME
    config['run_name'] = RUN_NAME

config['experiment_directory'] = os.path.abspath(LOC_SAVE_NEW_MODELS)


# get full list of training data from config
list_mesh_paths_ = config["list_mesh_paths"]
# get full list of validation data from config
list_val_paths_ = config["val_paths"]
# paths to where the data is. 
folder_training_data = 'dataset/meshes/train/subfolder_0'
folder_val_data = 'dataset/meshes/val/subfolder_0'

# iterate over all of the training/val lists, if the file exists in the folder
# then add the absolute path to the list_mesh_paths & list_val_paths
list_mesh_paths = []
for bone_mesh_path, cart_mesh_path in list_mesh_paths_:
    bone_path = os.path.abspath(os.path.join(folder_training_data, bone_mesh_path))
    cart_path = os.path.abspath(os.path.join(folder_training_data, cart_mesh_path))

    bone_exists = os.path.exists(bone_path)
    cart_exists = os.path.exists(cart_path)

    if bone_exists and cart_exists:
        list_mesh_paths.append([bone_path, cart_path])
    else:
        pass

list_val_paths = []
for bone_mesh_path, cart_mesh_path in list_val_paths_:
    bone_path = os.path.join(folder_val_data, bone_mesh_path)
    cart_path = os.path.join(folder_val_data, cart_mesh_path)

    bone_exists = os.path.exists(bone_path)
    cart_exists = os.path.exists(cart_path)

    if bone_exists and cart_exists:
        list_val_paths.append([bone_path, cart_path])
    else:
        pass

list_mesh_paths = list_mesh_paths[:N_TRAIN]
list_val_paths = list_val_paths[:N_VAL]

config['val_paths'] = list_val_paths

# Set the seed value!
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

sdf_dataset = MultiSurfaceSDFSamples(
    list_mesh_paths=list_mesh_paths,
    subsample=config["samples_per_object_per_batch"],
    print_filename=True,
    n_pts=config["n_pts_per_object"],
    p_near_surface=config['percent_near_surface'],
    p_further_from_surface=config['percent_further_from_surface'],
    sigma_near=config['sigma_near'],
    sigma_far=config['sigma_far'],
    rand_function=config['random_function'], 
    center_pts=config['center_pts'],
    scale_all_meshes=config['scale_all_meshes'],
    center_all_meshes=config['center_all_meshes'],
    mesh_to_scale=config['mesh_to_scale'],
    norm_pts=config['normalize_pts'],
    scale_method=config['scale_method'],
    scale_jointly=config['scale_jointly'],
    random_seed=config['seed'],
    reference_mesh=config['reference_mesh'],
    verbose=config['verbose'],
    save_cache=config['cache'],
    equal_pos_neg=config['equal_pos_neg'],
    fix_mesh=config['fix_mesh'],
    load_cache=config['load_cache'],
    store_data_in_memory=config['store_data_in_memory'],
    multiprocessing=config['multiprocessing'],
    n_processes=config['n_processes'],
)
print('sdf_dataset:', sdf_dataset)
print('len sdf_dataaset', len(sdf_dataset))

triplane_args = {
    'latent_dim': config['latent_size'],
    'n_objects': config['objects_per_decoder'],
    'conv_hidden_dims': config['conv_hidden_dims'],
    'conv_deep_image_size': config['conv_deep_image_size'],
    'conv_norm': config['conv_norm'], 
    'conv_norm_type': config['conv_norm_type'],
    'conv_start_with_mlp': config['conv_start_with_mlp'],
    'sdf_latent_size': config['sdf_latent_size'],
    'sdf_hidden_dims': config['sdf_hidden_dims'],
    'sdf_weight_norm': config['weight_norm'],
    'sdf_final_activation': config['final_activation'],
    'sdf_activation': config['activation'],
    'sdf_dropout_prob': config['dropout_prob'],
    'sum_sdf_features': config['sum_conv_output_features'],
    'conv_pred_sdf': config['conv_pred_sdf'],
}

model = TriplanarDecoder(**triplane_args)

train_deep_sdf_multi_surface(
    config=config,
    model=model,
    sdf_dataset=sdf_dataset,
    use_wandb=True,
)