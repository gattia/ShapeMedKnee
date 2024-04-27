import torch
import json
import os
from pymskt.mesh import BoneMesh

from NSM.models import TriplanarDecoder
from NSM.reconstruct import reconstruct_mesh


# Setup file paths - relative to this script
path_model_config = '231_nsm_femur_cartilage_v0.0.1/model_config.json'
path_model_state = '231_nsm_femur_cartilage_v0.0.1/model/2000.pth'
manually_created_meshes = False
loc_save_recons = 'example_recon/recons'
print_recon_error_metrics = True 

if manually_created_meshes:
    path_meshes = [
        'example_recon/segs/flipped_9000099_LEFT_femur.vtk'
        'example_recon/segs/flipped_9000099_LEFT_fem_cart.vtk'
    ]
else:
    path_meshes = [
        'example_recon/meshes/test/subfolder_0/9000099_LEFT_femur.vtk',
        'example_recon/meshes/test/subfolder_0/9000099_LEFT_fem_cart.vtk'
    ]

# Load config
with open(path_model_config, 'r') as f:
    config = json.load(f)

# define params as needed to be input into network 
params = {
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

# build the model 
model = TriplanarDecoder(**params)
saved_model_state = torch.load(path_model_state)
model.load_state_dict(saved_model_state["model"])
model = model.cuda()
model.eval()


# Reconstruct the meshes (bone/cartilage) using the NSM model
mesh_result = reconstruct_mesh(
    path=path_meshes,
    decoders=model,
    latent_size=config['latent_size'],
    # Fitting parameters:
    num_iterations=config['num_iterations_recon'],
    l2reg=config['l2reg_recon'],
    latent_reg_weight=config['l2reg_recon'],
    loss_type='l1',
    lr=config['lr_recon'],
    lr_update_factor=config['lr_update_factor_recon'],
    n_lr_updates=config['n_lr_updates_recon'],
    return_latent=True,
    register_similarity=True,
    scale_jointly=config['scale_jointly'],
    scale_all_meshes=True,
    objects_per_decoder=2,
    batch_size_latent_recon=config['batch_size_latent_recon'],
    get_rand_pts=config['get_rand_pts_recon'],
    n_pts_random=config['n_pts_random_recon'],
    sigma_rand_pts=config['sigma_rand_pts_recon'],
    n_samples_latent_recon=config['n_samples_latent_recon'], 

    calc_assd=print_recon_error_metrics,
    
    convergence=config['convergence_type_recon'], 
    convergence_patience=config['convergence_patience_recon'],
    clamp_dist=config['clamp_dist_recon'],

    fix_mesh=config['fix_mesh_recon'],
    verbose=True,
    return_registration_params=True,
)

# get meshes from results, convert bone to a "BoneMesh" so cartilage thickness can be calculated
bone_mesh = BoneMesh(mesh_result['mesh'][0].mesh)
cart_mesh = mesh_result['mesh'][1]

# compute cartilage thickness for the bone mesh
bone_mesh.calc_cartilage_thickness(list_cartilage_meshes=[cart_mesh])

# get latent - we're not doing anything with this now, but it can be used for downstream predictions
latent = mesh_result['latent'].detach().cpu().numpy().tolist()

# print reconstruction metrics
if print_recon_error_metrics:
    print('Reconstruction metrics:')
    print(f'ASSD bone: {mesh_result["assd_0"]:.2f}mm')
    print(f'ASSD cartilage: {mesh_result["assd_1"]:.2f}mm')

# save the reconstructed meshes 
if os.path.exists(loc_save_recons) == False:
    os.makedirs(loc_save_recons, exist_ok=True)

bone_mesh.save_mesh(os.path.join(loc_save_recons, f'recon_{os.path.basename(path_meshes[0])}'))
cart_mesh.save_mesh(os.path.join(loc_save_recons, f'recon_{os.path.basename(path_meshes[1])}'))

# if interested - the registration params used to align the mesh with the NSM reference mesh (latent vector = 0,0,0...)
# these can be seen in mesh_result['icp_trasform'], mesh_result['center'], mesh_result['scale']