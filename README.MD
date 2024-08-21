# ShapeMedKnee - Examples 

The model is located at: 
    https://huggingface.co/aagatti/ShapeMedKnee/

The data is located at: 
    https://huggingface.co/datasets/aagatti/ShapeMedKnee/

The NSM code for running the model is located at: 
    https://github.com/gattia/nsm

The medRxiv paper is located at: 
    https://www.medrxiv.org/content/10.1101/2024.05.06.24306965v1

## Instructions to download data, download model, and make predictions. 

1. Install NSM 
Follow instruction on repository (https://github.com/gattia/nsm) or see below. Note
the instructions below use conda. But if you have mamba installed, it will be MUCH 
faster. 

```bash
git clone https://github.com/gattia/nsm.git
cd nsm
conda create -n NSM python=3.8
conda activate NSM
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install .
```

2. Download model
Install huggingface API, if not alread. 
`pip install huggingface-hub`

Login to huggingface
`huggingface-cli login`
Then you will be prompted for an access token, whicih you can get from the settings tab in your huggingface account. 

Download the model: 
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="aagatti/ShapeMedKnee", local_dir='./')

```

3. Download data: 
Get one example, show how to convert segmentation into 3D model (surface) using pymskt. 
Show how to use mesh
    - Surface
    - Segmentation 

```bash
mkdir example_recon    
```

```python
from huggingface_hub import hf_hub_download

# download reference cartilage surface (test set)
hf_hub_download(
    repo_id="aagatti/ShapeMedKnee", 
    filename="9000099_LEFT_fem_cart.vtk", 
    subfolder='meshes/test/subfolder_0',
    local_dir='example_recon',
    repo_type='dataset'
)

# download reference bone surface (test set)
hf_hub_download(
    repo_id="aagatti/ShapeMedKnee",
    filename="9000099_LEFT_femur.vtk",
    subfolder='meshes/test/subfolder_0',
    local_dir='example_recon',
    repo_type='dataset'
)

# Download segmentation (test set)
hf_hub_download(
    repo_id="aagatti/ShapeMedKnee",
    filename="9000099_LEFT-label.nii.gz",
    subfolder='segs',
    local_dir='example_recon',
    repo_type='dataset'
)

```

4. Run reconstruction on data to get: 

If you want to start from a segmentation, then use the following to create a surface from the the segmentation.
Note, we are using a left knee example - these need a little extra care. The NSM was fit to only right knees
This is to remove variance due to anatomical side. We can in theory fit the NSM to the left knee and then 
just swap everything back as necessary. Below provides 2 options for how to create the mesh. 

The below two meshes will give slightly differently positioned meshes, but the overall result should be the
same from an NSM fitting perspective. The second version is likely easier to undo later for multiple meshes. 
E.g., it could be applied to the femur, tibia, and patella and then undone in the opposite way it was done. 
The image way could also be undone, but not as easily. 


### Option 1: Flip the image

```python
import os
import SimpleITK as sitk
from pymskt.mesh import Mesh, BoneMesh

subject_id = '9000099'
side = 'LEFT'

path_seg = f'example_recon/segs/{subject_id}_{side}-label.nii.gz'

seg_ = sitk.ReadImage(path_seg)
array = sitk.GetArrayFromImage(seg_)
array = array[::-1, :, :]
seg = sitk.GetImageFromArray(array)
seg.CopyInformation(seg_)
flipped_seg_name = f'flipped_{os.path.basename(path_seg)}'
flipped_seg_path = os.path.join(os.path.dirname(path_seg), flipped_seg_name)
sitk.WriteImage(seg, flipped_seg_path)

femur = BoneMesh(path_seg_image=flipped_seg_path, label_idx=1, crop_percent=0.8, bone='femur')
fem_cart = Mesh(path_seg_image=flipped_seg_path, label_idx=2)

femur.create_mesh(smooth_image_var=0.5)
fem_cart.create_mesh(smooth_image_var=0.3125/2)

femur.save_mesh(os.path.join(os.path.dirname(path_seg), f'flipped_{subject_id}_{side}_femur.vtk'))
fem_cart.save_mesh(os.path.join(os.path.dirname(path_seg), f'flipped_{subject_id}_{side}_fem_cart.vtk'))
```

### Option 2: Flip the mesh 
```python
import os
import numpy as np
from pymskt.mesh import Mesh, BoneMesh

subject_id = '9000099'
side = 'LEFT'

path_seg = f'example_recon/segs/{subject_id}_{side}-label.nii.gz' 

femur = BoneMesh(path_seg_image=path_seg, label_idx=1, crop_percent=0.8, bone='femur')
fem_cart = Mesh(path_seg_image=path_seg, label_idx=2)

femur.create_mesh(smooth_image_var=0.5)
fem_cart.create_mesh(smooth_image_var=0.3125/2)

# now, these are from a left knee, but the model was only fit to the right knee. So, we need to swap the medial/lateral axis
center = np.mean(femur.point_coords, axis=0)[0]
femur.point_coords = femur.point_coords * [-1, 1, 1]
femur.point_coords = femur.point_coords + [2*center, 0, 0]

# apply the same transformation to the cartilage so that they still line up. 
fem_cart.point_coords = fem_cart.point_coords * [-1, 1, 1]
fem_cart.point_coords = fem_cart.point_coords + [2*center, 0, 0]

femur.save_mesh(os.path.join(os.path.dirname(path_seg), f'flipped_{subject_id}_{side}_femur.vtk'))
fem_cart.save_mesh(os.path.join(os.path.dirname(path_seg), f'flipped_{subject_id}_{side}_fem_cart.vtk'))
```

Once the meshes are created, or if you just want to use the previously created and downloaded meshes, you 
can move on to running the script `recon_mesh.py` to reconstruct the bone/cartilage surface using the NSM. 

5. Get a femur "B-Score". 
The B-Score is a metric proposed by [Bowes M. et al. Ann Rheum Dis 2021](https://pubmed.ncbi.nlm.nih.gov/33188042/). 
This is a single metric that captures osteoarthritis severity. We have shown that the B-Score can be applied 
[using an NSM](https://www-sciencedirect-com.stanford.idm.oclc.org/science/article/pii/S2772654123000168). For
convenience, we have created a B-score decision line using the training data. You can use that decison line to 
fit a B-score to your data. 0 = average healthy knee (KL = 0), <0 is "super healthy", >0 is in the direction of OA
with a score >2 being someone that is not "healthy" (KL=0). Higher scores equate to more severe OA. 



## Instructions to download data, install package, train new model. 
1. Install NSM (see above # 1)

2. Download data 
We'll download the first subfolder of the training data (500 surfaces). Note, this
is 10GB of data (~20mb/file * 500 files). If you want to try it on less data, there
are results in the paper with as little as 50 examples, then try downloading 
individual files. 

```bash
mkdir example_recon    
```

```python
from huggingface_hub import snapshot_download

# download first 500 training examples
snapshot_download(
    repo_id="aagatti/ShapeMedKnee", 
    local_dir='dataset',
    allow_patterns='meshes/train/subfolder_0/*',
    repo_type='dataset'
)

# If you want to use the official validation set then, download it to: 
snapshot_download(
    repo_id="aagatti/ShapeMedKnee", 
    local_dir='dataset',
    allow_patterns='meshes/val/subfolder_0/*',
    repo_type='dataset'
)
```
3. Run `train_model.py` to train a new model 

Parameters to update: 
- CACHE: 
    This will save sampled meshes so they can be used for future runs.
    This information needs to be provided in some way - it is suggested
    to add it as an environment variable (`LOC_SDF_CACHE`) in your
    `~/.bashrc`
- USE_WANDB: 
    This will log to WANDB. If `True` you'll need to set an environment 
    variable `WANDB_KEY`. You can get this from the settings on your
    WANDB account, and set it in your `~/.bashrc` so that it is always 
    there for these experiments. You'll also need to update other v
    variables related to wandb logging: 
    - PROJECT_NAME
    - ENTITY_NAME
    - ENTITY
    - RUN_NAME
- N_TRAIN: 
    How many meshes you want to use for fitting the NSM. By default this 
    is just 50 so the experiment can get started quickly. This tutorial
    downloads upto 500. 
- N_VAL:
    Again, only a small number (10) are used for this example. 
- LOC_SAVE_NEW_MODELS:`
    This is where the model results wil be saved. Be default it is in this
    directory in `models/`

This script will train a model and save it iteratively during training. You 
can use your trained model to reconstruct bones as shown in the example for
loading data and reconstructing bones. 

