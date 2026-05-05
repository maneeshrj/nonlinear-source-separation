import os
import scipy
import numpy as np
import torch
import torch.nn.functional as F

### Utility functions for reading data, unpacking into tensors, normalization, padding, and cost weighting maps ###


# Read data from .mat file saved from MEDI toolbox
def read_single_mat_file(data_dir):
    mat_filenames = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    if len(mat_filenames) == 0:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    mat_filename = os.path.join(data_dir, mat_filenames[0])
    data = scipy.io.loadmat(mat_filename)
    return mat_filename, data


# Unpack data from .mat file into PyTorch tensors and move to GPU
def unpack_data(data, gpu):
    w1 = torch.tensor(data['mr'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)
    r = torch.tensor(data['R2p'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)
    br = torch.tensor(data['br'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)

    Ap = torch.tensor(data['Ap'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)
    Am = torch.tensor(data['Am'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)

    w2 = torch.tensor(data['mf'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)
    D = torch.tensor(data['D'], dtype=torch.complex64).unsqueeze(0).unsqueeze(0).to(gpu)
    f = torch.tensor(data['RDF'], dtype=torch.complex64).unsqueeze(0).unsqueeze(0).to(gpu)
    bf = torch.tensor(data['bf'], dtype=torch.complex64).unsqueeze(0).unsqueeze(0).to(gpu)

    N_std = torch.tensor(data['N_std'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)

    M_mag = torch.permute(torch.tensor(data['wGm'], dtype=torch.float32), (3,0,1,2)).unsqueeze(0).to(gpu)
    M_r = torch.permute(torch.tensor(data['wGr'], dtype=torch.float32), (3,0,1,2)).unsqueeze(0).to(gpu)
    M_CSF = torch.tensor(data['Mask_CSF'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)

    Mask =  torch.tensor(data['Mask'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)

    return w1, r, Ap, Am, w2, D, f, N_std, br, bf, M_mag, M_r, M_CSF, Mask


# Unpack data for nonlinear fitting, including magnitude image
def unpack_data_nonlinear(data, gpu):
    _, r, Ap, Am, w, D, f, N_std, br, bf, M_mag, M_r, M_CSF, Mask = unpack_data(data, gpu)

    mag_img  = torch.tensor(data['iMag'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(gpu)
    return r, w, D, f, N_std, br, bf, M_mag, M_r, M_CSF, Mask, mag_img


# Normalize tensor to have zero mean and unit variance
def normalize_zero_mean_unit_variance(tensor):
    return (tensor - tensor.mean()) / tensor.std()
     

# Move tensor from GPU to CPU and convert to NumPy array
def gpu_to_np(tensor):
    return tensor.detach().cpu().numpy()


# Flatten complex tensor to real by taking either real or imag part, depending on which has non-zero values
def flatten_complex(tensor):
    if tensor.is_complex() and tensor.real.sum() != 0:
        tensor_real = tensor.real
    elif tensor.is_complex() and tensor.imag.sum() != 0:
        tensor_real = tensor.imag
    else:
        tensor_real = tensor
    return tensor_real


# Pad tensor in x, y, z dimensions to make them multiples of 8 for compatibility with UNet architecture
def pad_xyz(xyz_shape, tensor):
    assert len(xyz_shape) == 3, "xyz_shape should be a tuple of (depth, height, width)"

    if xyz_shape[0] % 8 != 0:
        pad = 8 - (xyz_shape[0] % 8)
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad), mode='replicate')

    if xyz_shape[1] % 8 != 0:
        pad = 8 - (xyz_shape[1] % 8)
        tensor = F.pad(tensor, (0, 0, 0, pad, 0, 0), mode='replicate')

    if xyz_shape[2] % 8 != 0:
        pad = 8 - (xyz_shape[2] % 8)
        tensor = F.pad(tensor, (0, pad, 0, 0, 0, 0), mode='replicate')

    return tensor


# Get data term weighting mask based on noise standard deviation and brain mask
def get_dataterm_mask(N_std, Mask):
    with torch.no_grad():
        w = torch.where(N_std > 0, Mask / N_std, 0)
        w = w * Mask
        w_mean = w.sum() / torch.count_nonzero(w)
        w = w / w_mean
    return w


# Get reweighting map based on residuals to identify bad voxels and downweight them during training
def get_reweight_map(residual_f):
    with torch.no_grad():
        wres = residual_f.abs()
        mean = wres[torch.nonzero(wres, as_tuple=True)].mean()
        std = wres[torch.nonzero(wres, as_tuple=True)].std()
        wres = wres - mean
        std_res_f = std * 6
        
        # Reweight if residual is more than 6 stds away from the mean
        # This is a heuristic to identify bad voxels
        wres = wres/std_res_f
        reweight_map = torch.where(wres > 1, wres, 1.)
    return reweight_map
