#Imports
import os
import sys
basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(basepath)
from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parents[1])
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ext.my_functions as my
from datetime import datetime
import ERC_bayesian_segmentation.relabeling as relab
import glob
import scipy.sparse as sp
import ext.bias_field_correction_torch as bf
import csv
import argparse
import math
from torch.nn.functional import grid_sample

########################################################

parser = argparse.ArgumentParser(description='Bayesian segmentation.')
parser.add_argument("--i", help="Image to segment.")
parser.add_argument("--i_seg", help="SynthSeg of image to segment (must include parcels). Will be computed if it does not exist")
parser.add_argument("--i_field", help="Registration to our fake reference image, provided by synthmorph. Will be computed if if does not exist")
parser.add_argument("--atlas_dir", help="Atlas directory")
parser.add_argument("--gmm_mode", help="GMM mode", default="1mm")
parser.add_argument("--bf_mode", help="bias field basis function: dct, polynomial, or hybrid", default="dct")
parser.add_argument("--o", help="Output directory.")
parser.add_argument("--write_rgb", action="store_true", help="Write soft segmentation to dis as RGB file.")
parser.add_argument("--write_bias_corrected", action="store_true", help="Write bias field corrected image to disk")
parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("--threads", type=int, default=1, help="(optional) Number of CPU cores to be used. Default is 1. You can use -1 to use all available cores")
parser.add_argument("--skip", type=int, default=1, help="(optional) Skipping factor to easy memory requirements of priors when estimating Gaussian parameters. Default is 1.")
parser.add_argument("--synthmorph_reg", type=float, default=0.05, help="(optional) Value of SynthMorph regularizer. Must be between 0 and 1. Default is 0.05.")
parser.add_argument("--resolution", type=float, default=0.4, help="(optional) Resolution of output segmentation")
parser.add_argument("--skip_bf", action="store_true", help="Skip bias field correction")
args = parser.parse_args()

########################################################
if args.i is None:
    raise Exception('Input image is required')
if args.i_seg is None:
    raise Exception('SynthSeg file  of input image is required')
if args.i_field is None:
    raise Exception('Atlas registration file of input image is required')
if args.atlas_dir is None:
    raise Exception('Atlas directory must be provided')
if args.o is None:
    raise Exception('Output directory must be provided')

########################################################

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################################

# limit the number of threads to be used if running on CPU
if args.threads<0:
    args.threads = os.cpu_count()
    print('using all available threads ( %s )' % args.threads)
else:
    print('using %s thread(s)' % args.threads)
torch.set_num_threads(args.threads)

########################################################

# Disable gradients
torch.set_grad_enabled(False)


################

# Input data
input_volume = args.i
input_seg = args.i_seg
atlas_field = args.i_field
atlas_dir = args.atlas_dir
LUT_file = os.path.join(BASE_PATH, 'data_simplified', 'AllenAtlasLUT')
output_dir = args.o
skip_bf = args.skip_bf
bf_mode = args.bf_mode
synthmorph_reg = args.synthmorph_reg
if (synthmorph_reg<0) or (synthmorph_reg>1):
    print('Synthmorph regularizer must be between 0 and 1. Exitting...')
    sys.exit(1)
resolution = args.resolution
if (resolution<=0):
    print('Resolution must be non-negative; Exitting...')
    sys.exit(1)
skip = args.skip
if (skip<1):
    print('Skip cannot be less than 1; exitting...')
    sys.exit(1)

########################################################
# Detect problems with output directory right off the bat
if os.path.exists(output_dir):
    print('Warning: output directory exists (I will still proceed))')
else:
    os.mkdir(output_dir)


########################################################

# Constants
dtype = torch.float32
SET_BG_TO_CSF = True # True = median of ventricles -> it seems much better than 0!
RESOLUTION_ATLAS = 0.2
TOL = 1e-9

############

if dtype == torch.float64:
    numpy_dtype = np.float64
elif dtype == torch.float32:
    numpy_dtype = np.float32
elif dtype == torch.float16:
    numpy_dtype = np.float16
else:
    raise Exception('type not supported')

########################################################

if torch.cuda.is_available():
    print('Using the GPU')
    device = torch.device('cuda:0')
else:
    print('Using the CPU')
    device = torch.device('cpu')

########################################################

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

########################################################
print('Reading input image')
Iim, aff = my.MRIread(input_volume)
Iim = np.squeeze(Iim)

########################################################
synthseg_exists = False
if os.path.isfile(input_seg):
    print('Found input synthseg segmentation')
    synthseg_exists = True

if synthseg_exists:
    print('Input segmentation exists; making sure it includes parcellation!')
    tmp, _ = my.MRIread(input_seg)
    if np.sum(tmp>1000)==0:
        raise Exception('Segmentation does not include parcellation! Please use different file or re-run SynthSeg with --parc')
else:
    print('Running SynthSeg')
    cmd = 'mri_synthseg --i ' + input_volume +  ' --o '  + input_seg + ' --threads ' + str(args.threads) + ' --cpu --parc --robust '
    cmd = cmd + ' --vol ' + output_dir + '/SynthSeg_volumes.csv'
    a = os.system(cmd + ' >/dev/null')
    if a > 0:
        print('Error in mri_synthseg; exitting...')
        sys.exit(1)


TMP_RESAMPLED = output_dir + '/tmp.resampled.mgz'
a = os.system('mri_convert ' + input_seg + ' ' + TMP_RESAMPLED + ' -rl ' + input_volume + ' -rt nearest -odt float >/dev/null')
if a>0:
    print('Error in mri_convert; exitting...')
    sys.exit(1)
Sim, _ = my.MRIread(TMP_RESAMPLED)
os.system('rm -rf ' + TMP_RESAMPLED)

########################################################
if skip_bf==False:
    print('Correcting bias field')
    print('   Trying model with polynomial basis functions')
    try:
        Iim, _ = bf.correct_bias(Iim, Sim, maxit=100, penalty=0.1, order=4, device=device, dtype=dtype, basis=bf_mode)
    except:
        if device.type=='cpu':
            raise Exception('Bias correction failed (out of memory?)')
        else:
            print('Bias correction on GPU failed; trying with CPU')
            Iim, _ = bf.correct_bias(Iim, Sim, maxit=100, penalty=0.1, order=4, device='cpu', dtype=dtype, basis=bf_mode)
    if args.write_bias_corrected:
        aux = (Iim / np.max(Iim) * 255).astype(np.uint8)
        my.MRIwrite(aux, aff, output_dir + '/bf_corrected.mgz')

print('Normalizing intensities')
Iim = Iim * 110 / np.median(Iim[(Sim==2) | (Sim==41)])

# We should do tensors at this point...
Sim = torch.tensor(Sim, dtype=torch.int, device=device)
Iim = torch.tensor(Iim, dtype=dtype, device=device)

########################################################

# Registration part

BRAINSTEM = 16
BRAINSTEM_L = 161
BRAINSTEM_R = 162

if os.path.isfile(atlas_field):
    print('   Registration file found; no need to run SynthMorph!')
else:
    print('   Running SynthMorph')
    atlas_file = os.path.join(BASE_PATH, 'data_mni','atlas.nii.gz')
    cmd = 'mri_synthmorph register -m joint'
    cmd+= (' -j ' + str(args.threads)  + ' -t ' + atlas_field)
    cmd+= (' -r  ' + str(synthmorph_reg))
    if device.type!='cpu':
        cmd += ' -g'
    cmd+= (' ' + atlas_file + '  ' + input_volume + ' >/dev/null')
    a = os.system(cmd)
    if a>0:
        print('Error in mri_synthmorph; exitting...')
        sys.exit(1)
FIELD, FIELDaff = my.MRIread(atlas_field)
FIELD = torch.tensor(FIELD, dtype=dtype, device=device)
# sanity check
if np.sum(np.abs(aff-FIELDaff))>1e-6:
    raise Exception('affine matrix of deformation field does not coincide with that of input')
    sys.exit(1)

# FIELD is a shift in RAS; I need to compute the final RAS coordinates
# This is also a good opportunity to get the grids!
vectors = [torch.arange(0, s, device=device) for s in Iim.shape]
IJKim = torch.stack(torch.meshgrid(vectors))
RAS = torch.stack([aff[0,0] * IJKim[0] + aff[0,1] * IJKim[1] + aff[0,2] * IJKim[2] + aff[0,3] + FIELD[...,0],
                     aff[1,0] * IJKim[0] + aff[1,1] * IJKim[1] + aff[1,2] * IJKim[2] + aff[1,3] + FIELD[...,1],
                     aff[2,0] * IJKim[0] + aff[2,1] * IJKim[1] + aff[2,2] * IJKim[2] + aff[2,3] + FIELD[...,2]])

# Added in summer 2024: kill bottom of medulla
CSF = 24
Sim[((Sim==BRAINSTEM) | (Sim==CSF)) & (RAS[2]<(-60))] = 0
# Subdivide
LEFT = (RAS[0]<0)
Sim[(Sim==BRAINSTEM) & LEFT] = BRAINSTEM_L
Sim[(Sim==BRAINSTEM) & (LEFT==0)] = BRAINSTEM_R

#######################################

# Prepare data for each hemisphere
Sim_orig = Sim.clone()
Iim_orig = Iim.clone()
aff_orig = aff.copy()
Mim = []
Iim = []
Sim = []
aff = []
RAShemis = []

for h in range(2):
    print('  Creating and applying mask')
    if h==0: # left hemi
        M = ( (Sim_orig < 30) | (Sim_orig==161) | ( (Sim_orig > 1000) & (Sim_orig < 2000) ) )
    else: # right hemi
        M = ( ( (Sim_orig > 40) & (Sim_orig<100) )  | (Sim_orig==162) | (Sim_orig>2000) )
    if True:
        M[Sim_orig==4] = 0
        M[Sim_orig==5] = 0
        M[Sim_orig==43] = 0
        M[Sim_orig==44] = 0
    M[Sim_orig==14] = 0
    M[Sim_orig==15] = 0
    M[Sim_orig==24] = 0
    M[Sim_orig==0] = 0
    # I now do this with the resampled mask at each level, to avoid blurring mask edges
    # Iim[M==0] = 0
    Mim_temp, cropping = my.cropLabelVol(M.cpu().detach().numpy(), margin=5)
    Sim_temp = my.applyCropping(Sim_orig.cpu().detach().numpy(), cropping)
    Iim_temp = my.applyCropping(Iim_orig.cpu().detach().numpy(), cropping)
    aff_temp = aff_orig.copy()
    aff_temp[:3, -1] = aff_temp[:3, -1] + aff_temp[:-1, :-1] @ cropping[:3]
    Mim.append(torch.tensor(Mim_temp, dtype=torch.bool, device=device))
    Iim.append(torch.tensor(Iim_temp, dtype=dtype, device=device))
    Sim.append(torch.tensor(Sim_temp, dtype=torch.int, device=device))
    aff.append(aff_temp)
    RAShemis.append(RAS[:,cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5]])


########################################################

# Read atlas (only necessiary for the first pass, ie, the left hemisphere
print('Reading in atlas')

# Get the label groupings and atlas labels from the config files
# We assume left hemi (it doesn't matter)
aseg_label_list = torch.unique(Sim[0][Mim[0]>0]).cpu().detach().numpy()
tissue_index, grouping_labels, label_list, number_of_gmm_components = relab.get_tissue_settings(
            os.path.join(BASE_PATH, 'data_simplified', 'atlas_names_and_labels.yaml'),
            os.path.join(BASE_PATH, 'data_simplified', 'combined_atlas_labels_' + args.gmm_mode + '.yaml'),
            os.path.join(BASE_PATH, 'data_simplified', 'combined_aseg_labels_' + args.gmm_mode + '.yaml'),
            os.path.join(BASE_PATH, 'data_simplified', 'gmm_components_' + args.gmm_mode + '.yaml'),
            aseg_label_list
)
tidx = tissue_index[np.where(label_list == 0)[0][0]]
if tidx>0:
    raise Exception('First tissue class must be the background')
n_tissues = np.max(tissue_index) + 1
n_labels = len(label_list)
atlas_names = sorted(glob.glob(atlas_dir + '/label_*.npz'))
atlas_size = np.load(atlas_dir + '/size.npy')

# We do need the grouping labels for the right hemi, too
_, grouping_labels_right, _, _ = relab.get_tissue_settings(
            os.path.join(BASE_PATH, 'data_simplified', 'atlas_names_and_labels.yaml'),
            os.path.join(BASE_PATH, 'data_simplified', 'combined_atlas_labels_' + args.gmm_mode + '.yaml'),
            os.path.join(BASE_PATH, 'data_simplified', 'combined_aseg_labels_' + args.gmm_mode + '.yaml'),
            os.path.join(BASE_PATH, 'data_simplified', 'gmm_components_' + args.gmm_mode + '.yaml'),
            torch.unique(Sim[1][Mim[1]>0]).cpu().detach().numpy()
)

class LabelDataset(Dataset):

    def __init__(self, fnames):
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        print(item, self.fnames[item])
        prior = sp.load_npz(self.fnames[item])
        prior_indices = torch.as_tensor(prior.row)
        prior_values = torch.as_tensor(prior.data)
        return prior_indices, prior_values

# TODO: without this line, I get weird runtime errors...
prefetch = 4
workers = 2
prefetch_factor = max(prefetch//workers, 1)
label_loader = DataLoader(LabelDataset(atlas_names), num_workers=workers, prefetch_factor=prefetch_factor)
A = np.zeros([*atlas_size, n_tissues], dtype=numpy_dtype)
for n, (prior_indices, prior_values) in enumerate(label_loader):
    print('Reading in label ' + str(n+1) + ' of ' + str(n_labels))
    if prior_indices.numel() == 0:
        continue
    prior_indices = torch.as_tensor(prior_indices, device=device, dtype=torch.long).squeeze()
    prior_values = torch.as_tensor(prior_values, device=device, dtype=dtype).squeeze()
    idx = tissue_index[n]
    if n == 0:
        prior = torch.sparse_coo_tensor(prior_indices[None], prior_values,
                                        [torch.Size(atlas_size).numel()]).to_dense()
        del prior_indices, prior_values
        prior = prior.reshape(torch.Size(atlas_size)).cpu().numpy()
        A[:, :, :, idx] = A[:, :, :, idx] + prior
    else:
        prior_indices = my.ind2sub(prior_indices, atlas_size)
        min_x, max_x = prior_indices[0].min().item(), prior_indices[0].max().item() + 1
        min_y, max_y = prior_indices[1].min().item(), prior_indices[1].max().item() + 1
        min_z, max_z = prior_indices[2].min().item(), prior_indices[2].max().item() + 1
        crop_atlas_size = [max_x - min_x, max_y - min_y, max_z - min_z]
        prior_indices[0] -= min_x
        prior_indices[1] -= min_y
        prior_indices[2] -= min_z
        prior = torch.sparse_coo_tensor(prior_indices, prior_values, crop_atlas_size).to_dense()
        crop = (slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z))
        A[(*crop, idx)] = A[(*crop, idx)] + prior.cpu().numpy()

A = torch.tensor(A, dtype=dtype, device=device)
aff_A = np.diag([.2, .2, .2, 1]) # assumes left hemisphere; we'll deal with rest later

########################################################
print('Computing initial values for means and variances')
mus_ini = []
vars_ini = []
mixture_weights = []

for t in range(len(number_of_gmm_components)-1):
    x = []
    for l in grouping_labels[t+1]:
        x.append(Iim[0][Sim[0]==l])
    for l in grouping_labels_right[t+1]:
        x.append(Iim[0][Sim[0]==l])

    if len(x) > 0:
        x = torch.concatenate(x)
        mu = torch.median(x)
        std = 1.4826 * torch.median(torch.abs(x - mu))
        var = std ** 2
        if number_of_gmm_components[t+1]==1:
            mus_ini.append(mu[None])
            vars_ini.append(var[None])
            mixture_weights.append(torch.ones(1,dtype=dtype,device=device))
        else:
            # Estimate GMM with shared variance (avoids a component with tiny variance)
            nc = number_of_gmm_components[t+1]
            nx = len(x)
            gmm_mus = torch.linspace(mu - 0.5 * std, mu + 0.5 * std, nc, dtype=dtype, device=device)
            gmm_var= var * torch.ones(1, dtype=dtype, device=device)
            gmm_ws = (1 / float(nc)) * torch.ones(nc, dtype=dtype, device=device)
            W = torch.zeros([nx, nc], dtype=dtype, device=device)
            for its in range(200):
                # E step
                for c in range(nc):
                    W[:, c] = gmm_ws[c] / torch.sqrt(2.0 * torch.pi * torch.sqrt(gmm_var)) * torch.exp(-0.5 * (x - gmm_mus[c])**2 / gmm_var)
                normalizer = torch.sum(W + 1e-9, axis=1)
                # print(-torch.mean(torch.log(normalizer)))
                W /= normalizer[:, None]
                # M step
                denominators = torch.sum(W, axis=0)
                gmm_ws = denominators / torch.sum(denominators)
                gmm_var = 0
                for c in range(nc):
                    gmm_mus[c] = torch.sum(W[:, c] * x) / denominators[c]
                    aux = x - gmm_mus[c]
                    gmm_var += torch.sum(W[:, c] * aux * aux)
                gmm_var /= torch.sum(denominators)

            mus_ini.append(gmm_mus)
            vars_ini.append(gmm_var * torch.ones(nc, dtype=dtype, device=device))
            mixture_weights.append(gmm_ws)

mus_ini = torch.concatenate(mus_ini)
vars_ini = torch.concatenate(vars_ini)
mixture_weights = torch.concatenate(mixture_weights)

if SET_BG_TO_CSF:
    x = []
    for l in [4, 5]: # , 24]:
        x.append(Iim[0][Sim[0]==l])
    for l in [43, 44]: # , 24]:
        x.append(Iim[1][Sim[1]==l])
    mu_bg  = torch.median(torch.concatenate(x))
else:
    mu_bg = torch.tensor(0, dtype=dtype, device=device)

########################################################

# Process each hemisphere
means_both_sides = []
variances_both_sides = []
weights_both_sides = []
for h in range(2):
    if h==0:
        print('Estimating GMM parameters for left hemisphere')
    else:
        print('Estimating GMM parameters for right hemisphere')

    print('  Resizing image, mask, and coordinates')
    I_r, aff_r = my.torch_resize(Iim[h], aff[h], resolution, device, dtype=dtype)
    M_r, _ = my.torch_resize(Mim[h], aff[h], resolution, device, dtype=dtype)
    I_r[M_r<0.5] = mu_bg
    RAS_r, _ = my.torch_resize(RAShemis[h].permute([1,2,3,0]), aff[h], resolution, device, dtype=dtype)

    print('  Deforming atlas')
    Tatl = np.load(os.path.join(BASE_PATH, 'data_mni', 'atlas.T.npy'))
    T = np.linalg.inv(aff_A) @ Tatl
    if h==1: # right hemisphere; easy!
        T[:,0] = -T[:,0]
    Iskip = T[0, 0] * RAS_r[::skip, ::skip, ::skip, 0] + T[0, 1] * RAS_r[::skip, ::skip, ::skip, 1] + T[0, 2] * RAS_r[::skip, ::skip, ::skip, 2] + T[0, 3]
    Jskip = T[1, 0] * RAS_r[::skip, ::skip, ::skip, 0] + T[1, 1] * RAS_r[::skip, ::skip, ::skip, 1] + T[1, 2] * RAS_r[::skip, ::skip, ::skip, 2] + T[1, 3]
    Kskip = T[2, 0] * RAS_r[::skip, ::skip, ::skip, 0] + T[2, 1] * RAS_r[::skip, ::skip, ::skip, 1] + T[2, 2] * RAS_r[::skip, ::skip, ::skip, 2] + T[2, 3]

    Iskip = 2 * (Iskip / (A.shape[0] - 1)) - 1
    Jskip = 2 * (Jskip / (A.shape[1] - 1)) - 1
    Kskip = 2 * (Kskip / (A.shape[2] - 1)) - 1

    # Careful with grid_sample switching the order of the dimensions (go figure...)
    priors = grid_sample(A.permute([3,0,1,2])[None, ...], torch.stack([Kskip, Jskip, Iskip],axis=-1)[None,...], align_corners=True)
    priors = torch.permute(priors[0], [1,2,3,0])

    # Deal with voxels outside the FOV
    missing_mass = 1 - torch.sum(priors, axis=-1)
    priors[..., 0] += missing_mass

    ####
    print('  EM for parameter estimation')
    # data
    means = torch.tensor([mu_bg, *mus_ini], device=device, dtype=dtype)
    var_bg = torch.min(vars_ini)
    variances = torch.tensor([var_bg, *vars_ini], device=device, dtype=dtype)
    weights = torch.tensor([1.0, *mixture_weights], dtype=dtype, device=device)
    W = torch.zeros([*priors.shape[:-1], number_of_gmm_components.sum()], dtype=dtype, device=device)
    x = I_r[::skip, ::skip, ::skip]

    # We now put a Scaled inverse chi-squared prior on the variances to prevent them from going to zero
    prior_count = 100 / (resolution ** 3) / (skip ** 3)
    prior_variance = var_bg
    loglhood_old = -10000
    for em_it in range(100):
        # E step
        for c in range(n_tissues):
            prior = priors[:, :, :, c]
            num_components = number_of_gmm_components[c]
            for g in range(num_components):
                gaussian_number = sum(number_of_gmm_components[:c]) + g
                d = x - means[gaussian_number]
                W[:, :, :, gaussian_number] = weights[gaussian_number] * prior * torch.exp(
                    -d * d / (2 * variances[gaussian_number])) / torch.sqrt(
                    2.0 * torch.pi * variances[gaussian_number])

        normalizer = 1e-9 + torch.sum(W, dim=-1, keepdim=True)
        loglhood = torch.mean(torch.log(normalizer)).detach().cpu().numpy()
        W = W / normalizer

        # M step
        prior_loglhood = torch.zeros(1,dtype=dtype,device=device)
        for c in range(number_of_gmm_components.sum()):
            # crucially, we skip the background when we update the parameters (but we still add it to the cost)
            if c > 0:
                norm = torch.sum(W[:, :, :, c])
                means[c] = torch.sum(x * W[:, :, :, c]) / norm
                d = x - means[c]
                variances[c] = (torch.sum(d * d * W[:, :, :, c]) + prior_count * prior_variance) / ( norm + prior_count + 2)
            v = variances[c]
            prior_loglhood = prior_loglhood - ((1 + 0.5 * prior_count) * torch.log(v) + 0.5 * prior_count * prior_variance / v) / torch.numel(normalizer)
        loglhood = loglhood + prior_loglhood.detach().cpu().numpy()

        mixture_weights = torch.sum(W[:, :, :, 1:].reshape([np.prod(priors.shape[:-1]), number_of_gmm_components.sum() - 1]) + 1e-9, axis=0)
        for c in range(n_tissues - 1):
            # mixture weights are normalized (those belonging to one mixture sum to one)
            num_components = number_of_gmm_components[c + 1]
            gaussian_numbers = torch.tensor(np.sum(number_of_gmm_components[1:c + 1]) + \
                                            np.array(range(num_components)), device=device, dtype=dtype).int()

            mixture_weights[gaussian_numbers] /= torch.sum(mixture_weights[gaussian_numbers])

        weights[1:] = mixture_weights

        if (torch.sum(torch.isnan(means)) > 0) or (torch.sum(torch.isnan(variances)) > 0):
            print('nan in Gaussian parameters...')
            import pdb;

            pdb.set_trace()

        # print('         Step %d of EM, -loglhood = %.6f' % (em_it + 1, -loglhood ), flush=True)
        if (loglhood - loglhood_old) < TOL:
            print('         Decrease in loss below tolerance limit')
            break
        else:
            loglhood_old = loglhood

    means_both_sides.append(means)
    variances_both_sides.append(variances)
    weights_both_sides.append(weights)


###########

print('Computing Gaussians at full resolution (we will reuse over and over)')
GAUSSIAN_LHOODS = []
for h in range(2):
    I_full, aff_full = my.torch_resize(Iim[h], aff[h], resolution, device, dtype=dtype)
    M_full, _ = my.torch_resize(Mim[h], aff[h], resolution, device, dtype=dtype)
    I_full[M_full < 0.5] = mu_bg
    GAUSSIAN_LHOODS.append(torch.zeros([*I_full.shape, sum(number_of_gmm_components)], dtype=dtype, device=device))
    for c in range(sum(number_of_gmm_components)):
        # The 1e-9 ensures no zeros were prior is not zero
        GAUSSIAN_LHOODS[h][..., c] = 1.0 / torch.sqrt(2 * math.pi * variances_both_sides[h][c]) * torch.exp(
        -0.5 * torch.pow(I_full - means_both_sides[h][c], 2.0) / variances_both_sides[h][c]) + 1e-9

print('Computing deformed coordiantes at full resolution (we will reuse over and over)')
I = []
J = []
K = []
for h in range(2):
    Tatl = np.load(os.path.join(BASE_PATH, 'data_mni', 'atlas.T.npy'))
    T = np.linalg.inv(aff_A) @ Tatl
    if h==1: # right hemisphere; easy!
        T[:,0] = -T[:,0]
    RAS_r, _ = my.torch_resize(RAShemis[h].permute([1, 2, 3, 0]), aff[h], resolution, device, dtype=dtype)
    I.append(2 * ((T[0, 0] * RAS_r[..., 0] + T[0, 1] * RAS_r[..., 1] + T[0, 2] * RAS_r[..., 2] + T[0, 3]) / (A.shape[0] - 1)) - 1)
    J.append(2 * ((T[1, 0] * RAS_r[..., 0] + T[1, 1] * RAS_r[..., 1] + T[1, 2] * RAS_r[..., 2] + T[1, 3]) / (A.shape[1] - 1)) - 1)
    K.append(2 * ((T[2, 0] * RAS_r[..., 0] + T[2, 1] * RAS_r[..., 1] + T[2, 2] * RAS_r[..., 2] + T[2, 3]) / (A.shape[2] - 1)) - 1)

print('Computing normalizers (faster to do now with clustered priors)')
# We deform one class at the time; slower, but less memory
normalizers = []
for h in range(2):
    normalizers.append(torch.zeros(GAUSSIAN_LHOODS[h].shape[:-1], dtype=dtype, device=device))
    # normalizers[h][:] = 1e-9
    gaussian_number = 0
    for c in range(A.shape[-1]):
        prior = grid_sample(A[None, None, ..., c], torch.stack([K[h], J[h], I[h]], axis=-1)[None, ...], align_corners=True)[0,0,...]
        if c==0: # background
            prior[(I[h]<(-1)) | (I[h]>1) |  (J[h]<(-1)) | (J[h]>1) | (K[h]<(-1)) | (K[h]>1)] = 1.0
        lhood = torch.zeros_like(prior)
        for g in range(number_of_gmm_components[c]):
            lhood += (weights_both_sides[h][gaussian_number] * GAUSSIAN_LHOODS[h][..., gaussian_number])
            gaussian_number += 1
        normalizers[h] += (prior * lhood)


print('Deforming one label at the time')
names, colors = my.read_LUT(LUT_file)
seg = []
seg_rgb = []
max_p = []
vols = []
for h in range(2):
    seg.append(torch.zeros(normalizers[h].shape, dtype=torch.int, device=device))
    seg_rgb.append(torch.zeros([*normalizers[h].shape, 3], dtype=dtype, device=device))
    max_p.append(torch.zeros(normalizers[h].shape, dtype=dtype, device=device))
    vols.append(torch.zeros(n_labels, device=device, dtype=dtype))


# TODO: choose good number of workers/prefetch factor
for n, (prior_indices, prior_values) in enumerate(label_loader):
    print('Deforming label ' + str(n + 1) + ' of ' + str(n_labels))

    if prior_indices.numel() == 0:
        continue
    prior_indices = torch.as_tensor(prior_indices, device=device, dtype=torch.long).squeeze()
    prior_values = torch.as_tensor(prior_values, device=device, dtype=dtype).squeeze()

    if n == 0:
        # background
        prior = torch.sparse_coo_tensor(prior_indices[None], prior_values,
                                        [torch.Size(atlas_size).numel()]).to_dense()
        del prior_indices, prior_values
        prior = prior.reshape(torch.Size(atlas_size))

    else:

        # find bounding box of label in atlas space
        prior_indices = my.ind2sub(prior_indices, atlas_size)
        min_x, max_x = prior_indices[0].min().item(), prior_indices[0].max().item()
        min_y, max_y = prior_indices[1].min().item(), prior_indices[1].max().item()
        min_z, max_z = prior_indices[2].min().item(), prior_indices[2].max().item()
        crop_atlas_size = [max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1]
        prior_indices[0] -= min_x
        prior_indices[1] -= min_y
        prior_indices[2] -= min_z
        prior = torch.sparse_coo_tensor(prior_indices, prior_values, crop_atlas_size).to_dense()
        del prior_indices, prior_values

    for h in range(2):
        skip_this_label = False
        if n==0:
            Irescaled = I[h]
            Jrescaled = J[h]
            Krescaled = K[h]
            lr_crop = (slice(None),) * 3
        else:
            # find bounding box of label in MRI space
            Irescaled = I[h] * ((A.shape[0] - 1) / (max_x - min_x)) + ( (A.shape[0] - 1 - min_x - max_x) / (max_x - min_x) )
            Jrescaled = J[h] * ((A.shape[1] - 1) / (max_y - min_y)) + ( (A.shape[1] - 1 - min_y - max_y) / (max_y - min_y) )
            Krescaled = K[h] * ((A.shape[2] - 1) / (max_z - min_z)) + ( (A.shape[2] - 1 - min_z - max_z) / (max_z - min_z) )
            mask = (Irescaled >= (-1))
            mask &= (Irescaled <= 1)
            mask &= (Jrescaled >= (-1))
            mask &= (Jrescaled <= 1)
            mask &= (Krescaled >= (-1))
            mask &= (Krescaled <= 1)
            if mask.any()==False:
                skip_this_label = True
            else:
                nx, ny, nz = mask.shape
                tmp = mask.reshape([nx, -1]).any(-1).nonzero()
                lr_min_x, lr_max_x = tmp.min().item(), tmp.max().item() + 1
                tmp = mask.movedim(0, -1).reshape([ny, -1]).any(-1).nonzero()
                lr_min_y, lr_max_y = tmp.min().item(), tmp.max().item() + 1
                tmp = mask.reshape([-1, nz]).any(0).nonzero()
                lr_min_z, lr_max_z = tmp.min().item(), tmp.max().item() + 1
                del tmp, mask
                lr_crop = (slice(lr_min_x, lr_max_x), slice(lr_min_y, lr_max_y), slice(lr_min_z, lr_max_z))

        if skip_this_label==False:

            prior_resampled = grid_sample(prior[None, None, ...], torch.stack([Krescaled[lr_crop], Jrescaled[lr_crop], Irescaled[lr_crop]], axis=-1)[None, ...], align_corners=True)[0, 0, ...]
            if n==0: # background
                prior_resampled[(Krescaled[lr_crop]<(-1)) | (Krescaled[lr_crop]>1) | (Jrescaled[lr_crop]<(-1)) | (Jrescaled[lr_crop]>1)  | (Irescaled[lr_crop]<(-1)) | (Irescaled[lr_crop]>1)  ] = 1.0
            del Irescaled; del Jrescaled; del Krescaled
            num_components = number_of_gmm_components[tissue_index[n]]
            gaussian_numbers = torch.tensor(np.sum(number_of_gmm_components[:tissue_index[n]]) + \
                                    np.array(range(num_components)), device=device, dtype=dtype).int()
            lhood = torch.sum(GAUSSIAN_LHOODS[h][:, :, :, gaussian_numbers] * weights_both_sides[h][None, None, None, gaussian_numbers], 3)
            post = torch.squeeze(prior_resampled)
            post *= lhood[lr_crop]
            post /= normalizers[h][lr_crop]
            if n==0:
                post[torch.isnan(post)] = 1.0
            else:
                post[torch.isnan(post)] = 0.0
            del prior_resampled
            vols[h][n] = torch.sum(post) * (resolution ** 3)
            mask = (post > max_p[h][lr_crop])
            max_p[h][lr_crop][mask] = post[mask]
            lab = int(label_list[n])
            seg[h][lr_crop].masked_fill_(mask, lab)
            del mask
            for c in range(3):
                seg_rgb[h][(*lr_crop, c)].add_(post, alpha=colors[lab][c])
print('\n')

########################################################

print('Writing results to disk')

_, aff_l = my.torch_resize(Iim[0], aff[0], resolution, device, dtype=dtype)
my.MRIwrite(seg[0].detach().cpu().numpy().astype(np.uint16), aff_l, output_dir + '/seg_left.nii.gz')
_, aff_r = my.torch_resize(Iim[1], aff[1], resolution, device, dtype=dtype)
my.MRIwrite(seg[1].detach().cpu().numpy().astype(np.uint16), aff_r, output_dir + '/seg_right.nii.gz')
if args.write_rgb:
    my.MRIwrite(seg_rgb[0].detach().cpu().numpy().astype(np.uint8), aff_l, output_dir + '/seg_left.rgb.nii.gz')
    my.MRIwrite(seg_rgb[1].detach().cpu().numpy().astype(np.uint8), aff_r, output_dir + '/seg_right.rgb.nii.gz')
vols_l = vols[0].detach().cpu().numpy()
vols_r = vols[1].detach().cpu().numpy()

with open(output_dir + '/vols_left.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    aux = label_list[1:]
    row = []
    for l in aux:
        row.append(names[int(l)])
    writer.writerow(row)

    row = []
    for j in range(1, len(vols_l)):
        row.append(str(vols_l[j]))
    writer.writerow(row)

with open(output_dir + '/vols_right.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    aux = label_list[1:]
    row = []
    for l in aux:
        row.append(names[int(l)])
    writer.writerow(row)

    row = []
    for j in range(1, len(vols_r)):
        row.append(str(vols_r[j]))
    writer.writerow(row)


# Copy LUT file, for convenience
a = os.system('cp ' + LUT_file + ' ' + output_dir +  '/lookup_table.txt >/dev/null')
if a==0:
    LUT_file = output_dir +  '/lookup_table.txt'

# Print commands to visualize output
print('You can try these commands to visualize outputs:')
cmd = '  freeview -v ' + input_volume
if args.write_bias_corrected:
    cmd = cmd + ' -v ' + output_dir + '/bf_corrected.mgz '
cmd = cmd + ' -v ' + output_dir + '/seg_left.nii.gz:colormap=lut:lut=' + LUT_file
cmd = cmd + ' -v ' + output_dir + '/seg_right.nii.gz:colormap=lut:lut=' + LUT_file
if args.write_rgb:
    cmd = cmd + ' -v ' + output_dir + '/seg_left.rgb.nii.gz:rgb=true'
    cmd = cmd + ' -v ' + output_dir + '/seg_right.rgb.nii.gz:rgb=true'
print(cmd)
print(' ')
print('All done!')

now2 = datetime.now()

current_time = now2.strftime("%H:%M:%S")
print("Current Time =", current_time)

runtime = now2 - now

print("Running Time =", runtime)

