import argparse
from pyuvdata import UVData, UVCal
import corrcal
import numpy as np
import re
from pathlib import Path
import yaml
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.optimize import fmin_cg


# FUNCTIONS
def get_models(data_lsts, model_cov_dir):
    """
    get model files that have starting lsts for model_lst_start <= data_lst range <= model_lst_end
    data_lsts need to be covered by the range of the model lsts
    if data lsts are wrapped --> model wrapped in same way
    eg. data = [6.28271469e+00 2.34071661e-04]
        models = [PosixPath('/lustre/aoc/projects/hera/rpascua/corrcal/cov_models/h6c/zen.LST.6.2820915.h6c_corrcal_model.npz'),
                PosixPath('/lustre/aoc/projects/hera/rpascua/corrcal/cov_models/h6c/zen.LST.0.0000000.h6c_corrcal_model.npz'),
                PosixPath('/lustre/aoc/projects/hera/rpascua/corrcal/cov_models/h6c/zen.LST.0.0010938.h6c_corrcal_model.npz')]
    """
    # Get all files in model dir
    model_files = sorted(model_cov_dir.glob("*.npz"))
    # Get start LSTs for all model files
    model_start_lsts = np.array(
            [float(re.findall("\d+.\d+", fn.name)[0]) for fn in model_files]
        )
    # Get files corresponding to data LST range
    start = np.argwhere(model_start_lsts<=data_lsts[0]).flatten()[-1]
    stop = np.argwhere(model_start_lsts>=data_lsts[-1]).flatten()
    # ^ not working if data_lst is eg [6.2823, 6.2830] since it doesnt want to grab the model at 0.00
    stop = 0 if len(stop)==0 else stop[0]  # fix for now (grab the first model file)
    
    return model_files[start:stop+1] if (start<stop) else np.concatenate([model_files[start:], model_files[:stop+1]]) 

def update_baselines(keep_baselines, ant_1_array, ant_2_array, edges, model_diff_matrices, model_src_matrices):
    """
    update models to keep only the given baselines
    ant_arrays and edges need to be the model arrays 
    keep_bl has size Nbls*2 (to match arrays with split real and imag parts)
    """
    new_edges = [0]
    keep_bl = []
    counter = 0  # counting num of good baselines

    # Go through redundant groups
    for start, stop in zip(edges//2,edges[1:]//2):
        ant_1_grp = ant_1_array[start:stop]
        ant_2_grp = ant_2_array[start:stop]

        # Go through bls in group
        for ai,aj in zip(ant_1_grp, ant_2_grp):
            if (ai,aj) in keep_baselines or (aj,ai) in keep_baselines:
                # keep_bl.append(True)
                keep_bl += 2 * [True]
                counter += 2
            else:  # dont keep
                keep_bl += 2 * [False]
                # keep_bl.append(False)
        # Update edges
        if counter != new_edges[-1]:
            new_edges.append(counter)

    # Update model matrices and ant arrays
    ant_1_array = ant_1_array.copy()[keep_bl[::2]]
    ant_2_array = ant_2_array.copy()[keep_bl[::2]]
    edges = np.array(new_edges)
    model_diff_matrices = model_diff_matrices.copy()[:,:,:,keep_bl]
    model_src_matrices = model_src_matrices.copy()[:,:,:,keep_bl]

    return ant_1_array, ant_2_array, edges, model_diff_matrices, model_src_matrices

def update_ants(ant_1_array, ant_2_array, edges, ants=[], keep=True):
    """
    update models to keep baselines with good antennas or remove bls with bad ants
    ants: list of ants to either keep or remove
    ant_arrays: size Nbls, 1st & 2nd antennas for each baseline
    keep: True to keep only the given ants, False to remove the given ants
    keep_bl has size Nbls*2 (to match arrays with split real and imag parts)
    """
    new_edges = [0]
    keep_bl = []
    counter = 0  # counting num of good baselines

    # Go through redundant groups
    for start, stop in zip(edges//2,edges[1:]//2):
        ant_1_grp = ant_1_array[start:stop]
        ant_2_grp = ant_2_array[start:stop]

        # Go through bls in group
        for ai,aj in zip(ant_1_grp, ant_2_grp):
            # keep (good) antennas
            if keep:
                if (ai in ants) and (aj in ants):
                    keep_bl += 2 * [True]
                    counter += 2
                else:
                    keep_bl += 2 * [False]
            # remove (bad) antennas
            else:
                if (ai in ants) or (aj in ants):
                    keep_bl += 2 * [False]
                else:
                    keep_bl += 2 * [True]
                    counter += 2
        
        # Update edges
        if counter != new_edges[-1]:
            new_edges.append(counter)

    # Update model matrices and ant arrays
    ant_1_array = ant_1_array.copy()[keep_bl[::2]]
    ant_2_array = ant_2_array.copy()[keep_bl[::2]]
    edges = np.array(new_edges)

    return keep_bl, ant_1_array, ant_2_array, edges

def interleaved_noise_variance_estimate(vis, kernel=[[1, -2, 1], [-2, 4, -2], [1, -2, 1]]):
    '''Estimate the noise on a visibility per frequency and time using weighted differencing of
    neighboring frequency and time samples.

    Arguments:
        vis: complex visibility waterfall, usually a numpy array of size (Ntimes, Nfreqs)
        kernel: differencing kernel for how to weight each visibility relative to its neighbors
            in time and frequency. Must sum to zero and must be 2D (either dimension could be length 1)

    Returns:
        variance: estimate of the noise variance on the input visibility with the same shape
    '''
    assert (np.sum(kernel) == 0), 'The kernal must sum to zero for difference-based noise estimation.'
    assert (np.array(kernel).ndim == 2), 'The kernel must be 2D.'
    
    variance = np.abs(convolve2d(vis, kernel, mode='same', boundary='wrap'))**2
    variance = variance / np.sum(np.array(kernel)**2)
    return variance

def update_gain_ants(bad_ants, gain_ants, gain_matrix, ant_ind_dict):
    """
    remove ants from the gain array

    bad_ants: set of antennas to remove
    gain_ants: list of current gain antennas
    """
    # Antenna info
    bad_inds = [ant_ind_dict[ant] for ant in bad_ants]
    keep_ant = np.full(gain_matrix.shape[-1], True)
    keep_ant[bad_inds] = False

    # Update gains
    gain_ants_new = gain_ants.copy()[keep_ant]
    gain_matrix_new = gain_matrix.copy()[:,:,keep_ant]

    return gain_ants_new, gain_matrix_new

def update_ant_inds(gain_ants, ant_1_array, ant_2_array):
    """
    create new ant indices arrays to match the updated ants in the gain array
    """
    # Make a dictionary mapping antenna numbers -> indices,
    ant_ind_dict = {ant:ind for ind,ant in enumerate(gain_ants)}

    # Ant indices arrays
    ant_1_indices = np.array([ant_ind_dict[ant] for ant in ant_1_array])
    ant_2_indices = np.array([ant_ind_dict[ant] for ant in ant_2_array])

    return ant_ind_dict, ant_1_indices, ant_2_indices



# GET INPUT ARGUMENTS
parser = argparse.ArgumentParser()

# Required
parser.add_argument("data_file", type=Path, help='Data file to calibrate')
parser.add_argument("model_cov_dir", type=Path, help='Directory with model files')
parser.add_argument("save_location", type=Path, help='Where to save the results')
# Optional
parser.add_argument(
    "-f", 
    "--freqs_to_calibrate", 
    type=float, 
    nargs=2, 
    metavar=('START_FREQ', 'END_FREQ'), 
    help='Frequency range to calibrate in MHz'
)
parser.add_argument(
    "-p", 
    "--pols_to_calibrate", 
    type=str, 
    nargs='+', 
    default=['nn','ee'], 
    choices=['xx','yy','nn','ee'], 
    help='Polarizations to calibrate'
)
parser.add_argument(
    "-F", 
    "--flag_yaml_file", 
    type=Path, 
    help='Flags'
)
parser.add_argument(
    "-g", 
    "--init_gains_file", 
    type=Path, 
    help='Initial guess for the gains'
)
parser.add_argument(
    "-l", 
    "--min_bl_length", 
    type=float, 
    default=0, 
    help='min baseline length'
)
parser.add_argument(
    "-L", 
    "--max_bl_length", 
    type=float, 
    default=1000, 
    help='max baseline length'
)
parser.add_argument(
    "-s", 
    "--min_group_size", 
    type=int, 
    default=1, 
    help='min redundant group size'
)
parser.add_argument(
    "-i", 
    "--maxiter", 
    type=int, 
    default=500,
    help='Maximum number of iterations (for minimizer)'
)
parser.add_argument(
    "-r", 
    "--reruns", 
    type=int, 
    default=1, 
    help='Number of times to rerun calibration'
)
parser.add_argument(
    "-a", 
    "--extra_bad_ants", 
    type=int, 
    nargs='+', 
    default=[], 
    help='Extra bad ants to remove right before calibration'
)

args = parser.parse_args()

# READ FILES
# Read data file
file_JD = '.'.join(str(args.data_file).split('.')[1:3])
uvdata_full = UVData.from_file(args.data_file, use_future_array_shapes=True)
times, sort = np.unique(uvdata_full.time_array, return_index=True) 
data_LST = uvdata_full.lst_array[sort]

# Get model files
model_files = get_models(data_LST, args.model_cov_dir)
# Read the model files into a list of dicts
model_dicts = [dict(np.load(file)) for file in model_files]

# Flags
if args.flag_yaml_file is not None:
    with open(args.flag_yaml_file, 'r') as f:
        flag_yaml_info = yaml.load(f.read(), Loader=yaml.SafeLoader)

# Read initial gains file
if args.init_gains_file is not None:
    init_gains_full = UVCal.from_file(args.init_gains_file, use_future_array_shapes=True)
# else: ?


# SELECT DATA
# Select frequencies to calibrate
if args.freqs_to_calibrate is not None:
    f_model = model_dicts[0]['freq_array']

    # Frequency channels to keep (only within input range)
    freqs_select_data = np.where(
        (args.freqs_to_calibrate[0] <= uvdata_full.freq_array/1e6) 
        & (uvdata_full.freq_array/1e6 <= args.freqs_to_calibrate[1])
    )[0]
    freqs_select_models = np.where(
        (args.freqs_to_calibrate[0] <= f_model/1e6) 
        & (f_model/1e6 <= args.freqs_to_calibrate[1])
    )[0]
    # Select in data
    uvdata_full.select(freq_chans=freqs_select_data)
    # Select in gains
    init_gains_full.select(freq_chans=freqs_select_data)
    # Select in models
    for model in model_dicts:
        # update freq_array
        model['freq_array'] = model['freq_array'][freqs_select_models]
        # update diffuse matrix  (Ntimes, Nfreqs, Npols, Nbls, Neig)
        model['diff_mat'] = model['diff_mat'][:,freqs_select_models]
        # update source matrix  (Ntimes, Nfreqs, Npols, Nbls, Nsrc)
        model['src_mat'] = model['src_mat'][:,freqs_select_models]

# Combine the model matrices
model_diff_matrices = np.concatenate(
    [model['diff_mat']   # (Ntimes, Nfreqs, Npols, Nbls, Neig)
    for model in model_dicts]
)
model_src_matrices = np.concatenate(
    [model['src_mat']  # (Ntimes, Nfreqs, Npols, Nbls, Nsrc)
    for model in model_dicts]
)
# model_matrices shape is now (Nmodelfiles*Ntimes, Nfreqs, Npols, Nbls, Neig or Nsrc)
# Model arrays & info
ant_1_array = model_dicts[0]['ant_1_array']
ant_2_array = model_dicts[0]['ant_2_array']
edges = model_dicts[0]['edges']
n_eig = model_dicts[0]['n_eig']

# Match the baselines in model and data (keep only bls that are in both)
data_baselines = set(zip(uvdata_full.ant_1_array, uvdata_full.ant_2_array))
# Keep model bls that are also in the data
ant_1_array, ant_2_array, edges, model_diff_matrices, model_src_matrices = update_baselines(
    keep_baselines=data_baselines, 
    ant_1_array=ant_1_array, 
    ant_2_array=ant_2_array, 
    edges=edges, 
    model_diff_matrices=model_diff_matrices, 
    model_src_matrices=model_src_matrices
)
model_baselines = set(zip(ant_1_array, ant_2_array))  # New model baselines
# keep data bls that are also in the model
keep_data_bls = []
for data_bl in zip(uvdata_full.ant_1_array, uvdata_full.ant_2_array):
    if data_bl in model_baselines or data_bl[::-1] in model_baselines:
        keep_data_bls.append(data_bl)
keep_data_bls = list(keep_data_bls)
uvdata_full.select(bls=keep_data_bls)

# Remove flagged antennas
if args.flag_yaml_file is not None:  # flags file given
    # Figure out which antennas to keep based on the antenna flags.
    data_ants = set(uvdata_full.ant_1_array).union(uvdata_full.ant_2_array)
    bad_ants = {ant for ant, _pol in flag_yaml_info["ex_ants"]}
    ants_to_keep = [
        ant for ant in uvdata_full.antenna_numbers 
        if ant in data_ants and ant not in bad_ants
    ]
    # Update data and gains
    uvdata_full.select(antenna_nums=ants_to_keep, inplace=True)
    init_gains_full.select(antenna_nums=ants_to_keep, inplace=True)
    # Updated matrices and arrays
    ants_to_keep = set(ants_to_keep)
    keep_bl, ant_1_array, ant_2_array, edges = update_ants(
        ant_1_array=ant_1_array,
        ant_2_array=ant_2_array, 
        edges=edges, 
        ants=ants_to_keep,
        keep=True
    )
    model_diff_matrices = model_diff_matrices[:,:,:,keep_bl]
    model_src_matrices = model_src_matrices[:,:,:,keep_bl]


# Check group size and baseline length
new_edges = [0]
keep_bl = np.full(2*len(ant_1_array), False)  # match size of model matrices
antpos, _ = uvdata_full.get_ENU_antpos()  # Antenna positions
# Go through redundant groups
for start, stop in zip(edges//2,edges[1:]//2):
    ant_1_grp = ant_1_array[start:stop]
    ant_2_grp = ant_2_array[start:stop]
    group_size = stop-start
    bl_len = np.linalg.norm(antpos[ant_1_grp[0]] - antpos[ant_2_grp[0]])  # using first bl of group to get bl len

    if (group_size >= args.min_group_size) and (bl_len >= args.min_bl_length) and (bl_len <= args.max_bl_length):  # keep group
        new_edges.append(new_edges[-1]+group_size*2)
        keep_bl[start*2:stop*2] = True

# Update model matrices and ant arrays
ant_1_array = ant_1_array[keep_bl[::2]]
ant_2_array = ant_2_array[keep_bl[::2]]
edges = np.array(new_edges)
model_diff_matrices = model_diff_matrices[:,:,:,keep_bl]
model_src_matrices = model_src_matrices[:,:,:,keep_bl]
# Update data & initial gains
ants_to_keep = list(set(ant_1_array).union(ant_2_array))
# uvdata_full.select(antenna_nums=ants_to_keep)
init_gains_full.select(antenna_nums=ants_to_keep)


# Go through polarizations
for pol in args.pols_to_calibrate:
    pol_index = 0 if (pol=='nn' or pol=='xx') else 1  # make more robust
    # Select pol to calibrate in uvdata & model
    uvdata = uvdata_full.select(polarizations=[pol], inplace=False)
    model_diff_matrices = model_diff_matrices[:,:,pol_index]
    model_src_matrices = model_src_matrices[:,:,pol_index]

    # Get data array
    data_all = np.array(
        [
            uvdata.get_data(ai, aj)
            for ai, aj in zip(ant_1_array, ant_2_array)
        ]
    )  # (N_bls, N_times, N_freqs), single pol

    # Get gains for all freqs and times
    init_gains_all = init_gains_full.gain_array[:,:,:,pol_index]  # all gains for chosen pol
    init_gains_all = np.swapaxes(init_gains_all,0,2)  # (N_times, N_freqs, N_ants)
    gain_ants = init_gains_full.ant_array
    # Make a dictionary mapping antenna numbers -> indices
    antind_dict = {ant:ind for ind,ant in enumerate(init_gains_full.ant_array)}
    # Ant indices arrays
    ant_1_indices = np.array([antind_dict[ant] for ant in ant_1_array])
    ant_2_indices = np.array([antind_dict[ant] for ant in ant_2_array])


    # INTERPOLATE
    # Model LSTs
    model_LST = np.concatenate([model['lst_array'] for model in model_dicts])
    model_LST_unwrapped = np.array([(m if m>=model_LST[0] else m+2*np.pi)for m in model_LST])
    # Interpolate
    interp_diff = interp1d(model_LST_unwrapped, model_diff_matrices, axis=0, kind='cubic')
    interp_src = interp1d(model_LST_unwrapped, model_src_matrices, axis=0, kind='cubic')
    # Unwrap data LSTs
    data_LST_unwrapped = np.array([(d if d>=data_LST[0] else d+2*np.pi)for d in data_LST])
    # Get model matrices at data's LSTs
    diff_matrix = interp_diff(data_LST_unwrapped)
    src_matrix = interp_src(data_LST_unwrapped)


    # NOISE MATRIX
    noise_matrix = np.array(
        [
            interleaved_noise_variance_estimate(data_bl) 
            for data_bl in data_all
        ]
    )  # (N_bls, N_times, N_freqs)
    # make values half and double elements along bls axis to match the models
    noise_matrix = np.repeat(noise_matrix/2, 2, axis=0)


    # REMOVING SOME ANTS (if given)
    bad_ants = set(args.extra_bad_ants)
    if len(bad_ants) > 0: 
        keep_bl, ant_1_array, ant_2_array, edges = update_ants(
            ant_1_array=ant_1_array, 
            ant_2_array=ant_2_array, 
            edges=edges, 
            ants=bad_ants, 
            keep=False
        )
        data_all = data_all[keep_bl[::2]]
        diff_matrix = diff_matrix[:,:,keep_bl]
        src_matrix = src_matrix[:,:,keep_bl]
        noise_matrix = noise_matrix[keep_bl]
        # update gains and antenna indices
        gain_ants, init_gains_all = update_gain_ants(
            bad_ants=bad_ants, 
            gain_ants=gain_ants, 
            gain_matrix=init_gains_all, 
            ant_ind_dict=antind_dict
        )
        antind_dict, ant_1_indices, ant_2_indices = update_ant_inds(
            gain_ants=gain_ants, 
            ant_1_array=ant_1_array, 
            ant_2_array=ant_2_array
        )


    # CALIBRATION
    result_x_all = []
    chain_all = []
    # not split (N__bls): data_all, ant_arrays, gains (match ant_ind)
    # split (2*N_bls): edges, models, noise
    
    # Go through LSTs
    for lst_index, lst in enumerate(data_LST):
        print(f'lst={lst:.3f} h')
        # Go through frequencies 
        for freq_index, freq in enumerate(uvdata.freq_array):
            print(f'freq={freq/1e6:.3f} MHz')
            # Get initial arrays for freq and lst (to modify)
            data = data_all.copy()[:,lst_index,freq_index]
            diff_mat = diff_matrix.copy()[lst_index,freq_index]
            src_mat = src_matrix.copy()[lst_index,freq_index]
            noise = noise_matrix.copy()[:,lst_index,freq_index]
            # gains = init_gains_all.copy()[lst_index,freq_index]
            gains = np.ones(init_gains_all.shape)[lst_index,freq_index]  # testing when init gains are all one
            # Make copies of arrays to modify in the reruns
            ant_1_arr = ant_1_array.copy()
            ant_2_arr = ant_2_array.copy()
            ant_1_ind = ant_1_indices.copy()
            ant_2_ind = ant_2_indices.copy()
            edges_copy = edges.copy()

            for r in range(args.reruns):
                # Sparse cov
                cov = corrcal.sparse.SparseCov(
                    noise=noise,
                    diff_mat=diff_mat,
                    src_mat=src_mat,
                    edges=edges_copy,
                    n_eig=n_eig,
                    isinv=False,
                )
                # Split up the gains into the real/imaginary parts for calibration.
                split_gains = np.zeros(2*gains.size, dtype=float)
                split_gains[::2] = gains.real
                split_gains[1::2] = gains.imag
                # Split up data into the real/imaginary parts for calibration.
                split_data = np.zeros(2*data.size, dtype=float)
                split_data[::2] = data.real
                split_data[1::2] = data.imag
        
                # Run corrcal
                gain_scale = 1  # ?
                phs_norm = 0.1  # ?
                opt_args = (
                    cov, split_data, ant_1_ind, ant_2_ind, gain_scale, phs_norm
                )
                # result = minimize(
                #     corrcal.optimize.nll,
                #     gain_scale*split_gains,
                #     args=opt_args,
                #     method="CG",
                #     jac=corrcal.optimize.grad_nll,
                #     options={"maxiter": maxiter}
                # )
                # result_x_all.append(result.x)
                fit_gains, fmin, nfev, njev, _, chain = fmin_cg(
                    corrcal.optimize.nll,
                    gain_scale*split_gains,
                    args=opt_args,
                    fprime=corrcal.optimize.grad_nll,
                    maxiter=args.maxiter,
                    full_output=True,
                    retall=True,
                )
                chain = np.array(chain)  # chain has shape (nit+1, 2*N_ants), so each row is the gains at each iteration
                chain_all.append(chain)
                result_x_all.append(fit_gains)

                # check conditions...
                # did we remove ants?
                # is it better without the removed ants?# get a list of bad ants?
                # bad_ants = {1,2,3,4,5}

                # update arrays to remove the bad ants/bls
                # put results for this time and freq somewhere




# CALIBRATION (some abs cal)?
# code is somewhere



# SAVE
    # gain solutions --> calfits
    # antennas thrown away
    # calibration time 

other_info = ".less_ants"
other_info += ".ones"
# need to go change line with gains in calibration loop when doing ones
npz_filename = str(args.save_location) + f'/{file_JD}' + other_info + ".npz"

np.savez(
    npz_filename,
    data=data_all,
    init_gains=init_gains_all,
    diff_matrix=diff_matrix,
    src_matrix=src_matrix,
    noise_matrix=noise_matrix,
    data_LST=data_LST,
    data_freqs=uvdata.freq_array/1e6,
    gain_ants=gain_ants,
    ant_1_array=ant_1_array,
    ant_2_array=ant_2_array,
    ant_1_indices=ant_1_indices,
    ant_2_indices=ant_2_indices,
    edges=edges,
    n_eig=n_eig, 
    chain_all=np.array(chain_all, dtype=object),
    result_x_all=result_x_all
)