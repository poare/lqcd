import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import h5py
import os

# Reads in an old file format.
def read_text(file):
    f = open(file, 'r')
    # New read in format: Each line after the first of form cfgId|a1+ib a2+ib ...
    f.readline()
    for i, line in enumerate(f):
        cur_line = np.zeros(L3, dtype=complex)
        cfgIdx, rest = line.split('|')
        cfgIdx = int(cfgIdx)
        interps = rest.split(' ')    # now get real and complex
        for n_t, x in enumerate(interps):
            if not x.isspace():
                real, imag = x.split('+i(')    #form should be x = a+i(b)
                real = float(real)
                imag = float(imag.replace(')', ''))
                cur_line[n_t] = complex(real, imag)
        cur_line = [cur_line]
        if C == []:
            C = np.array(cur_line)
        else:
            C = np.append(C, cur_line, axis = 0)
    return C

# directory should contain hdf5 files.
def read_h5(directory, mom = False):
    files = []
    for (dirpath, dirnames, file) in os.walk(directory):
        files.extend(file)
    C = {}
    for file in files:
        path_to_file = directory + '/' + file
        f = h5py.File(path_to_file, 'r')
        config_id = str([x for x in f['twopt']][0])
        if mom:
            correlators = f['twopt/' + config_id]
            p = [x for x in correlators.keys()]
        else:
            correlators = f['twopt']
            p = [0]
        for momenta in p:
            if mom:
                data = [x for x in correlators[momenta]]
            else:
                data = [x for x in correlators[config_id]]
            if momenta in C:
                C[momenta] = np.vstack([C[momenta], data])
            else:
                C[momenta] = np.array(data)
        # for i, data in correlators.items():
        #     if len(C) == 0:
        #         C = np.array(data)
        #     else:
        #         C = np.vstack([C, data])
    return C

# Bootstraps a set of correlation functions.
def bootstrap(C, n_boot = 500):
    num_configs = C.shape[0]
    n_t = C.shape[1]
    samples = np.zeros((n_boot, num_configs, n_t), dtype = complex)
    for i in range(n_boot):
        cfgIds = np.random.choice(num_configs, num_configs)    #Configuration ids to pick
        samples[i, :, :] = C[cfgIds, :]
    return samples

# Returns the mean and standard deviation of effective mass ensemble. Ensemble average should
# be an ensemble of n_boot averages at each time slice, where the average is over the
# bootstrapped sample. Returns effective mass computed at each time slice.
def get_effective_mass(ensemble_avg):
    ratios = np.abs(ensemble_avg / np.roll(ensemble_avg, shift = -1, axis = 1))[:, :-1]
    m_eff_ensemble = np.log(ratios)
    m_eff = np.mean(m_eff_ensemble, axis = 0)
    sigma = np.std(m_eff_ensemble, axis = 0, ddof = 1)
    return m_eff, sigma

# Returns the mean and standard deviation of cosh corrected effective mass ensemble.
def get_cosh_effective_mass(ensemble_avg, N = 48):
    ratios = np.abs(ensemble_avg / np.roll(ensemble_avg, shift = -1, axis = 1))[:, :-1]
    m_eff_ensemble = np.log(ratios)
    cosh_m_eff_ensemble = np.zeros(ratios.shape)
    for ens_idx in range(ratios.shape[0]):
        for t in range(ratios.shape[1]):
            m = root(lambda m : ratios[ens_idx, t] - np.cosh(m * (t - N / 2)) / np.cosh(m * (t + 1 - N / 2)), \
                         m_eff_ensemble[ens_idx, t])
            cosh_m_eff_ensemble[ens_idx, t] = m.x
    m_cosh = np.mean(cosh_m_eff_ensemble, axis = 0)
    sigma = np.std(cosh_m_eff_ensemble, axis = 0, ddof = 1)
    return m_cosh, sigma

# Flips half the data to negative, for use in effective mass calculation.
def flip_half_data(data, n_t):
    half = n_t // 2
    flipped = np.concatenate([data[:half], -data[half:]])
    return flipped

# Fits the effective mass data at each time slice to a constant to retrieve the
# (scalar) effective mass. Assumes the effective mass flips from negative to positive
# at the halfway time slice. TODO: can play around with exponential decay to make better.
def extract_mass(fit_region, eff_mass):
    m = np.polyfit(fit_region, eff_mass[fit_region], 0)[0]
    return m

# Determines how the error at base_time scales as we increase the number of
# samples used in the computation. C is the set of two point correlators.
# n_start and n_step are the configuration numbers to start and end at, and
# n_step is the number of steps to take between different configuration numberes.
# To see pictorally, plot returned cfg_list versus err
def error_analysis(C, base_time, n_start, n_step):
    num_configs = C.shape[0]
    cfg_list = range(n_start, num_configs, n_step)
    err = np.zeros(len(cfg_list))
    means = np.zeros(len(cfg_list))
    for i, n in enumerate(cfg_list):    # sample n configurations from C
        config_ids = np.random.choice(num_configs, n, replace = False)
        C_sub = C[config_ids, :]    #now get error on the subsampled C
        subensemble = bootstrap(C_sub)
        subensemble_avg = np.mean(subensemble, axis = 1)
        μ = np.abs(np.mean(subensemble_avg, axis = 0))
        σ = np.abs(np.std(subensemble_avg, axis = 0))
        err[i] = σ[base_time]
        means[i] = μ[base_time]
    return cfg_list, err, means
