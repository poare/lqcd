################################################################################
# Monte Carlo implementation to see what the Pfaffian of the Dirac operator    #
# D will look like. Note that configurations will be sampled randomly, and     #
# there is a chance that exceptional configurations that make the Pfaffian     #
# appear ill-determined do not really come up in the path integral with any    # 
# non-zero measure.                                                            #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

n_boot = 100
import numpy as np
import h5py
import os
import time
import multiprocessing

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
import formattools as fmt

import rhmc

start_time = time.time()

Nc = 2                                  # number of colors
kappa = 0.2                             # hopping parameter
L, T = 4, 4                             # lattice size
# L, T = 6, 6                             # lattice size
# L, T = 8, 8                             # lattice size
bcs = (1, -1)                           # boundary conditions

n_samps = 100                           # total samples to compute

# n_samps = 20
# n_logs = 10                             # number of times to log while running
# log_idx = n_samps / n_logs              # index to mod by to log

# np.random.seed(20)
np.random.seed(10)

serial = False
nprocs = 10
# nprocs = multiprocessing.cpu_count()    # number of processes

def params_to_fname(Nc, L, T, n_samps, kappa):
    """Converts a set of parameters to a string to use for a file name."""
    kappa_str = fmt.format_float(kappa)
    return f'pf_Nc{Nc}_L{L}_T{T}_n{n_samps}_k{kappa_str}'
out_dir = '/Users/theoares/Dropbox (MIT)/research/2d_adjoint_qcd/meas/monte_carlo_pf/'
folder = params_to_fname(Nc, L, T, n_samps, kappa)
data_path = out_dir + folder
fname = data_path + '/data.h5'
# if os.path.isdir(data_path):
#     os.rename(data_path, f'{data_path}/data_old.h5')
# else:
#     os.mkdir(data_path)
if not os.path.isdir(data_path):
    os.mkdir(data_path)
print(f'Saving data to path: {fname}')

gens = rhmc.get_generators(Nc)
Lat = rhmc.Lattice(L, T)

def gen_cfgs(n_cfgs, par = True, n_logs = 10):
    """
    Generates n_cfgs configurations U and computes the Pfaffian of D[U] for each 
    configuration U. If parellelized, sets the seed to be the process ID, otherwise 
    does not seed the RNG. 

    Parameters
    ----------
    n_cfgs : int
        Number of configurations to generate and compute the Pfaffian on.
    par : bool (default = True)
        Whether the code is run in serial (par = False) or parallel (par = True).
    
    Returns
    -------
    np.array [n_cfgs]
        Pfaffian of the Dirac operator, computed on n_cfgs configurations.
    """
    if par:
        proc_id = os.getpid()
        np.random.seed(proc_id)
    pf_list = np.zeros((n_cfgs), dtype = np.complex128)
    log_idx = n_cfgs / n_logs              # index to mod by to log
    for i in range(n_cfgs):
        U = rhmc.gen_random_fund_field(Nc, lat = Lat)
        V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
        D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
        Q = rhmc.hermitize_dirac(D)
        pf = rhmc.pfaffian(Q)
        if i % log_idx == 0:
            if par:
                print(f'[{proc_id}] Random sample {i}, pf(D) = {pf}')
            else:
                print(f'Random sample {i}, pf(D) = {pf}')
        pf_list[i] = pf
    return pf_list

if __name__ == '__main__':
    if serial:
        pf_list = gen_cfgs(n_samps, par = False)
    else:
        n_cfgs = n_samps // nprocs
        with multiprocessing.Pool(nprocs) as p:
            pf_list = p.map(gen_cfgs, [n_cfgs for n in range(nprocs)])
        pf_list = np.array(pf_list)
        print(pf_list.shape)


#     # TODO multithread this with nprocs processes
    
#     for i in range(n_samps):
#         U = rhmc.gen_random_fund_field(Nc, lat = Lat)
#         V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
#         D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
#         Q = rhmc.hermitize_dirac(D)
#         pf = rhmc.pfaffian(Q)
#         if i % log_idx == 0:
#             print(f'Random sample {i}, pf(D) = {pf}')
#         pf_list[i] = pf

# Standard unparallelized computation
# pf_list = np.zeros((n_samps), dtype = np.complex128)
# for i in range(n_samps):
#     U = rhmc.gen_random_fund_field(Nc, lat = Lat)
#     V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
#     D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
#     Q = rhmc.hermitize_dirac(D)
#     pf = rhmc.pfaffian(Q)
#     if i % log_idx == 0:
#         print(f'Random sample {i}, pf(D) = {pf}')
#     pf_list[i] = pf

f = h5py.File(fname, 'w')
f['pf'] = pf_list
f.close()
print(f'Pfaffian data written to: {fname}.')

print(f'Time for {n_samps} iterations: {time.time() - start_time} seconds.')
