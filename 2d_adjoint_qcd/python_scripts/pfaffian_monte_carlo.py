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
eps = 0.5                               # spread around 1 to generate group elements.
kappa = 0.2                             # hopping parameter
L, T = 6, 6                             # lattice size
# L, T = 8, 8                             # lattice size
bcs = (1, -1)                           # boundary conditions

# n_samps = 20
n_samps = 10000                           # total samples to compute
n_logs = 100                             # number of times to log while running
log_idx = n_samps / n_logs              # index to mod by to log

np.random.seed(20)

nprocs = multiprocessing.cpu_count()    # number of processes

def params_to_fname(Nc, L, T, n_samps, eps, kappa):
    """Converts a set of parameters to a string to use for a file name."""
    eps_str = fmt.format_float(eps)
    kappa_str = fmt.format_float(kappa)
    return f'pf_Nc{Nc}_L{L}_T{T}_n{n_samps}_e{eps_str}_k{kappa_str}'
out_dir = '/Users/theoares/Dropbox (MIT)/research/2d_adjoint_qcd/meas/monte_carlo_pf/'
folder = params_to_fname(Nc, L, T, n_samps, eps, kappa)
data_path = out_dir + folder
os.mkdir(data_path)
fname = data_path + '/data.h5'
print(f'Saving data to path: {fname}')

gens = rhmc.get_generators(Nc)
Lat = rhmc.Lattice(L, T)

# TODO use MPI instead of multiprocessing
# def gen_cfgs(n_cfgs):
#     pf_list = np.zeros((n_samps), dtype = np.complex128)
#     proc_id = os.getpid()
#     for i in range(n_cfgs):
#         U = rhmc.gen_random_fund_field(Nc, eps, lat = Lat)
#         V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
#         D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
#         Q = rhmc.hermitize_dirac(D)
#         pf = rhmc.pfaffian(Q)
#         if i % log_idx == 0:
#             print(f'[{proc_id}] Random sample {i}, pf(D) = {pf}')
#         pf_list[i] = pf
#     return pf_list

# pf_list = np.zeros((n_samps), dtype = np.complex128)
# if name == '__main__':
#     # TODO multithread this with nprocs processes
    
#     for i in range(n_samps):
#         U = rhmc.gen_random_fund_field(Nc, eps, lat = Lat)
#         V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
#         D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
#         Q = rhmc.hermitize_dirac(D)
#         pf = rhmc.pfaffian(Q)
#         if i % log_idx == 0:
#             print(f'Random sample {i}, pf(D) = {pf}')
#         pf_list[i] = pf

# Standard unparallelized computation
pf_list = np.zeros((n_samps), dtype = np.complex128)
for i in range(n_samps):
    U = rhmc.gen_random_fund_field(Nc, eps, lat = Lat)
    V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
    D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
    Q = rhmc.hermitize_dirac(D)
    pf = rhmc.pfaffian(Q)
    if i % log_idx == 0:
        print(f'Random sample {i}, pf(D) = {pf}')
    pf_list[i] = pf

f = h5py.File(fname, 'w')
f['pf'] = pf_list
f.close()
print(f'Pfaffian data written to: {fname}.')

print(f'Time for {n_samps} iterations: {time.time() - start_time} seconds.')
