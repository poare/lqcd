################################################################################
# Processes data for Pfaffian Monte Carlo, which is generated in the script    #
# pfaffian_monte_carlo.py. Each dataset is stored in a directory labeled with  # 
# the given parameters of the Monte Carlo process, where the folder name is    #
# formatted as the following Python fstring:                                   #
#        f'pf_Nc{Nc}_L{L}_T{T}_n{n_samps}_e{eps}_k{kappa}'                     #
# where Nc is the number of colors, (L, T) is the lattice shape, n_samps is    #
# the number of Monte Carlo samples, eps is the epsilon parameter used to      # 
# generate a random spread of SU(Nc) matrices, and kappa is the hopping        #
# parameter. Outputs a histogram of the Pfaffian data to the same directory.   #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import os
import itertools

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from formattools import *
import plottools as pt

style = styles['prd_twocol']
pt.set_font()

import rhmc

parent = '/Users/theoares/Dropbox (MIT)/research/2d_adjoint_qcd/meas/monte_carlo_pf/'
dir_name = 'pf_Nc2_L4_T4_n10000_k0p20'
# dir_name = 'pf_Nc2_L8_T8_n1500_k0p20'
# dir_name = 'pf_Nc2_L6_T6_n10000_e0p50_k0p20'
data_dir = parent + dir_name
fname = data_dir + '/data.h5'
out_name = data_dir + '/hist.pdf'
print(f'Directory name: {dir_name}')
print(f'Reading data from: {fname}')

def strip_dir_name(dir_name):
    """Strips the directory name f'pf_Nc{Nc}_L{L}_T{T}_n{n_samps}_e{eps}_k{kappa}' to extract 
    the parameters Nc, L, T, n_samps, eps, and kappa."""
    assert dir_name[:3] == 'pf_', 'Wrong directory name input.'
    tokens = dir_name.split('_')[1:]                    # tokens = ['{Nc}', '{L}', '{T}', '{n_samps}', '{kappa}']
    # labels = ['Nc', 'L', 'T', 'n', 'e', 'k']
    # casters = [int, int, int, int, float, float]
    labels = ['Nc', 'L', 'T', 'n', 'k']
    casters = [int, int, int, int, float]    # casting function for each piece
    for i, cast in enumerate(casters):
        if cast == float:
            tokens[i] = ''.join([c if c != 'p' else '.' for c in tokens[i]])
    output = [casters[i](tokens[i].split(labels[i])[1]) for i in range(len(casters))]
    return tuple(output)

# Nc, L, T, n_samps, eps, kappa = strip_dir_name(dir_name)
# print(f'Parameters: Nc = {Nc}, L = {L}, T = {T}, n_samps = {n_samps}, eps = {eps}, kappa = {kappa}.')
Nc, L, T, n_samps, kappa = strip_dir_name(dir_name)
print(f'Parameters: Nc = {Nc}, L = {L}, T = {T}, n_samps = {n_samps}, kappa = {kappa}.')
f = h5py.File(fname, 'r')
pf = f['pf'][()]
assert np.allclose(np.imag(pf), np.zeros((len(pf)), dtype = np.float64)), 'Pfaffian samples have imaginary component(s).'
pf = np.real(pf)

n_neg = len(pf[pf < 0])
print(f'Number of configurations with negative Pfaffians: {n_neg}.')
print(f'Locations of negative Pfaffians: {np.where(pf < 0)}.')
print(f'Values of negative Pfaffians: {pf[pf < 0]}')

min_pos, max_pos = np.min(pf[pf > 0]), np.max(pf[pf > 0])
print(f'Positive data has support ({min_pos}, {max_pos}).')
if n_neg > 0:
    print(f'Negative data has support ({np.min(pf[pf < 0])}, {np.max(pf[pf < 0])}).')

# rg = (0.0, 2.5)
# rg = (min_pos, max_pos)
rg = (0.7, 1.35)
fig, axes = pt.add_subplots()
ax = axes[0]
# ax.hist(pf, bins = 100, range = rg, color = pt.pal[3])
ax.hist(pf, bins = 100, range = rg, color = pt.pal[0])
pt.add_xlabel(ax, r'$\mathrm{pf}(D_W)$')
pt.add_ylabel(ax, 'Counts')
pt.add_title(ax, str(n_samps) + r' Random Pfaffian Samples on $'+str(L)+r' \times '+str(T)+r'$ Lattice with $N_c = '+str(Nc) + r'$.')
pt.save_figure(out_name)
print(f'Figure saved to: {out_name}')
