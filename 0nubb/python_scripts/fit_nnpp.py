# use CMU Serif
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.transforms import Bbox

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import h5py
import os
import itertools

from utils import *

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from plottools import *
from fittools import *
from formattools import *
style = styles['prd_twocol']

fpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_gamma.h5'
F = h5py.File(fpath, 'r')
Lat = Lattice(32, 48)
a = 0.145  # fm, placeholder for now
ainv = hbarc / a
mpi = 0.8    # MeV, placeholder for now
k_list = F['momenta'][()]
mom_list = np.array([Lat.to_linear_momentum(k, datatype=np.float64) for k in k_list])
mu_list = np.array([get_energy_scale(q, a, Lat) for q in k_list])
apsq_list = np.array([square(k) for k in mom_list])
Zq = np.real(F['Zq'][()])
n_momenta, n_boot = Zq.shape[0], Zq.shape[1]
ZV = np.real(F['ZV'][()])
ZA = np.real(F['ZA'][()])
Z = np.zeros((5, 5, n_momenta, n_boot), dtype = np.float64)
for i, j in itertools.product(range(5), repeat = 2):
    key = 'Z' + str(i + 1) + str(j + 1)
    Z[i, j] = np.real(F[key][()])
ZbyZVsq = np.einsum('ijqb,qb->ijqb', Z, 1 / (ZV ** 2))

Zq_mu = np.mean(Zq, axis = 1)
Zq_sigma = np.std(Zq, axis = 1, ddof = 1)
ZV_mu = np.mean(ZV, axis = 1)
ZV_sigma = np.std(ZV, axis = 1, ddof = 1)
ZA_mu = np.mean(ZA, axis = 1)
ZA_sigma = np.std(ZA, axis = 1, ddof = 1)
Z_mu = np.mean(Z, axis = 3)
Z_sigma = np.std(Z, axis = 3, ddof = 1)
ZbyZVsq_mu = np.mean(ZbyZVsq, axis = 3)
ZbyZVsq_sigma = np.std(ZbyZVsq, axis = 3, ddof = 1)

apsq = True         # x axis to plot against. If apsq, plots against (ap)^2, else plots against mu^2.
x_axis = apsq_list if apsq else mu_list
xlabel = '$(ap)^2$' if apsq else '$\\mu\;(\\mathrm{GeV})$'
xlimits = [0.0, 6.5] if apsq else [1.0, 4.1]
asp_ratio = 4/3

# generate plot of raw data
def plot_rcs_raw(cvs, sigmas, ylabel, ylimits, path, col = 'r'):
    """
    Plots data with central values cvs and error sigmas. Uses a subset of the energy scales
    sub_mulist, if not passed in then defaults to the entire momentum list.
    """
    with sns.plotting_context('paper'):
        # n_plts = cvs.shape[0]    # pass in list of plots
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        # overload to plot multiple
        _, caps, _ = plt.errorbar(x_axis, cvs, sigmas, fmt = '.', c = col, \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel(xlabel, fontsize = style['fontsize'])
        plt.ylabel(ylabel, fontsize = style['fontsize'])
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Plot ' + ylabel + ' saved at: \n   ' + path)

# generate multiple plots in same figure. TODO
def plot_rcs_raw_many(cvs, sigmas, ylabel, ylimits, path, col_list = ['r']):
    """
    Plots data with central values cvs and error sigmas. Uses a subset of the energy scales
    sub_mulist, if not passed in then defaults to the entire momentum list.
    """
    with sns.plotting_context('paper'):
        n_plts = cvs.shape[0]    # pass in list of plots
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        # overload to plot multiple
        for ii in range(n_plts):
            _, caps, _ = plt.errorbar(x_axis, cvs[ii], sigmas[ii], fmt = '.', c = col_list[ii], \
                    capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel(xlabel, fontsize = style['fontsize'])
        plt.ylabel(ylabel, fontsize = style['fontsize'])
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Plot ' + ylabel + ' saved at: \n   ' + path)

# /Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/raw/ZqVA for Zq, ZV, ZA
# Zq
Zq_range = np.array([0.7, 0.86])
plot_rcs_raw(Zq_mu, Zq_sigma, '$\mathcal{Z}_q^\mathrm{RI}$', Zq_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/raw/ZqVA/Zq_RI.pdf')

# ZV
ZV_range = np.array([0.6, 0.82])
plot_rcs_raw(ZV_mu, ZV_sigma, '$\mathcal{Z}_V$', ZV_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/raw/ZqVA/ZV.pdf')

# ZA
ZA_range = np.array([0.72, 0.88])
plot_rcs_raw(ZA_mu, ZA_sigma, '$\mathcal{Z}_A$', ZA_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/raw/ZqVA/ZA.pdf')

Z_range = np.array([
    [[0.45, 0.70], [-0.12, 0.0], [-0.012, 0.0], [0.0, 0.05], [0.005, 0.022]],
    [[-0.10, 0.0], [0.45, 0.70], [0.06, 0.14], [-0.10, 0.0], [0.0, 0.004]],
    [[0.0, 0.015], [0.0, 0.02], [0.28, 0.72], [0.0, 0.16], [-0.005, 0.0]],
    [[0.001, 0.011], [-0.002, 0.001], [0.0, 0.12], [0.35, 0.70], [-0.016, -0.002]],
    [[0.005, 0.02], [-0.003, 0.003], [0.0, 0.06], [-0.12, -0.05], [0.55, 0.75]]
])
ZbyZVsq_range = np.array([
    [[1.05, 1.35], [-0.30, 0.0], [-0.032, 0.0], [0.0, 0.15], [0.0, 0.08]],
    [[-0.25, 0.0], [1.05, 1.3], [0.10, 0.35], [-0.30, 0.0], [0.0, 0.008]],
    [[0.0, 0.04], [0.0, 0.035], [0.8, 1.05], [0.0, 0.4], [-0.01, 0.0]],
    [[0.0, 0.03], [-0.003, 0.001], [0.0, 0.3], [1.01, 1.07], [-0.03, -0.005]],
    [[0.0, 0.06], [-0.008, 0.005], [0.0, 0.15], [-0.32, -0.08], [1.1, 1.6]]
])

for n, m in itertools.product(range(5), repeat = 2):
    plot_rcs_raw(Z_mu[n, m], Z_sigma[n, m], '$\mathcal{Z}_{' + str(n + 1) + str(m + 1) + '}$', Z_range[n, m], \
                    '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/raw/Zops/Z' + str(n + 1) + str(m + 1) + '.pdf')
    plot_rcs_raw(ZbyZVsq_mu[n, m], ZbyZVsq_sigma[n, m], '$\mathcal{Z}_{' + str(n + 1) + str(m + 1) + '} / \mathcal{Z}_V^2$', ZbyZVsq_range[n, m], \
                    '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/raw/ZopsbyZVsq/Z' + str(n + 1) + str(m + 1) + '.pdf')

# Start fitting. Use fit forms here.
all_fit_forms = [
    Model(lambda params : lambda x : params[0], 1),         # const
]

# Output file to save data to
outpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/fits.h5'
fout = h5py.File(outpath, 'w')

# Zq


fout.close()
