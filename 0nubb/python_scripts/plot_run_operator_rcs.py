# use CMU Serif
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext'] = True

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.transforms import Bbox

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import root
import h5py
import os
import itertools
from utils import *

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from fittools import *
from formattools import *
style = styles['prd_twocol']

# Read out chiral extrapolation data
n_boot = 50
n_spacings = 2
n_ens_sp = [2, 3]    # 2 ensembles per lattice spacing
n_momenta = [8, 8, 8, 8, 8]
n_mom = n_momenta[0]
# avals = [0.11, 0.08]    # fm
ainv = [1.784, 2.382]       # GeV
Lat_24I = Lattice(24, 64)
Lat_32I = Lattice(32, 64)

Zq_extrap = [np.zeros((n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]
ZV_extrap = [np.zeros((n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]
ZA_extrap = [np.zeros((n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]
Z_extrap_lin = [np.zeros((5, 5, n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]

Zq_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
Zq_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
ZV_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
ZV_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
ZA_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
ZA_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
Z_extrap_mu = np.zeros((n_spacings, 5, 5, n_mom), dtype = np.float64)          # [24I/32I, i, j, q_idx]
Z_extrap_sigma = np.zeros((n_spacings, 5, 5, n_mom), dtype = np.float64)

chi_extrap_lin_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/24I/chiral_extrap/Z_extrap.h5', \
                    '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/32I/chiral_extrap/Z_extrap.h5']
for idx in range(n_spacings):
    print(chi_extrap_lin_paths[idx])
    f = h5py.File(chi_extrap_lin_paths[idx], 'r')
    Zq_extrap[idx] = np.real(f['Zq/values'][()])
    ZV_extrap[idx] = np.real(f['ZV/value'][()])
    ZA_extrap[idx] = np.real(f['ZA/value'][()])
    print(f['Z11'][()])
    for i, j in itertools.product(range(5), repeat = 2):
        key = 'Z' + str(i + 1) + str(j + 1)
        try:
            Z_extrap_lin[idx][i, j] = np.real(f[key][()])
        except:
            print('no key at ' + key)
for sp_idx in range(n_spacings):
    for mom_idx in range(n_mom):
        Zq_tmp = Superboot(n_ens_sp[sp_idx])
        Zq_tmp.boots = Zq_extrap[sp_idx][mom_idx]
        Zq_extrap_mu[sp_idx, mom_idx] = Zq_tmp.compute_mean()
        Zq_extrap_sigma[sp_idx, mom_idx] = Zq_tmp.compute_std()
        ZV_tmp = Superboot(n_ens_sp[sp_idx])
        ZV_tmp.boots = ZV_extrap[sp_idx][mom_idx]
        ZV_extrap_mu[sp_idx, mom_idx] = ZV_tmp.compute_mean()
        ZV_extrap_sigma[sp_idx, mom_idx] = ZV_tmp.compute_std()
        ZA_tmp = Superboot(n_ens_sp[sp_idx])
        ZA_tmp.boots = ZA_extrap[sp_idx][mom_idx]
        ZA_extrap_mu[sp_idx, mom_idx] = ZA_tmp.compute_mean()
        ZA_extrap_sigma[sp_idx, mom_idx] = ZA_tmp.compute_std()
        for i, j in itertools.product(range(5), repeat = 2):
            Z_tmp = Superboot(n_ens_sp[sp_idx])
            Z_tmp.boots = Z_extrap_lin[sp_idx][i, j, mom_idx]
            Z_extrap_mu[sp_idx, i, j, mom_idx] = Z_tmp.compute_mean()
            Z_extrap_sigma[sp_idx, i, j, mom_idx] = Z_tmp.compute_std()
k_list_chiral = f['momenta'][()]
f.close()
mom_list_24I = [Lat_24I.to_linear_momentum(k, datatype = np.float64) for k in k_list_chiral]
mom_list_32I = [Lat_32I.to_linear_momentum(k, datatype = np.float64) for k in k_list_chiral]
# mom_list_24I = [Lat_24I.to_lattice_momentum(k, datatype = np.float64) for k in k_list_chiral]
# mom_list_32I = [Lat_32I.to_lattice_momentum(k, datatype = np.float64) for k in k_list_chiral]
apsq_list_24I = [square(k) for k in mom_list_24I]
apsq_list_32I = [square(k) for k in mom_list_32I]
mu_list_24I = [ainv[0] * np.sqrt(square(p)) for p in mom_list_24I]
mu_list_32I = [ainv[1] * np.sqrt(square(p)) for p in mom_list_32I]
musq_list_24I = [mu ** 2 for mu in mu_list_24I]
musq_list_32I = [mu ** 2 for mu in mu_list_32I]
print('24I mu list: ' + str(mu_list_24I))
print('32I mu list: ' + str(mu_list_32I))

sp_colors = colors[:2]
sp_labels = ['a = 0.11 fm', 'a = 0.08 fm']
xlimits = [[1.0, 4.1], [1.0, 4.4]]
asp_ratio = 4/3
def plot_rcs_raw_apsq(cvs, sigmas, ylabel, path, sub_momlist = [mom_list_24I, mom_list_32I]):
    """
    Plots data with central values cvs and error sigmas. Uses a subset of the momenta
    sub_momlist, if not passed in then defaults to the entire momentum list.
    """
    with sns.plotting_context('paper'):
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        for idx in range(n_spacings):
            _, caps, _ = plt.errorbar([square(k) for k in sub_momlist[idx]], cvs[idx], sigmas[idx], fmt = '.', c = sp_colors[idx], \
                    label = sp_labels[idx], capsize = style['endcaps'], markersize = style['markersize'], \
                    elinewidth = style['ebar_width'])
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('$(ap)^2$', fontsize = style['fontsize'])
        plt.ylabel(ylabel, fontsize = style['fontsize'])
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

def plot_rcs_raw(sp_idx, cvs, sigmas, ylabel, ylimits, path, x_axis = [mu_list_24I, mu_list_32I]):
    """
    Plots data with central values cvs and error sigmas. Uses a subset of the energy scales
    sub_mulist, if not passed in then defaults to the entire momentum list.
    """
    with sns.plotting_context('paper'):
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        _, caps, _ = plt.errorbar(x_axis[sp_idx], cvs[sp_idx], sigmas[sp_idx], fmt = '.', c = sp_colors[sp_idx], \
                label = sp_labels[sp_idx], capsize = style['endcaps'], markersize = style['markersize'], \
                elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('$\\mu\;(\\mathrm{GeV})$', fontsize = style['fontsize'])
        plt.ylabel(ylabel, fontsize = style['fontsize'])
        plt.xlim(xlimits[sp_idx])
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

# mark μ0 with a red 'x', and put a dashed line at μ1
