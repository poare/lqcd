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
n_samp = n_boot
n_spacings = 2
n_ens_sp = [2, 3]    # 2 ensembles per lattice spacing
n_momenta = [8, 8, 8, 8, 8]
n_mom = n_momenta[0]
# avals = [0.11, 0.08]    # fm
ainv = [1.784, 2.382]       # GeV
Lat_24I = Lattice(24, 64)
Lat_32I = Lattice(32, 64)

# Get value of ZA from DWF analysis. Change extrapolation to physical point? Probably keep it at chiral limit
ZA_result_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/24I/results/ZA.h5', \
        '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/32I/results/ZA.h5']
f_ZAout = [h5py.File(ZA_path, 'r') for ZA_path in ZA_result_paths]
ZA0 = [f['ZA'][()] for f in f_ZAout]
[f.close() for f in f_ZAout]
ZA0_24I, ZA0_32I = Superboot(2), Superboot(3)
ZA0_24I.boots, ZA0_32I.boots = ZA0[0], ZA0[1]
ZA0_mu = [ZA0_24I.compute_mean(), ZA0_32I.compute_mean()]
ZA0_std = [ZA0_24I.compute_std(), ZA0_32I.compute_std()]
print('Extrapolation from RBC/UKQCD values: ')
print('24I: ' + str(ZA0_mu[0]) + ' \pm ' + str(ZA0_std[0]))
print('32I: ' + str(ZA0_mu[1]) + ' \pm ' + str(ZA0_std[1]))

# Zq_extrap = [np.zeros((n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]
# ZV_extrap = [np.zeros((n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]
# ZA_extrap = [np.zeros((n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]
# Z_extrap_lin = [np.zeros((5, 5, n_mom, n_ens_sp[i], n_boot), dtype = np.float64) for i in range(n_spacings)]

# Zq_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
# Zq_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZV_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZV_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZA_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZA_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
# Z_extrap_mu = np.zeros((n_spacings, 5, 5, n_mom), dtype = np.float64)          # [24I/32I, i, j, q_idx]
# Z_extrap_sigma = np.zeros((n_spacings, 5, 5, n_mom), dtype = np.float64)

Zq_extrap = np.zeros((n_spacings, n_mom, n_boot), dtype = np.float64)
ZV_extrap = np.zeros((n_spacings, n_mom, n_boot), dtype = np.float64)
ZA_extrap = np.zeros((n_spacings, n_mom, n_boot), dtype = np.float64)
Z_extrap = np.zeros((n_spacings, 5, 5, n_mom, n_boot), dtype = np.float64)

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
            Z_extrap[idx, i, j] = np.real(f[key][()])
        except:
            print('no key at ' + key)
print(Zq_extrap.shape)
Zq_extrap_mu = np.mean(Zq_extrap, axis = 2)
Zq_extrap_sigma = np.std(Zq_extrap, axis = 2, ddof = 1)
ZV_extrap_mu = np.mean(ZV_extrap, axis = 2)
ZV_extrap_sigma = np.std(ZV_extrap, axis = 2, ddof = 1)
ZA_extrap_mu = np.mean(ZA_extrap, axis = 2)
ZA_extrap_sigma = np.std(ZA_extrap, axis = 2, ddof = 1)
Z_extrap_mu = np.mean(Z_extrap, axis = 4)
Z_extrap_sigma = np.std(Z_extrap, axis = 4, ddof = 1)

# ZV_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZV_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZA_extrap_mu = np.zeros((n_spacings, n_mom), dtype = np.float64)
# ZA_extrap_sigma = np.zeros((n_spacings, n_mom), dtype = np.float64)
# Z_extrap_mu = np.zeros((n_spacings, 5, 5, n_mom), dtype = np.float64)          # [24I/32I, i, j, q_idx]
# Z_extrap_sigma = np.zeros((n_spacings, 5, 5, n_mom), dtype = np.float64)

# for sp_idx in range(n_spacings):
#     for mom_idx in range(n_mom):
#         Zq_tmp = Superboot(n_ens_sp[sp_idx])
#         Zq_tmp.boots = Zq_extrap[sp_idx][mom_idx]
#         Zq_extrap_mu[sp_idx, mom_idx] = Zq_tmp.compute_mean()
#         Zq_extrap_sigma[sp_idx, mom_idx] = Zq_tmp.compute_std()
#         ZV_tmp = Superboot(n_ens_sp[sp_idx])
#         ZV_tmp.boots = ZV_extrap[sp_idx][mom_idx]
#         ZV_extrap_mu[sp_idx, mom_idx] = ZV_tmp.compute_mean()
#         ZV_extrap_sigma[sp_idx, mom_idx] = ZV_tmp.compute_std()
#         ZA_tmp = Superboot(n_ens_sp[sp_idx])
#         ZA_tmp.boots = ZA_extrap[sp_idx][mom_idx]
#         ZA_extrap_mu[sp_idx, mom_idx] = ZA_tmp.compute_mean()
#         ZA_extrap_sigma[sp_idx, mom_idx] = ZA_tmp.compute_std()
#         for i, j in itertools.product(range(5), repeat = 2):
#             Z_tmp = Superboot(n_ens_sp[sp_idx])
#             Z_tmp.boots = Z_extrap_lin[sp_idx][i, j, mom_idx]
#             Z_extrap_mu[sp_idx, i, j, mom_idx] = Z_tmp.compute_mean()
#             Z_extrap_sigma[sp_idx, i, j, mom_idx] = Z_tmp.compute_std()

# chi_extrap_lin_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/24I/chiral_extrap/Z_extrap_old.h5', \
#                     '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/32I/chiral_extrap/Z_extrap_old.h5']
# for idx in range(n_spacings):
#     print(chi_extrap_lin_paths[idx])
#     f = h5py.File(chi_extrap_lin_paths[idx], 'r')
#     Zq_extrap[idx] = np.real(f['Zq/values'][()])
#     ZV_extrap[idx] = np.real(f['ZV/value'][()])
#     ZA_extrap[idx] = np.real(f['ZA/value'][()])
#     print(f['Z11'][()])
#     for i, j in itertools.product(range(5), repeat = 2):
#         key = 'Z' + str(i + 1) + str(j + 1)
#         try:
#             Z_extrap_lin[idx][i, j] = np.real(f[key][()])
#         except:
#             print('no key at ' + key)
# for sp_idx in range(n_spacings):
#     for mom_idx in range(n_mom):
#         Zq_tmp = Superboot(n_ens_sp[sp_idx])
#         Zq_tmp.boots = Zq_extrap[sp_idx][mom_idx]
#         Zq_extrap_mu[sp_idx, mom_idx] = Zq_tmp.compute_mean()
#         Zq_extrap_sigma[sp_idx, mom_idx] = Zq_tmp.compute_std()
#         ZV_tmp = Superboot(n_ens_sp[sp_idx])
#         ZV_tmp.boots = ZV_extrap[sp_idx][mom_idx]
#         ZV_extrap_mu[sp_idx, mom_idx] = ZV_tmp.compute_mean()
#         ZV_extrap_sigma[sp_idx, mom_idx] = ZV_tmp.compute_std()
#         ZA_tmp = Superboot(n_ens_sp[sp_idx])
#         ZA_tmp.boots = ZA_extrap[sp_idx][mom_idx]
#         ZA_extrap_mu[sp_idx, mom_idx] = ZA_tmp.compute_mean()
#         ZA_extrap_sigma[sp_idx, mom_idx] = ZA_tmp.compute_std()
#         for i, j in itertools.product(range(5), repeat = 2):
#             Z_tmp = Superboot(n_ens_sp[sp_idx])
#             Z_tmp.boots = Z_extrap_lin[sp_idx][i, j, mom_idx]
#             Z_extrap_mu[sp_idx, i, j, mom_idx] = Z_tmp.compute_mean()
#             Z_extrap_sigma[sp_idx, i, j, mom_idx] = Z_tmp.compute_std()
k_list_chiral = f['momenta'][()]
f.close()
mom_list_24I = np.array([Lat_24I.to_linear_momentum(k, datatype = np.float64) for k in k_list_chiral])
mom_list_32I = np.array([Lat_32I.to_linear_momentum(k, datatype = np.float64) for k in k_list_chiral])
# mom_list_24I = [Lat_24I.to_lattice_momentum(k, datatype = np.float64) for k in k_list_chiral]
# mom_list_32I = [Lat_32I.to_lattice_momentum(k, datatype = np.float64) for k in k_list_chiral]
apsq_list_24I = np.array([square(k) for k in mom_list_24I])
apsq_list_32I = np.array([square(k) for k in mom_list_32I])
mu_list_24I = np.array([ainv[0] * np.sqrt(square(p)) for p in mom_list_24I])
mu_list_32I = np.array([ainv[1] * np.sqrt(square(p)) for p in mom_list_32I])
musq_list_24I = np.array([(ainv[0] ** 2) * apsq for apsq in apsq_list_24I])
musq_list_32I = np.array([(ainv[1] ** 2) * apsq for apsq in apsq_list_32I])
# musq_list_24I = np.array([mu ** 2 for mu in mu_list_24I])
# musq_list_32I = np.array([mu ** 2 for mu in mu_list_32I])
print('24I mu list: ' + str(mu_list_24I))
print('32I mu list: ' + str(mu_list_32I))
print('24I mu^2 list: ' + str(musq_list_24I))
print('32I mu^2 list: ' + str(musq_list_32I))

# print('ZA mean: ' + str(ZA_extrap_mu[:, 2]))
# print('ZA std: ' + str(ZA_extrap_sigma[:, 2]))
# print('ZV mean: ' + str(ZV_extrap_mu[:, 2]))
# print('ZV std: ' + str(ZV_extrap_sigma[:, 2]))

# apsq = True         # x axis to plot against. If apsq, plots against (ap)^2, else plots against mu^2.
apsq = False
# x_axis = [apsq_list_24I, apsq_list_32I] if apsq else [mu_list_24I, mu_list_32I]
x_axis = [apsq_list_24I, apsq_list_32I] if apsq else [musq_list_24I, musq_list_32I]
# xlabel = '$(ap)^2$' if apsq else '$\\mu\;(\\mathrm{GeV})$'
xlabel = '$(ap)^2$' if apsq else '$\\mu^2\;(\\mathrm{GeV}^2)$'
# xlimits = [[0.0, 4.0], [0.0, 2.5]] if apsq else [[1.0, 4.1], [1.0, 4.4]]
xlimits = [[0.0, 4.0], [0.0, 2.5]] if apsq else [[0., 12.], [0., 12.]]
# sp_colors = colors[:2]
sp_colors = [colors[0], colors[0]]
sp_labels = ['a = 0.11 fm', 'a = 0.08 fm']
asp_ratio = 4/3

def plot_rcs_raw(sp_idx, cvs, sigmas, ylabel, ylimits, path):
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
        plt.xlabel(xlabel, fontsize = style['fontsize'])
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

# Think about best way to plot Zq, ZV, ZA: plot against (ap)^2, or mu^2?
# Zq
Zq24I_range = [0.8, 3.0]
Zq32I_range = [0.8, 1.5]
plot_rcs_raw(0, Zq_extrap_mu, Zq_extrap_sigma, '$\mathcal{Z}_q^\mathrm{RI}$', Zq24I_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/raw_plots/Zq_24I.pdf')
plot_rcs_raw(1, Zq_extrap_mu, Zq_extrap_sigma, '$\mathcal{Z}_q^\mathrm{RI}$', Zq32I_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/raw_plots/Zq_32I.pdf')

yticks_32I = [0.7, 0.8, 0.9, 1.0]
ytick_labels_32I = ['0.7', '0.8', '0.9', '1.0']

# ZV
# ZV24I_range = [0.6, 2.6]
# ZV32I_range = [0.7, 1.4]
ZV24I_range = [0.6, 1.4]            # first 4 points
ZV32I_range = [0.7, 1.0]
# ZV32I_range = [0.7, 0.95]
# ZV32I_range = ZV24I_range         # check the error isn't crazily blown up
plot_rcs_raw(0, ZV_extrap_mu, ZV_extrap_sigma, '$\mathcal{Z}_V$', ZV24I_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/raw_plots/ZV_24I.pdf')
plot_rcs_raw(1, ZV_extrap_mu, ZV_extrap_sigma, '$\mathcal{Z}_V$', ZV32I_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/raw_plots/ZV_32I.pdf')

# ZA
# ZA24I_range = [0.6, 2.6]
# ZA32I_range = [0.7, 1.4]
ZA24I_range = [0.6, 1.4]
ZA32I_range = [0.7, 1.0]
# ZA32I_range = [0.7, 0.95]
# ZA32I_range = ZA24I_range
plot_rcs_raw(0, ZA_extrap_mu, ZA_extrap_sigma, '$\mathcal{Z}_A$', ZA24I_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/raw_plots/ZA_24I.pdf')
plot_rcs_raw(1, ZA_extrap_mu, ZA_extrap_sigma, '$\mathcal{Z}_A$', ZA32I_range,  \
                        '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/raw_plots/ZA_32I.pdf')

# fit data to model
def fit_data_model(sp_idx, cvs_all, sigmas_all, subset_idxs, model): #, x_axis = [apsq_list_24I, apsq_list_32I]):
    """
    Fits data Z to an arbitrary model by minimizing the correlated chi^2.
    lam is the parameter for linear shrinkage, i.e. lam = 0 is the uncorrelated covariance, and lam = 1 is the
    original covariance.
    """
    cvs, sigmas = np.array(cvs_all[sp_idx]), np.array(sigmas_all[sp_idx])
    x_list = np.array(x_axis[sp_idx])
    fitter = UncorrFitter(x_list[subset_idxs], cvs[subset_idxs], sigmas[subset_idxs], model)
    fit_out = fitter.fit()
    print('Best fit coeffs: ' + str(fit_out[0]))
    print('chi^2 / dof: ' + str(fit_out[1] / fit_out[2]))
    print('Parameter covariance: ' + str(fit_out[3]))
    return fit_out, fitter

# Plot fitted data. TODO add a fit band with the known value of ZA
# fill_color = 'b'
fill_color = (0, 0, 1, 0.3)
# Z0_color = (1, 1, 1, 0.3)
# Z0_color = 'k'
Z0_color = 'g'
# Z0_color = '0.8'
a_labels = ['0.11 fm', '0.08 fm']
def plot_fit_out(sp_idx, cvs, sigmas, fitter, fout, ylabel, ylimits, path, plt_known = False, ytick_locs = None, ytick_labels = None):
    x_band = np.linspace(xlimits[sp_idx][0], xlimits[sp_idx][1])
    fx_cvs, fx_sigmas = fitter.gen_fit_band(fout[0], fout[3], x_band)
    # print(x_band)
    with sns.plotting_context('paper'):
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        _, caps, _ = plt.errorbar(x_axis[sp_idx], cvs[sp_idx], sigmas[sp_idx], fmt = '.', c = sp_colors[sp_idx], \
                label = 'Data', capsize = style['endcaps'], markersize = style['markersize'], \
                elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.fill_between(x_band, fx_cvs + fx_sigmas, fx_cvs - fx_sigmas, color = fill_color, alpha = 0.2, linewidth = 0.0, label = 'Extrapolation')
        if plt_known:
            plt.fill_between(x_band, ZA0_mu[sp_idx] + ZA0_std[sp_idx], ZA0_mu[sp_idx] - ZA0_std[sp_idx], color = Z0_color, alpha = 1.0, linewidth = 0.0, \
                    label = '$\\mathcal{Z}_A$, Ref. [31]')
        plt.xlabel(xlabel, fontsize = style['fontsize'])
        plt.ylabel(ylabel + ' (a = ' + a_labels[sp_idx] + ')', fontsize = style['fontsize'])
        plt.xlim(xlimits[sp_idx])
        plt.ylim(ylimits)
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        if ytick_locs:
            ax.set_yticks(ytick_locs)
            ax.set_yticklabels(ytick_labels)
        plt.legend(prop={'size': style['fontsize'] * 0.8})
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Plot ' + ylabel + ' saved at: \n   ' + path)

# ZV for 24I
# subset_idxs = [0, 1, 2, 3, 4, 5, 6, 7]
subset_idxs = [0, 1, 2, 3]
def model_ZV24I(params):
    def model(apsq):
        return params[0] * (apsq ** 0) + params[1] * (apsq ** 1) + params[2] * (apsq ** 2)
    return model
m_ZV24I = Model(model_ZV24I, 3, ['', 'x', 'x^2'], ['c0', 'c1', 'c2'])
ZV24I_fout, ZV24I_fitter = fit_data_model(0, ZV_extrap_mu, ZV_extrap_sigma, subset_idxs, m_ZV24I)
ZV24I_params, ZV24I_cov = ZV24I_fout[0], ZV24I_fout[3]
ZV24I_cv, ZV24I_std = ZV24I_params[0], np.sqrt(ZV24I_cov[0, 0])
ZV24I_dist = gen_fake_ensemble([ZV24I_cv, ZV24I_std], n_samples = n_samp)
print('ZV for 24I = ' + export_float_latex(ZV24I_cv, ZV24I_std, sf = 2))
plot_fit_out(0, ZV_extrap_mu, ZV_extrap_sigma, ZV24I_fitter, ZV24I_fout, '$\mathcal{Z}_V$', ZV24I_range, \
                '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/fit_plots/ZV_24I.pdf')

# ZV for 32I
subset_idxs = [0, 1, 2, 3]
def model_ZV32I(params):
    def model(apsq):
        return params[0] * (apsq ** 0) + params[1] * (apsq ** 1) + params[2] * (apsq ** 2)
    return model
m_ZV32I = Model(model_ZV32I, 3, ['', 'x', 'x^2'], ['c0', 'c1', 'c2'])
ZV32I_fout, ZV32I_fitter = fit_data_model(1, ZV_extrap_mu, ZV_extrap_sigma, subset_idxs, m_ZV32I)
ZV32I_params, ZV32I_cov = ZV32I_fout[0], ZV32I_fout[3]
ZV32I_cv, ZV32I_std = ZV32I_params[0], np.sqrt(ZV32I_cov[0, 0])
ZV32I_dist = gen_fake_ensemble([ZV32I_cv, ZV32I_std], n_samples = n_samp)
print('ZV for 32I = ' + export_float_latex(ZV32I_cv, ZV32I_std, sf = 2))
plot_fit_out(1, ZV_extrap_mu, ZV_extrap_sigma, ZV32I_fitter, ZV32I_fout, '$\mathcal{Z}_V$', ZV32I_range, \
                '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/fit_plots/ZV_32I.pdf', ytick_locs = yticks_32I, ytick_labels = ytick_labels_32I)

# ZA for 24I
subset_idxs = [0, 1, 2, 3]
def model_ZA24I(params):
    def model(apsq):
        return params[0] * (apsq ** 0) + params[1] * (apsq ** 1) + params[2] * (apsq ** 2)
    return model
m_ZA24I = Model(model_ZA24I, 3, ['', 'x', 'x^2'], ['c0', 'c1', 'c2'])
ZA24I_fout, ZA24I_fitter = fit_data_model(0, ZA_extrap_mu, ZA_extrap_sigma, subset_idxs, m_ZA24I)
ZA24I_params, ZA24I_cov = ZA24I_fout[0], ZA24I_fout[3]
ZA24I_cv, ZA24I_std = ZA24I_params[0], np.sqrt(ZA24I_cov[0, 0])
ZA24I_dist = gen_fake_ensemble([ZA24I_cv, ZA24I_std], n_samples = n_samp)
print('ZA for 24I = ' + export_float_latex(ZA24I_cv, ZA24I_std, sf = 2))
plot_fit_out(0, ZA_extrap_mu, ZA_extrap_sigma, ZA24I_fitter, ZA24I_fout, '$\mathcal{Z}_A$', ZA24I_range, \
                '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/fit_plots/ZA_24I.pdf', plt_known = True)

# ZA for 32I
subset_idxs = [0, 1, 2, 3]
def model_ZA32I(params):
    def model(apsq):
        return params[0] * (apsq ** 0) + params[1] * (apsq ** 1) + params[2] * (apsq ** 2)
    return model
m_ZA32I = Model(model_ZA32I, 3, ['', 'x', 'x^2'], ['c0', 'c1', 'c2'])
ZA32I_fout, ZA32I_fitter = fit_data_model(1, ZA_extrap_mu, ZA_extrap_sigma, subset_idxs, m_ZA32I)
ZA32I_params, ZA32I_cov = ZA32I_fout[0], ZA32I_fout[3]
ZA32I_cv, ZA32I_std = ZA32I_params[0], np.sqrt(ZA32I_cov[0, 0])
ZA32I_dist = gen_fake_ensemble([ZA32I_cv, ZA32I_std], n_samples = n_samp)
print('ZA for 32I = ' + export_float_latex(ZA32I_cv, ZA32I_std, sf = 2))
plot_fit_out(1, ZA_extrap_mu, ZA_extrap_sigma, ZA32I_fitter, ZA32I_fout, '$\mathcal{Z}_A$', ZA32I_range, \
                '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/rcs/fit_plots/ZA_32I.pdf', plt_known = True, ytick_locs = yticks_32I, \
                ytick_labels = ytick_labels_32I)

out_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/ZVA.h5'
f_out = h5py.File(out_path, 'w')
f_out['momenta'] = k_list_chiral

f_out['ZV/24I/mean'] = ZV24I_cv
f_out['ZV/24I/std'] = ZV24I_std
f_out['ZV/24I/dist'] = ZV24I_dist

f_out['ZV/32I/mean'] = ZV32I_cv
f_out['ZV/32I/std'] = ZV32I_std
f_out['ZV/32I/dist'] = ZV32I_dist

f_out['ZA/24I/mean'] = ZA24I_cv
f_out['ZA/24I/std'] = ZA24I_std
f_out['ZA/24I/dist'] = ZA24I_dist

f_out['ZA/32I/mean'] = ZA32I_cv
f_out['ZA/32I/std'] = ZA32I_std
f_out['ZA/32I/dist'] = ZA32I_dist

print('Results for ZV, ZA saved at: ' + out_path)
f_out.close()
