################################################################################
##################### Run this instead of msbar_conversion #####################
################################################################################

# use CMU Serif
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext'] = True

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

# set parameters
sp_idx = int(sys.argv[1])
stem = ['24I', '32I'][sp_idx]
ensembles = [
    ['24I/ml0p005/', '24I/ml0p01/'],
    ['32I/ml0p004/', '32I/ml0p006/', '32I/ml0p008/']
][sp_idx]
l = [24, 32][sp_idx]
t = 64
ainv = [1.784, 2.382][sp_idx]
mpi_list = [
    [0.3396, 0.4322],
    [0.3020, 0.3597, 0.4108]
][sp_idx]
amq_list = [
    [0.005, 0.01],
    [0.004, 0.006, 0.008]
][sp_idx]
mu0_idx = [2, 2][sp_idx]                # index of mu0 mode
q_label = ['$q = (3, 3, 0, 0)$', '$q = (3, 3, 0, 0)$'][sp_idx]

xtick_locs = [[0.0, 0.005, 0.01], [0.0, 0.002, 0.004, 0.006, 0.008, 0.01]][sp_idx]
xtick_labels = [['0.0', '0.005', '0.01'], ['0.0', '0.002', '0.004', '0.006', '0.008', '0.01']][sp_idx]

L = Lattice(l, t)
a_fm = hbarc / ainv
n_ens = len(ensembles)
ampi_list = [mpi_list[i] / ainv for i in range(n_ens)]

file_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ens + 'Z_gamma.h5' for ens in ensembles]
out_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + stem + '/chiral_extrap/Z_extrap.h5'
plot_dir = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/amell_extrap/' + stem + '/'

Fs = [h5py.File(fpath, 'r') for fpath in file_paths]
k_list_ens = np.array([f['momenta'][()] for f in Fs])
print(k_list_ens)
assert np.array_equal(k_list_ens[0], k_list_ens[1])         # make sure each ensemble has same momentum modes
k_list = k_list_ens[0]
mom_list = np.array([L.to_linear_momentum(k, datatype=np.float64) for k in k_list])
# mu_list = np.array([get_energy_scale(q, a_fm, L) for q in k_list])
mu_list = np.array([get_energy_scale_linear(q, a_fm, L) for q in k_list])
print('Energy scales')
print(mu_list)
n_mom = len(mom_list)
mass_list = np.array(amq_list)
# mass_list = np.array(mpi_list)

# Get renormalization coefficients (not chirally extrapolated)
Zq_list = np.array([np.real(f['Zq'][()]) for f in Fs])
ZV_list = np.array([np.real(f['ZV'][()]) for f in Fs])
ZA_list = np.array([np.real(f['ZA'][()]) for f in Fs])
Z_list = []
for idx in range(n_ens):
    Z = np.zeros((5, 5, n_mom, n_boot), dtype = np.float64)
    f = Fs[idx]
    for i, j in itertools.product(range(5), repeat = 2):
        key = 'Z' + str(i + 1) + str(j + 1)
        Z[i, j] = np.real(f[key][()])
    Z_list.append(Z)
Z_list = np.array(Z_list)

# Get Lambda factor. Lambdas are bootstrapped, but the boots are uncorrelated. Shape is (n_ens, 5, 5, n_mom, n_boot)
Lambda_list = np.array([np.real(f['Lambda'][()]) for f in Fs])
multiplets = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [1, 2], [2, 1], [3, 4], [4, 3]], dtype = object)    # nonzero indices for Lambda

# get means and stds
Zq_mu = np.transpose(np.mean(Zq_list, axis = 2))                      # shape = (n_mom, n_ens)
Zq_std = np.transpose(np.std(Zq_list, axis = 2, ddof = 1))
ZV_mu = np.transpose(np.mean(ZV_list, axis = 2))
ZV_std = np.transpose(np.std(ZV_list, axis = 2, ddof = 1))
ZA_mu = np.transpose(np.mean(ZA_list, axis = 2))
ZA_std = np.transpose(np.std(ZA_list, axis = 2, ddof = 1))
Z_mu = np.transpose(np.mean(Z_list, axis = 4))                        # shape = (n_mom, 5, 5, n_ens)
Z_std = np.transpose(np.std(Z_list, axis = 4, ddof = 1))

Lambda_mu = np.mean(Lambda_list, axis = 4)
Lambda_std = np.std(Lambda_list, axis = 4, ddof = 1)

# fill_color = (0, 0, 1, 0.3)
fill_color = 'b'
# TODO figure out what we want xlimits to be. I could stack the figures for 24I and 32I
# xlimits = [
#     [0.0, 0.012],
#     [0.0, 0.012]
# ][sp_idx]
xlimits = [
    [-0.0005, 0.012],
    [-0.0005, 0.012]
][sp_idx]
scale_factors = [0.7, 1.3]          # scale factors for yrange
xlabel = '$a m_\ell$'
a_label = ['0.11 fm', '0.08 fm'][sp_idx]
asp_ratio = 4/3
# def plot_fit_out(mu0_idx, cvs, sigmas, fit_params, ylabel, ylimits, path):
def plot_fit_out(mu0_idx, cvs, sigmas, extrap_mu, extrap_sigma, fit_params, ylabel, path):
    x_band = np.linspace(xlimits[0], xlimits[1])
    fx_cvs = fit_params[mu0_idx][1].mean + fit_params[mu0_idx][0].mean * x_band
    fx_stds = np.sqrt(fit_params[mu0_idx][1].std**2 + (fit_params[mu0_idx][0].std * x_band)**2)
    # TODO include the covariance for the fit parameters here
    print([np.min(cvs[mu0_idx] - sigmas[mu0_idx]), extrap_mu[mu0_idx] - extrap_sigma[mu0_idx]])
    print([np.max(cvs[mu0_idx] + sigmas[mu0_idx]), extrap_mu[mu0_idx] + extrap_sigma[mu0_idx]])
    data_window = [min(np.min(fx_cvs - fx_stds), np.min(cvs[mu0_idx] - sigmas[mu0_idx]), extrap_mu[mu0_idx] - extrap_sigma[mu0_idx]), \
                    max(np.max(fx_cvs + fx_stds), np.max(cvs[mu0_idx] + sigmas[mu0_idx]), extrap_mu[mu0_idx] + extrap_sigma[mu0_idx])]
    Delta_window = data_window[1] - data_window[0]
    ylimits = [data_window[0] - Delta_window * scale_factors[0], data_window[1] + Delta_window * scale_factors[1]]
    with sns.plotting_context('paper'):
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        plt.vlines(0.0, 0.0, 10.0, linestyles = 'dashed', label = 'Chiral limit', linewidth = style['ebar_width'], color = 'k')
        _, caps, _ = plt.errorbar(mass_list, cvs[mu0_idx], sigmas[mu0_idx], fmt = '.', c = 'r', \
                label = 'Data', capsize = style['endcaps'], markersize = style['markersize'], \
                elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        _, caps, _ = plt.errorbar([0.0], [extrap_mu[mu0_idx]], [extrap_sigma[mu0_idx]], fmt = '.', c = fill_color, \
                label = 'Extrapolation', capsize = style['endcaps'], markersize = style['markersize'], \
                elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.fill_between(x_band, fx_cvs + fx_stds, fx_cvs - fx_stds, color = fill_color, alpha = 0.2, linewidth = 0.0)#, label = 'Fit band')
        plt.xlabel(xlabel, fontsize = style['fontsize'])
        plt.ylabel(ylabel + ' (a = ' + a_label + ')', fontsize = style['fontsize'])
        plt.xlim(xlimits)
        plt.ylim(ylimits)         # set this after we figure out the ylimits
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        plt.legend(prop={'size': style['fontsize'] * 0.8})
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Plot ' + ylabel + ' saved at: \n   ' + path)

# chirally extrapolate Zq
print('Chiral extrapolation for Zq.')
Zq_fit = np.zeros((n_mom, n_ens, n_boot), dtype = np.float64)
Zq_fit_params = []
Zq_fit_mu = np.zeros((n_mom), dtype = np.float64)
Zq_fit_std = np.zeros((n_mom), dtype = np.float64)
for mom_idx in range(n_mom):
    print('Momentum index ' + str(mom_idx))
    fitq_superboot = []
    for ii in range(n_ens):
        tmpq = Superboot(n_ens)
        tmpq.populate_ensemble(Zq_list[ii][mom_idx], ii)
        fitq_superboot.append(tmpq)
    fit_params_tmp, chi2_q, y_extrap_q = uncorr_linear_fit(mass_list, fitq_superboot, 0, label = 'Zq')
    Zq_fit_params.append(fit_params_tmp)
    Zq_fit[mom_idx] = y_extrap_q.boots
    Zq_fit_mu[mom_idx] = y_extrap_q.mean
    Zq_fit_std[mom_idx] = y_extrap_q.std
# print('c0 mean: ' + str(Zq_fit_params[mu0_idx][0].mean))
# print('c0 std: ' + str(Zq_fit_params[mu0_idx][0].std))
# print('c1 mean: ' + str(Zq_fit_params[mu0_idx][1].mean))
# print('c1 std: ' + str(Zq_fit_params[mu0_idx][1].std))
plot_fit_out(mu0_idx, Zq_mu, Zq_std, Zq_fit_mu, Zq_fit_std, Zq_fit_params, '$\mathcal{Z}_q^\mathrm{RI}$', plot_dir + 'Zq.pdf')

# chirally extrapolate ZV and ZA
print('Chiral extrapolation for ZV and ZA.')
ZV_fit = np.zeros((n_mom, n_ens, n_boot), dtype = np.float64)
ZA_fit = np.zeros((n_mom, n_ens, n_boot), dtype = np.float64)
ZV_fit_params = []
ZA_fit_params = []
ZV_fit_mu = np.zeros((n_mom), dtype = np.float64)
ZV_fit_std = np.zeros((n_mom), dtype = np.float64)
ZA_fit_mu = np.zeros((n_mom), dtype = np.float64)
ZA_fit_std = np.zeros((n_mom), dtype = np.float64)
for mom_idx in range(n_mom):
    print('Momentum index ' + str(mom_idx))
    fitV_superboot = []
    fitA_superboot = []
    for ii in range(n_ens):
        tmpV = Superboot(n_ens)
        tmpV.populate_ensemble(ZV_list[ii][mom_idx], ii)
        fitV_superboot.append(tmpV)
        tmpA = Superboot(n_ens)
        tmpA.populate_ensemble(ZA_list[ii][mom_idx], ii)
        fitA_superboot.append(tmpA)
    fit_params_tmp, chi2_V, y_extrap_V = uncorr_linear_fit(mass_list, fitV_superboot, 0, label = 'ZV')
    ZV_fit_params.append(fit_params_tmp)
    fit_params_tmp, chi2_A, y_extrap_A = uncorr_linear_fit(mass_list, fitA_superboot, 0, label = 'ZA')
    ZA_fit_params.append(fit_params_tmp)
    ZV_fit[mom_idx] = y_extrap_V.boots
    ZV_fit_mu[mom_idx] = y_extrap_V.mean
    ZV_fit_std[mom_idx] = y_extrap_V.std
    ZA_fit[mom_idx] = y_extrap_A.boots
    ZA_fit_mu[mom_idx] = y_extrap_A.mean
    ZA_fit_std[mom_idx] = y_extrap_A.std
plot_fit_out(mu0_idx, ZV_mu, ZV_std, ZV_fit_mu, ZV_fit_std, ZV_fit_params, '$\mathcal{Z}_V$', plot_dir + 'ZV.pdf')
plot_fit_out(mu0_idx, ZA_mu, ZA_std, ZA_fit_mu, ZA_fit_std, ZA_fit_params, '$\mathcal{Z}_A$', plot_dir + 'ZA.pdf')

# perform and save chiral extrapolation on Lambda
print('Chiral extrapolation for Lambda_ij.')
Lambda_fit = np.zeros((n_ens, n_boot, n_mom, 5, 5), dtype = np.float64)
for ii, mult_idx in enumerate(multiplets):
    print('Fitting Lambda' + str(mult_idx[0]) + str(mult_idx[1]))
    Lambda_mult = []
    for mom_idx in range(n_mom):
        print('Momentum index ' + str(mom_idx))
        fitdata_superboot = []    # create n_ens Superboot objects
        for ii in range(n_ens):
            tmp = Superboot(n_ens)
            tmp.populate_ensemble(Lambda_list[ii][mult_idx[0], mult_idx[1], mom_idx], ii)
            fitdata_superboot.append(tmp)
        fit_data, chi2, y_extrap = uncorr_linear_fit(mass_list, fitdata_superboot, 0)
        # chi2, y_extrap = uncorr_const_fit(mass_list, fitdata_superboot, 0.140)
        Lambda_fit[:, :, mom_idx, mult_idx[0], mult_idx[1]] = y_extrap.boots    # just reput them into new superboot objects when we read them in

# Process as a Z factor
scheme = 'gamma'                    # scheme == 'gamma' or 'qslash'
F = getF(L, scheme)                 # tree level projections
Z_chiral = np.zeros((5, 5, n_mom, n_ens, n_boot))
for mom_idx in range(n_mom):
    for ens_idx in range(n_ens):
        for b in range(n_boot):
            Lambda_inv = np.linalg.inv(Lambda_fit[ens_idx, b, mom_idx])
            Z_chiral[:, :, mom_idx, ens_idx, b] = (Zq_list[ens_idx][mom_idx, b] ** 2) * np.einsum('ik,kj->ij', F, Lambda_inv)
    for mult_idx in multiplets:
        Z_boot = Superboot(n_ens)
        Z_boot.boots = Z_chiral[mult_idx[0], mult_idx[1], mom_idx]
        Z_boot.compute_mean()
        Z_boot.compute_std()
        print('Z' + str(mult_idx) + ' at mom_idx ' + str(mom_idx) + ' = ' + str(Z_boot.mean) + ' \pm ' + str(Z_boot.std))

# save results of chiral extrapolation and of interpolation
# uncomment when I've verified we don't need interpCoeffs
# fchi_out = h5py.File(out_path, 'w')
# fchi_out['momenta'] = k_list
# fchi_out['Zq/values'] = Zq_fit
# fchi_out['ZV/value'] = ZV_fit
# fchi_out['ZA/value'] = ZA_fit
# for ii, mult_idx in enumerate(multiplets):
#     fchi_out['Lambda' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = np.einsum('ebq->qeb', Lambda_fit[:, :, :, mult_idx[0], mult_idx[1]])
#     fchi_out['Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Z_chiral[mult_idx[0], mult_idx[1]]
#     datapath = 'O' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)
#     fchi_out[datapath + '/ZijZVm2'] = Zij_by_ZVsq[mult_idx[0], mult_idx[1]]    # this is the full dataset with p^2 dependence
# print('Chiral extrapolation saved at: ' + out_path)
# fchi_out.close()
