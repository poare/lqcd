################################################################################
# Performs the amell --> 0 extrapolation, keeping into account correlations    #
# between the RCs computed on the same ensemble.                               #
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
from scipy.linalg import block_diag
import h5py
import os
import itertools
import pandas as pd
import gvar as gv
import lsqfit
from utils import *

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from fittools import *
from formattools import *
style = styles['prd_twocol']

# toggle switch to fit constant or linear in amell
# const = True
const = False
extrap_dir = 'const_amell_extrap' if const else 'linear_amell_extrap'

# set parameters
sp_idx = int(sys.argv[1])
stem = ['24I', '32I'][sp_idx]
l = [24, 32][sp_idx]
t = 64
ainv = [1.784, 2.382][sp_idx]
# mu0_idx = [2, 2][sp_idx]                # index of mu0 mode
mu0_idx = [2, 0][sp_idx]                # delete when we have the full 8 momenta run
# mu0_idx = [0, 0][sp_idx]                # use this if we only have the (3, 3, 0, 0) mode run

ensembles = [
    ['24I/ml0p005/', '24I/ml0p01/'],
    ['32I/ml0p004/', '32I/ml0p006/', '32I/ml0p008/']
][sp_idx]
mpi_list = [
    [0.3396, 0.4322],
    [0.3020, 0.3597, 0.4108]
][sp_idx]
amq_list = [
    [0.005, 0.01],
    [0.004, 0.006, 0.008]
][sp_idx]

q_label = ['$q = (3, 3, 0, 0)$', '$q = (3, 3, 0, 0)$'][sp_idx]
xtick_locs = [[0.0, 0.005, 0.01], [0.0, 0.002, 0.004, 0.006, 0.008, 0.01]][sp_idx]
xtick_labels = [['0.0', '0.005', '0.01'], ['0.0', '0.002', '0.004', '0.006', '0.008', '0.01']][sp_idx]

L = Lattice(l, t)
a_fm = hbarc / ainv
n_ens = len(ensembles)
ampi_list = [mpi_list[i] / ainv for i in range(n_ens)]

n_samp = n_boot                             # Number of samples for distribution to generate after fit
# n_samp = 500
n_band = 500                                # number of points to plot the fit band at
file_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ens + 'Z_gamma.h5' for ens in ensembles]
out_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + stem + '/chiral_extrap/Z_extrap.h5'
plot_dir = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/' + extrap_dir + '/' + stem + '/'

Fs = [h5py.File(fpath, 'r') for fpath in file_paths]
k_list_ens = np.array([f['momenta'][()] for f in Fs])
assert np.array_equal(k_list_ens[0], k_list_ens[1])         # make sure each ensemble has same momentum modes
k_list = k_list_ens[0]
print('k_list: ' + str(k_list))
mom_list = np.array([L.to_linear_momentum(k, datatype=np.float64) for k in k_list])
mu_list = np.array([get_energy_scale_linear(q, a_fm, L) for q in k_list])
print('Energy scales')
print(mu_list)
print('NPR scale mu0: mode ' + str(k_list[mu0_idx]) + ' with energy ' + str(mu_list[mu0_idx]) + ' GeV.')
n_mom = len(mom_list)
mass_list = np.array(amq_list)

# Get renormalization coefficients (not chirally extrapolated). Shape (n_ens, n_mom, n_boot)
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
n_multiplets = len(multiplets)

# get means and stds
Zq_mu = np.transpose(np.mean(Zq_list, axis = 2))                                # shape = (n_mom, n_ens)
Zq_std = np.transpose(np.std(Zq_list, axis = 2, ddof = 1))
ZV_mu = np.transpose(np.mean(ZV_list, axis = 2))
ZV_std = np.transpose(np.std(ZV_list, axis = 2, ddof = 1))
ZA_mu = np.transpose(np.mean(ZA_list, axis = 2))
ZA_std = np.transpose(np.std(ZA_list, axis = 2, ddof = 1))
Z_mu = np.einsum('eijq->qije', np.mean(Z_list, axis = 4))                                  # shape = (n_mom, 5, 5, n_ens)
Z_std = np.einsum('eijq->qije', np.std(Z_list, axis = 4, ddof = 1))
Lambda_mu = np.einsum('eijq->qije', np.mean(Lambda_list, axis = 4))                        # shape = (n_mom, 5, 5, n_ens)
Lambda_std = np.einsum('eijq->qije', np.std(Lambda_list, axis = 4, ddof = 1))

n_vars = 3 + n_multiplets
# shape: (n_vars, n_ens, n_mom, n_boot)
all_vars = np.concatenate((
    np.expand_dims(Zq_list, axis = 0),
    np.expand_dims(ZV_list, axis = 0),
    np.expand_dims(ZA_list, axis = 0),
    np.array([Lambda_list[:, mult_idx[0], mult_idx[1], :, :] for mult_idx in multiplets])
))
all_vars_mu = np.mean(all_vars, axis = 3)
all_vars_std = np.std(all_vars, axis = 3, ddof = 1)
var_labels = ['Zq', 'ZV', 'ZA', 'F_matrix/F11', 'F_matrix/F22', 'F_matrix/F33', 'F_matrix/F44', 'F_matrix/F55', 'F_matrix/F23', \
    'F_matrix/F32', 'F_matrix/F45', 'F_matrix/F54']
latex_var_labels = [r'$\mathcal{Z}_q^\mathrm{RI}', r'$\mathcal{Z}_V', r'$\mathcal{Z}_A']
for ii, mult_idx in enumerate(multiplets):
    latex_var_labels.append(r'$F_{' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + r'}')

fill_color = 'b'
xlimits = [
    [-0.0005, 0.012],
    [-0.0005, 0.012]
][sp_idx]
x_band = np.linspace(xlimits[0], xlimits[1])
scale_factors = [0.7, 1.3]          # scale factors for yrange
xlabel = '$a m_\ell$'
a_label = [r'0.11\;\mathrm{fm}', r'0.08\;\mathrm{fm}'][sp_idx]
asp_ratio = 4/3
def plot_fit_out(cvs, sigmas, extrap_mu, extrap_sigma, fx_dom, fx_lower, fx_upper, ylabel, path, plt_band = True):
    if plt_band:
        data_window = [min(np.min(fx_lower), np.min(cvs - sigmas), extrap_mu - extrap_sigma), \
                        max(np.max(fx_upper), np.max(cvs + sigmas), extrap_mu + extrap_sigma)]
    else:
        data_window = [min(np.min(cvs - sigmas), extrap_mu - extrap_sigma), max(np.max(cvs + sigmas), extrap_mu + extrap_sigma)]
    Delta_window = data_window[1] - data_window[0]
    ylimits = [data_window[0] - Delta_window * scale_factors[0], data_window[1] + Delta_window * scale_factors[1]]
    with sns.plotting_context('paper'):
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        plt.vlines(0.0, ylimits[0], ylimits[1], linestyles = 'dashed', label = '$am_\ell = 0$', linewidth = style['ebar_width'], color = 'k')
        _, caps, _ = plt.errorbar(mass_list, cvs, sigmas, fmt = '.', c = 'r', \
                label = 'Data', capsize = style['endcaps'], markersize = style['markersize'], \
                elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        _, caps, _ = plt.errorbar([0.0], [extrap_mu], [extrap_sigma], fmt = '.', c = fill_color, \
                capsize = style['endcaps'], markersize = style['markersize'], \
                elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        if plt_band:
            plt.fill_between(fx_dom, fx_lower, fx_upper, color = fill_color, alpha = 0.2, linewidth = 0.0, label = 'Extrapolation')
        plt.xlabel(xlabel, fontsize = style['fontsize'])
        ylabel_dec = ylabel + r'\; (a=' + a_label + r')$'
        plt.ylabel(ylabel_dec, fontsize = style['fontsize'])
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
        plt.close()

all_boots = np.zeros((n_ens, n_mom, n_vars, n_boot), dtype = np.float64)
# covar = np.zeros((n_ens, n_mom, n_vars, n_vars), dtype = np.float64)
for ens_idx in range(n_ens):
    for mom_idx in range(n_mom):
        all_boots[ens_idx, mom_idx, 0, :] = Zq_list[ens_idx, mom_idx]
        all_boots[ens_idx, mom_idx, 1, :] = ZV_list[ens_idx, mom_idx]
        all_boots[ens_idx, mom_idx, 2, :] = ZA_list[ens_idx, mom_idx]
        for ii, mult_idx in enumerate(multiplets):
            all_boots[ens_idx, mom_idx, 3 + ii, :] = Lambda_list[ens_idx, mult_idx[0], mult_idx[1], mom_idx]
        # covar[ens_idx, mom_idx] = np.cov(all_boots[ens_idx, mom_idx])

# assemble data into correct form
def make_data(bootstraps, domain = mass_list, lam = 0.9):
    """
    Makes data for lsqfit. 
    
    Parameters
    ----------
    bootstraps : np.array (n_ens, n_vars, n_boot)
        Bootstrap data to fit, arranged in order (Zq, ZV, ZA, F11, ..., F55)
    """
    cvs = np.mean(bootstraps, axis = 2)
    mean = np.concatenate(cvs)
    covar = np.zeros((n_ens, n_vars, n_vars), dtype = np.float64)
    for ens_idx in range(n_ens):
        covar[ens_idx] = np.cov(bootstraps[ens_idx])
    cov = block_diag(*covar)
    cov = shrinkage(cov, lam)
    return domain, gv.gvar(mean, cov)

def fcn(m, p):
    """
    Constant or linear fitting function f(m; c) = c0 + c1 m. Note that c0 = [c0Zq, c0ZV, ..., c0F55] is 12-dimensional, 
    so the return should be a 12 * len(m)-dimensional vector. 
    """
    c0 = p['c0']        # n_vars = 12 dimensional
    # filler = np.ones((n_ens), dtype = np.float64)
    filler = np.ones((len(m)), dtype = np.float64)
    c0_full = np.kron(filler, c0)
    if const:
        # return c0_full + np.zeros(c0_full.shape, dtype = np.float64)*m
        c1 = np.zeros(c0.shape, dtype = np.float64)
    else:
        c1 = p['c1']
    return c0_full + np.kron(m, c1)

def get_fit_band(params, fcn, xlims):
    """
    Generates a fit band for a given set of parameters from an lsqfit. Note for a multivariate function, the return bands will 
    be of shape (n_vars, n_band).
    """
    xx = np.linspace(xlims[0], xlims[1], n_band)
    fx = fcn(xx, params)            # shape = (n_vars * len(xx))
    fx_lower = gv.mean(fx) - gv.sdev(fx)
    fx_upper = gv.mean(fx) + gv.sdev(fx)
    lower_reshape = np.reshape(fx_lower, (n_vars, n_band), order = 'F')
    upper_reshape = np.reshape(fx_upper, (n_vars, n_band), order = 'F')
    fx_reshape = np.reshape(fx, (n_vars, n_band), order = 'F')
    return xx, lower_reshape, upper_reshape, fx_reshape

# fit data to model
def fit_data_model(rcs):
    """
    Fits data Z to an arbitrary model by minimizing the uncorrelated chi^2.
    """
    dom, Zfit = make_data(rcs)          # Zfit = np.array(gvars) of shape (n_ens * n_vars)
    Zfit_ens = np.reshape(Zfit, (n_ens, n_vars))
    Z_cv = np.mean(Zfit_ens, axis = 0)          # average Zfit over every n_vars variables to get an estimate of Z

    p0 = {'c0' : gv.mean(Z_cv)} if const else {'c0' : gv.mean(Z_cv), 'c1' : np.zeros((n_vars), dtype = np.float64)}
    print('Initial guess: ' + str(p0))
    fit = lsqfit.nonlinear_fit(data = (dom, Zfit), fcn = fcn, prior = None, p0 = p0)
    print(fit)
    return fit

# perform fit
Zall_out = np.zeros((n_mom, n_vars), dtype = object)
Zall_mu = np.zeros((n_mom, n_vars), dtype = np.float64)
Zall_std = np.zeros((n_mom, n_vars), dtype = np.float64)
Zall_cov = np.zeros((n_mom, n_vars, n_vars), dtype = np.float64)
# Zall_dist = np.zeros((n_mom, n_vars, n_samp), dtype = np.float64)
Zall_params = []
for mom_idx in range(n_mom):
    print('Momentum: ' + str(k_list[mom_idx]))
    fout = fit_data_model(all_boots[:, mom_idx, :, :])
    Zall_out[mom_idx, :] = fout.p['c0']
    Zall_mu[mom_idx, :] = gv.mean(fout.p['c0'])
    Zall_std[mom_idx, :] = gv.sdev(fout.p['c0'])
    Zall_cov[mom_idx, :, :] = gv.evalcov(fout.p['c0'])
    # Zall_dist[mom_idx, :, :] = gen_corr_dist(Zall_mu[mom_idx], Zall_cov[mom_idx], n_samples = n_samp)
    Zall_params.append(fout.p)

# print and plot at mu0_idx
fx_lower, fx_upper = np.zeros((n_mom, n_vars, n_band), dtype = np.float64), np.zeros((n_mom, n_vars, n_band), dtype = np.float64)
fit_band = np.zeros((n_mom, n_vars, n_band), dtype = object)
for mom_idx in range(n_mom):
    xx, fx_lower[mom_idx], fx_upper[mom_idx], fit_band[mom_idx] = get_fit_band(Zall_params[mom_idx], fcn, xlimits)

for ii, lbl in enumerate(var_labels):
    print(lbl + '/' + stem + ' for mu0 at am_\ell = 0 is ' + export_float_latex(Zall_mu[mu0_idx, ii], Zall_std[mu0_idx, ii], sf = 2))
    plot_fit_out(all_vars_mu[ii, :, mu0_idx], all_vars_std[ii, :, mu0_idx], Zall_mu[mu0_idx, ii], Zall_std[mu0_idx, ii], xx, \
        fx_lower[mu0_idx, ii], fx_upper[mu0_idx, ii], latex_var_labels[ii], plot_dir + lbl + '.pdf')

# Process as a Z factor.
scheme = 'gamma'                    # scheme == 'gamma' or 'qslash'
F_tree = np.real(getF(L, scheme))                 # tree level projections
F_blks = [
    np.array([[F_tree[0, 0]]]),
    np.array([[F_tree[1, 1], F_tree[1, 2]], [F_tree[2, 1], F_tree[2, 2]]]),
    np.array([[F_tree[3, 3], F_tree[3, 4]], [F_tree[4, 3], F_tree[4, 4]]])
]

def gv_inv(A):
    """
    Inverse of matrix of gvars A. Only implemented for 1x1 and 2x2.
    """
    if A.shape == (1, 1):
        return 1 / A
    detA = A[1, 1] * A[0, 0] - A[0, 1] * A[1, 0]
    Ainv = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], dtype = object)
    return Ainv / detA

def eval_Zblock_gvar(Lam_blk, Zq, F_tree_blk):
    """
    Evaluates a block of Z from Lambda and Zq. Assumes all inputs are arrays of gv.gvars.
    """
    Lam_blk_inv = gv_inv(Lam_blk)
    return (Zq**2) * (F_tree_blk @ Lam_blk_inv)

def eval_Z_gvar(Lam, Zq):
    # break into blocks
    Lam_blocks = [
        np.array([[Lam[0, 0]]], dtype = object),
        np.array([
            [Lam[1, 1], Lam[1, 2]], 
            [Lam[2, 1], Lam[2, 2]]], 
        dtype = object),
        np.array([
            [Lam[3, 3], Lam[3, 4]], 
            [Lam[4, 3], Lam[4, 4]]], 
        dtype = object)
    ]

    # evaluate each block
    Zblock = []
    for blk_idx in range(3):
        Zblock.append(
            eval_Zblock_gvar(Lam_blocks[blk_idx], Zq, F_blks[blk_idx])
        )
    
    # broadcast back into a matrix
    Z_gvar = np.zeros((5, 5), dtype = object)
    Z_gvar[0, 0] = Zblock[0][0, 0]
    Z_gvar[1:3, 1:3] = Zblock[1]
    Z_gvar[3:5, 3:5] = Zblock[2]
    return Z_gvar

# unpack variables
Zq_gvar, ZV_gvar, ZA_gvar = Zall_out[:, 0], Zall_out[:, 1], Zall_out[:, 2]      # (n_mom)
Zq_band, ZV_band, ZA_band = fit_band[:, 0], fit_band[:, 1], fit_band[:, 2]      # (n_mom, n_band)
Lambda_gvar = np.zeros((n_mom, 5, 5), dtype = object)
Lambda_band = np.zeros((n_mom, n_band, 5, 5), dtype = object)
for ii, mult_idx in enumerate(multiplets):
    Lambda_gvar[:, mult_idx[0], mult_idx[1]] = Zall_out[:, 3 + ii]
    # Lambda_band[:, :, mult_idx[0], mult_idx[1]] = fit_band[:, ii, :]
    Lambda_band[:, :, mult_idx[0], mult_idx[1]] = fit_band[:, 3 + ii, :]

# generate fit band
Z_chiral_gvar = np.zeros((n_mom, 5, 5), dtype = object)
ZbyZVsq_gvar = np.zeros((n_mom, 5, 5), dtype = object)
Z_chiral_band = np.zeros((n_mom, 5, 5, n_band), dtype = object)
Z_chiral_lower = np.zeros((n_mom, 5, 5, n_band), dtype = np.float64)
Z_chiral_upper = np.zeros((n_mom, 5, 5, n_band), dtype = np.float64)
for mom_idx in range(n_mom):
    Z_chiral_gvar[mom_idx, :, :] = eval_Z_gvar(Lambda_gvar[mom_idx, :, :], Zq_gvar[mom_idx])
    ZbyZVsq_gvar[mom_idx, :, :] = Z_chiral_gvar[mom_idx, :, :] / (ZV_gvar[mom_idx]**2)
    for jj in range(n_band):
        Z_chiral_band[mom_idx, :, :, jj] = eval_Z_gvar(Lambda_band[mom_idx, jj], Zq_band[mom_idx, jj])
        Z_chiral_lower[mom_idx, :, :, jj] = gv.mean(Z_chiral_band[mom_idx, :, :, jj]) - gv.sdev(Z_chiral_band[mom_idx, :, :, jj])
        Z_chiral_upper[mom_idx, :, :, jj] = gv.mean(Z_chiral_band[mom_idx, :, :, jj]) + gv.sdev(Z_chiral_band[mom_idx, :, :, jj])

print('Z(amell --> 0) = ' + str(Z_chiral_gvar[mu0_idx]))

Z_chiral = np.zeros((5, 5, n_mom, n_samp), dtype = np.float64)
Z_chiral_mu = np.zeros((5, 5, n_mom), dtype = np.float64)
Z_chiral_std = np.zeros((5, 5, n_mom), dtype = np.float64)
for mom_idx in range(n_mom):
    for ii, mult_idx in enumerate(multiplets):
        dist = [Z_chiral_gvar[mom_idx, mult_idx[0], mult_idx[1]].mean, Z_chiral_gvar[mom_idx, mult_idx[0], mult_idx[1]].sdev]
        Z_chiral[mult_idx[0], mult_idx[1], mom_idx, :] = gen_fake_ensemble(dist, n_samples = n_samp, s = ii + 5)

for ii, mult_idx in enumerate(multiplets):
    label = r'$\mathcal{Z}_{' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + r'}'
    path = plot_dir + 'Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + '.pdf'
    plot_fit_out(Z_mu[mu0_idx, mult_idx[0], mult_idx[1]], Z_std[mu0_idx, mult_idx[0], mult_idx[1]], Z_chiral_gvar[mu0_idx, mult_idx[0], \
        mult_idx[1]].mean, Z_chiral_gvar[mu0_idx, mult_idx[0], mult_idx[1]].sdev, xx, Z_chiral_lower[mu0_idx, mult_idx[0], mult_idx[1]], \
        Z_chiral_upper[mu0_idx, mult_idx[0], mult_idx[1]], label, path, plt_band = True)

# (Zq, ZV, ZA, F11, ..., F54, Z11, ..., Z54)
n_total_vars = n_vars + len(multiplets)
Ztotal = np.zeros((n_mom, n_total_vars), dtype = object)
Ztotal_mu = np.zeros((n_mom, n_total_vars), dtype = np.float64)
Ztotal_cov = np.zeros((n_mom, n_total_vars, n_total_vars), dtype = np.float64)
Ztotal_dist = np.zeros((n_mom, n_total_vars, n_samp), dtype = np.float64)
Ztotal[:, :n_vars] = Zall_out[:, :n_vars]
for ii, mult_idx in enumerate(multiplets):
    Ztotal[:, n_vars + ii] = Z_chiral_gvar[:, mult_idx[0], mult_idx[1]]
for mom_idx in range(n_mom):
    Ztotal_mu[mom_idx] = gv.mean(Ztotal[mom_idx])
    Ztotal_cov[mom_idx] = gv.evalcov(Ztotal[mom_idx])
    Ztotal_dist[mom_idx, :, :] = gen_corr_dist(Ztotal_mu[mom_idx], Ztotal_cov[mom_idx], n_samples = n_samp)
Zq_dist = Ztotal_dist[:, 0, :]          # (n_mom, n_samp)
ZV_dist = Ztotal_dist[:, 1, :]
ZA_dist = Ztotal_dist[:, 2, :]
Lambda_dist = np.zeros((n_mom, 5, 5, n_samp), dtype = np.float64)
Z_dist = np.zeros((n_mom, 5, 5, n_samp), dtype = np.float64)
ZbyZVsq_dist = np.zeros((n_mom, 5, 5, n_samp), dtype = np.float64)
for ii, mult_idx in enumerate(multiplets):
    Lambda_dist[:, mult_idx[0], mult_idx[1], :] = Ztotal_dist[:, 3 + ii, :]
    Z_dist[:, mult_idx[0], mult_idx[1], :] = Ztotal_dist[:, n_vars + ii, :]
    ZbyZVsq_dist[:, mult_idx[0], mult_idx[1], :] = np.einsum('qb,qb->qb', Z_dist[:, mult_idx[0], mult_idx[1], :], ZV_dist**(-2))
print('Z/ZV^2 (amell --> 0) = ' + str(ZbyZVsq_gvar[mu0_idx]))

# save results of chiral extrapolation and of interpolation
fchi_out = h5py.File(out_path, 'w')
fchi_out['momenta'] = k_list
fchi_out['Zq/values'] = Zq_dist
fchi_out['ZV/value'] = ZV_dist
fchi_out['ZA/value'] = ZA_dist
for ii, mult_idx in enumerate(multiplets):
    fchi_out['Lambda' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Lambda_dist[:, mult_idx[0], mult_idx[1], :]
    fchi_out['Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Z_dist[:, mult_idx[0], mult_idx[1], :]
    datapath = 'O' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)
    # fchi_out[datapath + '/ZijZVm2'] = np.einsum('qb,qb->qb', Z_dist[:, mult_idx[0], mult_idx[1], :], ZV_dist**(-2))
    fchi_out[datapath + '/ZijZVm2'] = ZbyZVsq_dist[:, mult_idx[0], mult_idx[1], :]
print('amell --> 0 extrapolation saved at: ' + out_path)
fchi_out.close()
