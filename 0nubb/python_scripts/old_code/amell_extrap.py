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
const = True
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
print(k_list_ens)
assert np.array_equal(k_list_ens[0], k_list_ens[1])         # make sure each ensemble has same momentum modes
k_list = k_list_ens[0]
mom_list = np.array([L.to_linear_momentum(k, datatype=np.float64) for k in k_list])
mu_list = np.array([get_energy_scale_linear(q, a_fm, L) for q in k_list])
print('Energy scales')
print(mu_list)
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

# assemble data into correct form
def make_data(corr, domain = mass_list):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_ens, n_boot). Here lam is the shrinkage parameter λ, which is 
    set to 1 (fully correlated) by default. 
    """
    d = {m : corr[ii, :] for ii, m in enumerate(domain)}
    df = pd.DataFrame(d)
    mean = np.array(df.mean())
    full_cov = np.array(df.cov())
    cov = np.zeros(full_cov.shape, dtype = np.float64)
    for i in range(full_cov.shape[0]):
        cov[i, i] = full_cov[i, i]
    return domain, gv.gvar(mean, cov)

# assemble data into correct form
def make_data_bootstrap(corr, bidx, domain = mass_list):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_ens, n_boot). Here lam is the shrinkage parameter λ, which is 
    set to 1 (fully correlated) by default. 
    """
    d = {m : corr[ii, :] for ii, m in enumerate(domain)}
    df = pd.DataFrame(d)
    # mean = np.array(df.mean())
    boot = corr[:, bidx]
    full_cov = np.array(df.cov())
    cov = np.zeros(full_cov.shape, dtype = np.float64)
    for i in range(full_cov.shape[0]):
        cov[i, i] = full_cov[i, i]
    return domain, gv.gvar(boot, cov)

def fcn(m, p):
    """Constant or linear fitting function f(m; c) = c0 + c1 m."""
    c0 = p['c0']
    if const:
        return c0 + 0*m
    c1 = p['c1']
    return c0 + c1*m

def get_fit_band(params, fcn, xlims):
    xx = np.linspace(xlims[0], xlims[1], n_band)
    fx = fcn(xx, params)
    # print(fx)
    fx_lower = gv.mean(fx) - gv.sdev(fx)
    fx_upper = gv.mean(fx) + gv.sdev(fx)
    return xx, fx_lower, fx_upper, fx

# fit data to model
def fit_data_model(rcs):
    """
    Fits data Z to an arbitrary model by minimizing the uncorrelated chi^2.
    """
    dom, Zfit = make_data(rcs)
    Z_cv = np.mean(Zfit)
    p0 = {'c0' : Z_cv.mean} if const else {'c0' : Z_cv.mean, 'c1' : 0.0}
    print('Initial guess: ' + str(p0))
    fit = lsqfit.nonlinear_fit(data = (dom, Zfit), fcn = fcn, prior = None, p0 = p0)
    print(fit)
    return fit

def fit_data_bootstrap(rcs, bidx):
    """
    Fits data rcs at bootstrap bidx to an arbitrary model by minimizing the uncorrelated chi^2.
    """
    dom, Zfit = make_data_bootstrap(rcs, bidx)
    Z_cv = np.mean(Zfit)
    p0 = {'c0' : Z_cv.mean} if const else {'c0' : Z_cv.mean, 'c1' : 0.0}
    fit = lsqfit.nonlinear_fit(data = (dom, Zfit), fcn = fcn, prior = None, p0 = p0)
    # print(fit)
    return fit

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

Zq_gvar = np.zeros((n_mom), dtype = object)
Zq_fit_mu = np.zeros((n_mom), dtype = np.float64)
Zq_fit_std = np.zeros((n_mom), dtype = np.float64)
Zq_fit = np.zeros((n_mom, n_samp), dtype = np.float64)
Zq_params = []
Zq_band_list = np.zeros((n_mom, n_band), dtype = object)
print('Chiral extrapolation for Zq.')
for mom_idx in range(n_mom):
    fout = fit_data_model(Zq_list[:, mom_idx, :])
    Zq_params.append(fout.p)
    Zq_gvar[mom_idx] = fout.p['c0']    # this is our value for Zq
    Zq_fit_mu[mom_idx] = fout.p['c0'].mean
    Zq_fit_std[mom_idx] = fout.p['c0'].sdev
    print('Zq '+stem+' at $am_\ell$ = 0 for momentum idx '+str(mom_idx)+': ' + export_float_latex(Zq_fit_mu[mom_idx], Zq_fit_std[mom_idx], sf = 2))
    Zq_fit[mom_idx, :] = gen_fake_ensemble([Zq_fit_mu[mom_idx], Zq_fit_std[mom_idx]], n_samples = n_samp, s = 2)
    _, _, _, Zq_band_list[mom_idx, :] = get_fit_band(fout.p, fcn, xlimits)
xx, fx_lower, fx_upper, Zq_band = get_fit_band(Zq_params[mu0_idx], fcn, xlimits)
plot_fit_out(Zq_mu[mu0_idx], Zq_std[mu0_idx], Zq_fit_mu[mu0_idx], Zq_fit_std[mu0_idx], xx, fx_lower, fx_upper, \
            r'$\mathcal{Z}_q^\mathrm{RI}', plot_dir + 'Zq.pdf')

ZV_fit_mu = np.zeros((n_mom), dtype = np.float64)
ZV_fit_std = np.zeros((n_mom), dtype = np.float64)
ZV_fit = np.zeros((n_mom, n_samp), dtype = np.float64)
ZV_params = []
print('Chiral extrapolation for ZV.')
for mom_idx in range(n_mom):
    fout = fit_data_model(ZV_list[:, mom_idx, :])
    ZV_params.append(fout.p)
    ZV_fit_mu[mom_idx] = fout.p['c0'].mean
    ZV_fit_std[mom_idx] = fout.p['c0'].sdev
    print('ZV '+stem+' at $am_\ell$ = 0 for momentum idx '+str(mom_idx)+': ' + export_float_latex(ZV_fit_mu[mom_idx], ZV_fit_std[mom_idx], sf = 2))
    ZV_fit[mom_idx, :] = gen_fake_ensemble([ZV_fit_mu[mom_idx], ZV_fit_std[mom_idx]], n_samples = n_samp, s = 3)
xx, fx_lower, fx_upper, _ = get_fit_band(ZV_params[mu0_idx], fcn, xlimits)
plot_fit_out(ZV_mu[mu0_idx], ZV_std[mu0_idx], ZV_fit_mu[mu0_idx], ZV_fit_std[mu0_idx], xx, fx_lower, fx_upper, r'$\mathcal{Z}_V', plot_dir + 'ZV.pdf')

ZA_fit_mu = np.zeros((n_mom), dtype = np.float64)
ZA_fit_std = np.zeros((n_mom), dtype = np.float64)
ZA_fit = np.zeros((n_mom, n_samp), dtype = np.float64)
ZA_params = []
print('Chiral extrapolation for ZA.')
for mom_idx in range(n_mom):
    fout = fit_data_model(ZA_list[:, mom_idx, :])
    ZA_params.append(fout.p)
    ZA_fit_mu[mom_idx] = fout.p['c0'].mean
    ZA_fit_std[mom_idx] = fout.p['c0'].sdev
    print('ZA '+stem+' at $am_\ell$ = 0 for momentum idx '+str(mom_idx)+': ' + export_float_latex(ZA_fit_mu[mom_idx], ZA_fit_std[mom_idx], sf = 2))
    ZA_fit[mom_idx, :] = gen_fake_ensemble([ZA_fit_mu[mom_idx], ZA_fit_std[mom_idx]], n_samples = n_samp, s = 4)
xx, fx_lower, fx_upper, _ = get_fit_band(ZA_params[mu0_idx], fcn, xlimits)
plot_fit_out(ZA_mu[mu0_idx], ZA_std[mu0_idx], ZA_fit_mu[mu0_idx], ZA_fit_std[mu0_idx], xx, fx_lower, fx_upper, r'$\mathcal{Z}_A', plot_dir + 'ZA.pdf')

# TODO think about correlation: Zq and Znm should be correlated, right? May have to change the seeding and add the correlation into the gvars
# Few things to change: 
# a) make sure the seeds are the same for each
# b) Add correlations between Lambda_gvar and Zq_gvar before we construct Z_chiral_gvar. Likely want to get the correlation from bootstrap fits?

Lambda_gvar = np.zeros((n_mom, 5, 5), dtype = object)
Lambda_fit_mu = np.zeros((n_mom, 5, 5), dtype = np.float64)
Lambda_fit_std = np.zeros((n_mom, 5, 5), dtype = np.float64)
Lambda_fit = np.zeros((n_mom, 5, 5, n_samp), dtype = np.float64)
Lambda_params = np.zeros((n_mom, 5, 5), dtype = object)
fx_lower_list, fx_upper_list = np.zeros((n_mom, 5, 5, n_band), dtype = np.float64), np.zeros((n_mom, 5, 5, n_band), dtype = np.float64)
Lambda_x_band = np.zeros((n_mom, 5, 5, n_band), dtype = object)
print('Chiral extrapolation for F_ij.')
for mom_idx in range(n_mom):
    print('Momentum index ' + str(mom_idx))
    for ii, mult_idx in enumerate(multiplets):
        print('Fitting F' + str(mult_idx[0]) + str(mult_idx[1]))
        fout = fit_data_model(Lambda_list[:, mult_idx[0], mult_idx[1], mom_idx, :])
        Lambda_params[mom_idx, mult_idx[0], mult_idx[1]] = fout.p
        Lambda_gvar[mom_idx, mult_idx[0], mult_idx[1]] = fout.p['c0']
        Lambda_fit_mu[mom_idx, mult_idx[0], mult_idx[1]] = fout.p['c0'].mean
        Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]] = fout.p['c0'].sdev
        print('F' + str(mult_idx[0]) + str(mult_idx[1]) + ', ' + stem + ' at $am_\ell$ = 0 for momentum idx ' + str(mom_idx) + ': ' \
                    + export_float_latex(Lambda_fit_mu[mom_idx, mult_idx[0], mult_idx[1]], Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]], sf = 2))
        Lambda_fit[mom_idx, mult_idx[0], mult_idx[1], :] = gen_fake_ensemble([Lambda_fit_mu[mom_idx, mult_idx[0], mult_idx[1]], \
                    Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]]], n_samples = n_samp, s = ii + 5)  # different seed --> uncorrelated
        xx, fx_lower, fx_upper, Lam_x = get_fit_band(fout.p, fcn, xlimits)
        fx_lower_list[mom_idx, mult_idx[0], mult_idx[1], :] = fx_lower
        fx_upper_list[mom_idx, mult_idx[0], mult_idx[1], :] = fx_upper
        Lambda_x_band[mom_idx, mult_idx[0], mult_idx[1], :] = Lam_x
for ii, mult_idx in enumerate(multiplets):
    label = r'$F_{' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + r'}'
    path = plot_dir + 'F_matrix/F' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + '.pdf'
    plot_fit_out(Lambda_mu[mu0_idx, mult_idx[0], mult_idx[1]], Lambda_std[mu0_idx, mult_idx[0], mult_idx[1]], Lambda_fit_mu[mu0_idx, mult_idx[0], \
        mult_idx[1]], Lambda_fit_std[mu0_idx, mult_idx[0], mult_idx[1]], xx, fx_lower_list[mu0_idx, mult_idx[0], mult_idx[1]], \
        fx_upper_list[mu0_idx, mult_idx[0], mult_idx[1]], label, path)

################################################################################
############################### Add correlations ###############################
################################################################################

# Important question here: correlations should survive after the uncorrelated 
# am_ell --> 0 extrapolation, right? If they do, then maybe the best way to do this is 
# to fit each bootstrap, then spread the bootstrapped ensembles to the correct error, 
# and use that as our final distribution.

# Fit each bootstrap to get correlations before we solve for Znm
Zq_dist = np.zeros((n_mom, n_boot), dtype = np.float64)
ZV_dist = np.zeros((n_mom, n_boot), dtype = np.float64)
ZA_dist = np.zeros((n_mom, n_boot), dtype = np.float64)
Lambda_dist = np.zeros((n_mom, 5, 5, n_boot), dtype = np.float64)
print('Fitting each bootstrap to determine correlations.')
for mom_idx in range(n_mom):
    for bidx in range(n_boot):
        Zq_dist[mom_idx, bidx] = fit_data_bootstrap(Zq_list[:, mom_idx, :], bidx).p['c0'].mean
        ZV_dist[mom_idx, bidx] = fit_data_bootstrap(ZV_list[:, mom_idx, :], bidx).p['c0'].mean
        ZA_dist[mom_idx, bidx] = fit_data_bootstrap(ZA_list[:, mom_idx, :], bidx).p['c0'].mean
        for ii, mult_idx in enumerate(multiplets):
            Lambda_dist[mom_idx, mult_idx[0], mult_idx[1], bidx] = \
                fit_data_bootstrap(Lambda_list[:, mult_idx[0], mult_idx[1], mom_idx, :], bidx).p['c0'].mean
    print('Spreading sdev on Zq. Original sdev: ' + str(np.std(Zq_dist[mom_idx], ddof = 1)) + ', new sdev: ' + str(Zq_fit_std[mom_idx]))
    Zq_dist[mom_idx] = spread_boots(Zq_dist[mom_idx], Zq_fit_std[mom_idx])
    ZV_dist[mom_idx] = spread_boots(ZV_dist[mom_idx], ZV_fit_std[mom_idx])
    ZA_dist[mom_idx] = spread_boots(ZA_dist[mom_idx], ZA_fit_std[mom_idx])
    for ii, mult_idx in enumerate(multiplets):
        Lambda_dist[mom_idx, mult_idx[0], mult_idx[1]] = \
            spread_boots(Lambda_dist[mom_idx, mult_idx[0], mult_idx[1]], Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]])

# print('gvar standard deviation for Zq at all momenta: ' + str(gv.sdev(Zq_gvar)))
# print('Bootstrap standard deviation for Zq at all momenta: ' + str(np.std(Zq_dist, axis = 1, ddof = 1)))
# for ii, mult_idx in enumerate(multiplets):
#     print('gvar standard deviation for Z' + str(mult_idx) + ' at all momenta: ' + str(gv.sdev(Lambda_gvar[:, mult_idx[0], mult_idx[1]])))
#     print('Bootstrap standard deviation for Z' + str(mult_idx) + ' at all momenta: ' + str(np.std(Lambda_dist[:, mult_idx[0], mult_idx[1], :], axis = 1, ddof = 1)))

# get covariance between Zq and Lambda_nm
print('Adding correlations')

all_boots = np.zeros((n_mom, 3 + len(multiplets), n_boot), dtype = np.float64)
all_boots[:, 0, :] = Zq_dist[:, :]
all_boots[:, 1, :] = ZV_dist[:, :]
all_boots[:, 2, :] = ZA_dist[:, :]
for ii, mult_idx in enumerate(multiplets):
    all_boots[:, 3 + ii, :] = Lambda_dist[:, mult_idx[0], mult_idx[1], :]
# now compute covariance of all_boots

Zq_Lambda_covar = np.zeros((n_mom, 5, 5, 2, 2), dtype = np.float64)
for mom_idx in range(n_mom):
    for ii, mult_idx in enumerate(multiplets):
        covar = np.cov(Zq_dist[mom_idx], Lambda_dist[mom_idx, mult_idx[0], mult_idx[1], :])
        Zq_Lambda_covar[mom_idx, mult_idx[0], mult_idx[1], :, :] = covar
        # tmp = gv.correlate([Zq_gvar[mom_idx], Lambda_gvar[mom_idx, mult_idx[0], mult_idx[1]]], covar)
        Zq_gvar[mom_idx], Lambda_gvar[mom_idx, mult_idx[0], mult_idx[1]] = gv.correlate([Zq_gvar[mom_idx], Lambda_gvar[mom_idx, mult_idx[0], mult_idx[1]]], covar)

# use spread bootstrapped distributions as sample distributions for remainder of analysis


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

# generate fit band
Z_chiral_gvar = np.zeros((n_mom, 5, 5), dtype = object)
Z_chiral_band = np.zeros((n_mom, 5, 5, n_band), dtype = object)
Z_chiral_lower = np.zeros((n_mom, 5, 5, n_band), dtype = np.float64)
Z_chiral_upper = np.zeros((n_mom, 5, 5, n_band), dtype = np.float64)
for qidx in range(n_mom):
    Z_chiral_gvar[qidx, :, :] = eval_Z_gvar(Lambda_gvar[qidx, :, :], Zq_gvar[qidx])
    for jj in range(n_band):
        Lam_x = Lambda_x_band[qidx, :, :, jj]
        Zq_jj = Zq_band_list[qidx, jj]
        Z_chiral_band[qidx, :, :, jj] = eval_Z_gvar(Lam_x, Zq_jj)
        Z_chiral_lower[qidx, :, :, jj] = gv.mean(Z_chiral_band[qidx, :, :, jj]) - gv.sdev(Z_chiral_band[qidx, :, :, jj])
        Z_chiral_upper[qidx, :, :, jj] = gv.mean(Z_chiral_band[qidx, :, :, jj]) + gv.sdev(Z_chiral_band[qidx, :, :, jj])

Z_chiral = np.zeros((5, 5, n_mom, n_samp), dtype = np.float64)
Z_chiral_mu = np.zeros((5, 5, n_mom), dtype = np.float64)
Z_chiral_std = np.zeros((5, 5, n_mom), dtype = np.float64)
for qidx in range(n_mom):
    for ii, mult_idx in enumerate(multiplets):
        dist = [Z_chiral_gvar[qidx, mult_idx[0], mult_idx[1]].mean, Z_chiral_gvar[qidx, mult_idx[0], mult_idx[1]].sdev]
        Z_chiral[mult_idx[0], mult_idx[1], qidx, :] = gen_fake_ensemble(dist, n_samples = n_samp, s = ii + 5)

for ii, mult_idx in enumerate(multiplets):
    label = r'$\mathcal{Z}_{' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + r'}'
    path = plot_dir + 'Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + '.pdf'
    plot_fit_out(Z_mu[mu0_idx, mult_idx[0], mult_idx[1]], Z_std[mu0_idx, mult_idx[0], mult_idx[1]], Z_chiral_gvar[mu0_idx, mult_idx[0], \
        mult_idx[1]].mean, Z_chiral_gvar[mu0_idx, mult_idx[0], mult_idx[1]].sdev, xx, Z_chiral_lower[mu0_idx, mult_idx[0], mult_idx[1]], \
        Z_chiral_upper[mu0_idx, mult_idx[0], mult_idx[1]], label, path, plt_band = True)

# def plot_fit_out(cvs, sigmas, extrap_mu, extrap_sigma, fx_cvs, fx_stds, ylabel, path, plt_band = True):
#     if plt_band:
#         data_window = [min(np.min(fx_cvs - fx_stds), np.min(cvs - sigmas), extrap_mu - extrap_sigma), \
#                         max(np.max(fx_cvs + fx_stds), np.max(cvs + sigmas), extrap_mu + extrap_sigma)]
#     else:
#         data_window = [min(np.min(cvs - sigmas), extrap_mu - extrap_sigma), max(np.max(cvs + sigmas), extrap_mu + extrap_sigma)]
#     Delta_window = data_window[1] - data_window[0]
#     ylimits = [data_window[0] - Delta_window * scale_factors[0], data_window[1] + Delta_window * scale_factors[1]]
#     with sns.plotting_context('paper'):
#         fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
#         plt.figure(figsize = fig_size)
#         plt.vlines(0.0, ylimits[0], ylimits[1], linestyles = 'dashed', label = '$am_\ell = 0$', linewidth = style['ebar_width'], color = 'k')
#         _, caps, _ = plt.errorbar(mass_list, cvs, sigmas, fmt = '.', c = 'r', \
#                 label = 'Data', capsize = style['endcaps'], markersize = style['markersize'], \
#                 elinewidth = style['ebar_width'])
#         for cap in caps:
#             cap.set_markeredgewidth(style['ecap_width'])
#         _, caps, _ = plt.errorbar([0.0], [extrap_mu], [extrap_sigma], fmt = '.', c = fill_color, \
#                 capsize = style['endcaps'], markersize = style['markersize'], \
#                 elinewidth = style['ebar_width'])
#         for cap in caps:
#             cap.set_markeredgewidth(style['ecap_width'])
#         if plt_band:
#             plt.fill_between(x_band, fx_cvs + fx_stds, fx_cvs - fx_stds, color = fill_color, alpha = 0.2, linewidth = 0.0, label = 'Extrapolation')
#         plt.xlabel(xlabel, fontsize = style['fontsize'])
#         ylabel_dec = ylabel + r'\; (a=' + a_label + r')$'
#         print(ylabel_dec)
#         plt.ylabel(ylabel_dec, fontsize = style['fontsize'])
#         plt.xlim(xlimits)
#         plt.ylim(ylimits)         # set this after we figure out the ylimits
#         ax = plt.gca()
#         ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
#         ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
#         ax.set_xticks(xtick_locs)
#         ax.set_xticklabels(xtick_labels)
#         for spine in spinedirs:
#             ax.spines[spine].set_linewidth(style['axeswidth'])
#         plt.xticks(fontsize = style['fontsize'])
#         plt.yticks(fontsize = style['fontsize'])
#         plt.legend(prop={'size': style['fontsize'] * 0.8})
#         plt.tight_layout()
#         plt.savefig(path, bbox_inches='tight')
#         print('Plot ' + ylabel + ' saved at: \n   ' + path)
#         plt.close()
# 
# # fit data to model
# def fit_data_model(cvs, sigmas, model):
#     """
#     Fits data Z to an arbitrary model by minimizing the correlated chi^2.
#     lam is the parameter for linear shrinkage, i.e. lam = 0 is the uncorrelated covariance, and lam = 1 is the
#     original covariance.
#     """
#     fitter = UncorrFitter(mass_list, cvs, sigmas, model)
#     fit_out = fitter.fit()
#     print('Best fit coeffs: ' + str(fit_out[0]))
#     print('chi^2 / dof: ' + str(fit_out[1] / fit_out[2]))
#     print('Parameter covariance: ' + str(fit_out[3]))
#     return fit_out, fitter
# 
# def linear_model(params):
#     def model(m):
#         return params[0] + params[1] * m
#     return model
# fit_model = Model(linear_model, 2, ['', 'm'], ['c0', 'c1'])
# 
# Zq_fit_mu = np.zeros((n_mom), dtype = np.float64)
# Zq_fit_std = np.zeros((n_mom), dtype = np.float64)
# Zq_fit = np.zeros((n_mom, n_samp), dtype = np.float64)
# Zq_params = []
# Zq_param_covar = []
# Zq_fitters = []
# print('Chiral extrapolation for Zq.')
# for mom_idx in range(n_mom):
#     fout, fitter = fit_data_model(Zq_mu[mom_idx], Zq_std[mom_idx], fit_model)
#     params, param_cov = fout[0], fout[3]
#     Zq_params.append(params)
#     Zq_param_covar.append(param_cov)
#     Zq_fitters.append(fitter)
#     Zq_fit_mu[mom_idx] = params[0]
#     Zq_fit_std[mom_idx] = np.sqrt(param_cov[0, 0])
#     print('Zq ' + stem + ' at $am_\ell$ = 0 for momentum idx ' + str(mom_idx) + ': ' + export_float_latex(Zq_fit_mu[mom_idx], Zq_fit_std[mom_idx], sf = 2))
#     Zq_fit[mom_idx, :] = gen_fake_ensemble([Zq_fit_mu[mom_idx], Zq_fit_std[mom_idx]], n_samples = n_samp)
# Zq_band_cvs, Zq_band_stds = Zq_fitters[mu0_idx].gen_fit_band(Zq_params[mu0_idx], Zq_param_covar[mu0_idx], x_band)
# plot_fit_out(Zq_mu[mu0_idx], Zq_std[mu0_idx], Zq_fit_mu[mu0_idx], Zq_fit_std[mu0_idx], Zq_band_cvs, Zq_band_stds, \
#             r'$\mathcal{Z}_q^\mathrm{RI}', plot_dir + 'Zq.pdf')
# 
# ZV_fit_mu = np.zeros((n_mom), dtype = np.float64)
# ZV_fit_std = np.zeros((n_mom), dtype = np.float64)
# ZV_fit = np.zeros((n_mom, n_samp), dtype = np.float64)
# ZV_params = []
# ZV_param_covar = []
# ZV_fitters = []
# print('Chiral extrapolation for ZV.')
# for mom_idx in range(n_mom):
#     fout, fitter = fit_data_model(ZV_mu[mom_idx], ZV_std[mom_idx], fit_model)
#     params, param_cov = fout[0], fout[3]
#     ZV_params.append(params)
#     ZV_param_covar.append(param_cov)
#     ZV_fitters.append(fitter)
#     ZV_fit_mu[mom_idx] = params[0]
#     ZV_fit_std[mom_idx] = np.sqrt(param_cov[0, 0])
#     print('ZV ' + stem + ' at $am_\ell$ = 0 for momentum idx ' + str(mom_idx) + ': ' + export_float_latex(ZV_fit_mu[mom_idx], ZV_fit_std[mom_idx], sf = 2))
#     ZV_fit[mom_idx, :] = gen_fake_ensemble([ZV_fit_mu[mom_idx], ZV_fit_std[mom_idx]], n_samples = n_samp)
# ZV_band_cvs, ZV_band_stds = ZV_fitters[mu0_idx].gen_fit_band(ZV_params[mu0_idx], ZV_param_covar[mu0_idx], x_band)
# plot_fit_out(ZV_mu[mu0_idx], ZV_std[mu0_idx], ZV_fit_mu[mu0_idx], ZV_fit_std[mu0_idx], ZV_band_cvs, ZV_band_stds, r'$\mathcal{Z}_V', plot_dir + 'ZV.pdf')
# 
# ZA_fit_mu = np.zeros((n_mom), dtype = np.float64)
# ZA_fit_std = np.zeros((n_mom), dtype = np.float64)
# ZA_fit = np.zeros((n_mom, n_samp), dtype = np.float64)
# ZA_params = []
# ZA_param_covar = []
# ZA_fitters = []
# print('Chiral extrapolation for ZA.')
# for mom_idx in range(n_mom):
#     fout, fitter = fit_data_model(ZA_mu[mom_idx], ZA_std[mom_idx], fit_model)
#     params, param_cov = fout[0], fout[3]
#     ZA_params.append(params)
#     ZA_param_covar.append(param_cov)
#     ZA_fitters.append(fitter)
#     ZA_fit_mu[mom_idx] = params[0]
#     ZA_fit_std[mom_idx] = np.sqrt(param_cov[0, 0])
#     print('ZA ' + stem + ' at $am_\ell$ = 0 for momentum idx ' + str(mom_idx) + ': ' + export_float_latex(ZA_fit_mu[mom_idx], ZA_fit_std[mom_idx], sf = 2))
#     ZA_fit[mom_idx, :] = gen_fake_ensemble([ZA_fit_mu[mom_idx], ZA_fit_std[mom_idx]], n_samples = n_samp)
# ZA_band_cvs, ZA_band_stds = ZA_fitters[mu0_idx].gen_fit_band(ZA_params[mu0_idx], ZA_param_covar[mu0_idx], x_band)
# plot_fit_out(ZA_mu[mu0_idx], ZA_std[mu0_idx], ZA_fit_mu[mu0_idx], ZA_fit_std[mu0_idx], ZA_band_cvs, ZA_band_stds, r'$\mathcal{Z}_A', plot_dir + 'ZA.pdf')
# 
# perform and save chiral extrapolation on Lambda
# Lambda_fit_mu = np.zeros((n_mom, 5, 5), dtype = np.float64)
# Lambda_fit_std = np.zeros((n_mom, 5, 5), dtype = np.float64)
# Lambda_fit = np.zeros((n_mom, 5, 5, n_samp), dtype = np.float64)
# Lambda_params = np.zeros((n_mom, 5, 5, 2), dtype = np.float64)
# Lambda_param_covar = np.zeros((n_mom, 5, 5, 2, 2), dtype = np.float64)
# Lambda_fitters = []
# print('Chiral extrapolation for F_ij.')
# for mom_idx in range(n_mom):
#     print('Momentum index ' + str(mom_idx))
#     Lambda_fitters.append([])
#     for ii, mult_idx in enumerate(multiplets):
#         print('Fitting F' + str(mult_idx[0]) + str(mult_idx[1]))
#         fout, fitter = fit_data_model(Lambda_mu[mom_idx, mult_idx[0], mult_idx[1]], Lambda_std[mom_idx, mult_idx[0], mult_idx[1]], fit_model)
#         params, param_cov = fout[0], fout[3]
#         Lambda_params[mom_idx, mult_idx[0], mult_idx[1], :] = params[:]
#         Lambda_param_covar[mom_idx, mult_idx[0], mult_idx[1], :, :] = param_cov[:, :]
#         Lambda_fitters[mom_idx].append(fitter)
#         Lambda_fit_mu[mom_idx, mult_idx[0], mult_idx[1]] = params[0]
#         Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]] = np.sqrt(param_cov[0, 0])
#         print('F' + str(mult_idx[0]) + str(mult_idx[1]) + ', ' + stem + ' at $am_\ell$ = 0 for momentum idx ' + str(mom_idx) + ': ' \
#                     + export_float_latex(Lambda_fit_mu[mom_idx, mult_idx[0], mult_idx[1]], Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]], sf = 2))
#         Lambda_fit[mom_idx, mult_idx[0], mult_idx[1], :] = gen_fake_ensemble([Lambda_fit_mu[mom_idx, mult_idx[0], mult_idx[1]], \
#                     Lambda_fit_std[mom_idx, mult_idx[0], mult_idx[1]]], n_samples = n_samp)
# for ii, mult_idx in enumerate(multiplets):
#     label = r'$F_{' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + r'}'
#     path = plot_dir + 'F_matrix/F' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + '.pdf'
#     F_band_cvs, F_band_sigmas = Lambda_fitters[mu0_idx][ii].gen_fit_band(Lambda_params[mu0_idx, mult_idx[0], mult_idx[1]], \
#                                     Lambda_param_covar[mu0_idx, mult_idx[0], mult_idx[1]], x_band)
#     plot_fit_out(Lambda_mu[mu0_idx, mult_idx[0], mult_idx[1]], Lambda_std[mu0_idx, mult_idx[0], mult_idx[1]], Lambda_fit_mu[mu0_idx, mult_idx[0], mult_idx[1]], \
#             Lambda_fit_std[mu0_idx, mult_idx[0], mult_idx[1]], F_band_cvs, F_band_sigmas, label, path)
# 
# # Process as a Z factor.
# scheme = 'gamma'                    # scheme == 'gamma' or 'qslash'
# F_tree = getF(L, scheme)                 # tree level projections
# Z_chiral = np.zeros((5, 5, n_mom, n_samp), dtype = np.float64)
# Z_chiral_mu = np.zeros((5, 5, n_mom), dtype = np.float64)
# Z_chiral_std = np.zeros((5, 5, n_mom), dtype = np.float64)
# for mom_idx in range(n_mom):
#     for ii in range(n_samp):
#         Lambda_inv = np.linalg.inv(Lambda_fit[mom_idx, :, :, ii])
#         Z_chiral[:, :, mom_idx, ii] = (Zq_fit[mom_idx, ii] ** 2) * np.einsum('ik,kj->ij', F_tree, Lambda_inv)
# print(Z_chiral.shape)
# Z_chiral_mu = np.mean(Z_chiral, axis = 3)
# Z_chiral_std = np.std(Z_chiral, axis = 3, ddof = 1)
# print(Z_chiral_mu.shape)
# for ii, mult_idx in enumerate(multiplets):
#     label = r'$\mathcal{Z}_{' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + r'}'
#     path = plot_dir + 'Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + '.pdf'
#     # Need to figure out how to generate the fit band
#     plot_fit_out(Z_mu[mu0_idx, mult_idx[0], mult_idx[1]], Z_std[mu0_idx, mult_idx[0], mult_idx[1]], Z_chiral_mu[mult_idx[0], mult_idx[1], mu0_idx], \
#                 Z_chiral_std[mult_idx[0], mult_idx[1], mu0_idx], 'FIT_BAND_CVS', 'FIT_BAND_STDS', label, path, plt_band = False)

# save results of chiral extrapolation and of interpolation
fchi_out = h5py.File(out_path, 'w')
fchi_out['momenta'] = k_list
fchi_out['Zq/values'] = Zq_fit
fchi_out['ZV/value'] = ZV_fit
fchi_out['ZA/value'] = ZA_fit
for ii, mult_idx in enumerate(multiplets):
    fchi_out['Lambda' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Lambda_fit[:, mult_idx[0], mult_idx[1], :]
    fchi_out['Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Z_chiral[mult_idx[0], mult_idx[1]]
    datapath = 'O' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)
    # fchi_out[datapath + '/ZijZVm2'] = Zij_by_ZVsq[mult_idx[0], mult_idx[1]]
    fchi_out[datapath + '/ZijZVm2'] = np.einsum('qb,qb->qb', Z_chiral[mult_idx[0], mult_idx[1]], ZV_fit**(-2))
print('Chiral extrapolation saved at: ' + out_path)
fchi_out.close()
