################################################################################
# Fits discretization artifacts for nnpp renormalization.                      #
# Note that by convention, in the hdf5 files Zq/ZV is saved as 'Zq', and       #
# Z_{nm}/ZV^2 is saved as 'Znm'.                                               #
################################################################################

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

# inputs to toggle
plot_raw = True                                        # True if we want to plot all raw data, False if only fitted data
apsq = True                                             # x axis to plot against. If apsq, plots against (ap)^2, else plots against mu^2.
RIsMOM = True                                           # True to fit RI/sMOM data, False to fit MSbar data.
# subset_idxs = None
# subset_idxs = list(range(6, 12))
subset_idxs = list(range(0, 12))

# MSbar subsets
# subset_idxs = list(range(2, 13))
# subset_idxs = list(range(4, 13))

if RIsMOM:
    finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_gamma.h5'
    # finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_114105/Z_gamma.h5'
    foutpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/fitsRI.h5'
    stem = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/RIsMOM/'
    Zq_range = np.array([0.66, 0.91])                   # Ranges for plotting
    ZV_range = np.array([0.56, 0.88])
    ZA_range = np.array([0.7, 0.95])
    Z_range = np.array([
        [[0.42, 0.82], [-0.12, 0.0], [-0.015, 0.0], [0.0, 0.05], [0.005, 0.022]],
        [[-0.10, 0.0], [0.4, 0.80], [-0.05, 0.16], [-0.10, 0.0], [0.0, 0.008]],
        [[0.0, 0.015], [0.0, 0.02], [0.2, 0.75], [0.0, 0.16], [-0.005, 0.0]],
        [[0.001, 0.011], [-0.002, 0.001], [0.0, 0.12], [0.35, 0.80], [-0.016, -0.002]],
        [[0.005, 0.02], [-0.003, 0.003], [0.0, 0.06], [-0.12, 0.05], [0.55, 0.82]]
    ])
    # ZbyZVsq_range = np.array([
    #     [[1.0, 1.35], [-0.30, 0.0], [-0.032, 0.0], [0.0, 0.15], [0.0, 0.08]],
    #     [[-0.25, 0.0], [0.95, 1.3], [-0.10, 0.35], [-0.30, 0.0], [0.0, 0.008]],
    #     [[0.0, 0.04], [0.0, 0.035], [0.8, 1.05], [0.0, 0.4], [-0.01, 0.0]],
    #     [[0.0, 0.03], [-0.003, 0.001], [0.0, 0.3], [0.95, 1.1], [-0.03, -0.005]],
    #     [[0.0, 0.06], [-0.008, 0.005], [0.0, 0.15], [-0.32, 0.08], [1.0, 1.6]]
    # ])
else:
    finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_MSbar.h5'
    # finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_114105/Z_MSbar.h5'
    foutpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/fitsMS.h5'
    stem = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/MSbar/'
    Zq_range = np.array([0.66, 0.91])                   # Ranges for plotting
    ZV_range = np.array([0.56, 0.88])
    ZA_range = np.array([0.7, 0.95])
    ZqbyZV_range = np.array([1.05, 1.16])
    Z_range = np.array([
        [[0.45, 0.90], [-0.12, 0.005], [-0.012, 0.005], [0.0, 0.05], [0.005, 0.03]],
        [[-0.10, 0.005], [0.52, 0.82], [0.02, 0.14], [-0.08, 0.005], [-0.005, 0.0035]],
        [[-0.005, 0.008], [0.005, 0.03], [0.60, 0.7], [-0.005, 0.11], [-0.003, 0.005]],
        [[0.002, 0.006], [0.0, 0.0003], [-0.005, 0.08], [0.60, 0.70], [-0.012, 0.02]],
        [[0.007, 0.022], [-0.0001, 0.002], [-0.005, 0.06], [-0.6, 0.22], [0.55, 1.1]]
    ])
    # ZbyZVsq_range = np.array([
    #     [[1.05, 1.35], [-0.30, 0.0], [-0.032, 0.0], [0.0, 0.15], [0.0, 0.08]],
    #     [[-0.25, 0.0], [1.05, 1.3], [0.10, 0.35], [-0.30, 0.0], [0.0, 0.008]],
    #     [[0.0, 0.04], [0.0, 0.035], [0.8, 1.05], [0.0, 0.4], [-0.01, 0.0]],
    #     [[0.0, 0.03], [-0.003, 0.001], [0.0, 0.3], [1.01, 1.07], [-0.03, -0.005]],
    #     [[0.0, 0.06], [-0.008, 0.005], [0.0, 0.15], [-0.32, -0.08], [1.1, 1.6]]
    # ])

F = h5py.File(finpath, 'r')
Lat = Lattice(32, 48)
a = 0.145                                   # fm, placeholder for now
ainv = hbarc / a
mpi = 0.8                                   # MeV, placeholder for now
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
# ZbyZVsq = np.einsum('ijqb,qb->ijqb', Z, 1 / (ZV ** 2))
# ZqbyZV = Zq / ZV

if subset_idxs != None:
    k_list = k_list[subset_idxs]
    mom_list = mom_list[subset_idxs]
    mu_list = mu_list[subset_idxs]
    apsq_list = apsq_list[subset_idxs]
    print('Fitting on momentum subset: ' + str(apsq_list))
    Zq = Zq[subset_idxs]
    ZV = ZV[subset_idxs]
    ZA = ZA[subset_idxs]
    # ZqbyZV = ZqbyZV[subset_idxs]
    Z = Z[:, :, subset_idxs, :]
    # ZbyZVsq = ZbyZVsq[:, :, subset_idxs, :]

Zq_mu = np.mean(Zq, axis = 1)
Zq_sigma = np.std(Zq, axis = 1, ddof = 1)
ZV_mu = np.mean(ZV, axis = 1)
ZV_sigma = np.std(ZV, axis = 1, ddof = 1)
ZA_mu = np.mean(ZA, axis = 1)
ZA_sigma = np.std(ZA, axis = 1, ddof = 1)
Z_mu = np.mean(Z, axis = 3)
Z_sigma = np.std(Z, axis = 3, ddof = 1)
# ZqbyZV_mu = np.mean(ZqbyZV, axis = 1)
# ZqbyZV_sigma = np.std(ZqbyZV, axis = 1, ddof = 1)
# ZbyZVsq_mu = np.mean(ZbyZVsq, axis = 3)
# ZbyZVsq_sigma = np.std(ZbyZVsq, axis = 3, ddof = 1)

x_axis = apsq_list if apsq else mu_list
xlabel = '$(ap)^2$' if apsq else '$\\mu\;(\\mathrm{GeV})$'
# xlimits = [0.0, 6.5] if apsq else [1.0, 4.1]
xlimits = [0.0, 18.0] if apsq else [1.0, 4.1]
asp_ratio = 4/3

print(xlabel + ' Z is evaluated at: ' + str(x_axis))

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
        # plt.ylim(ylimits)
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

# generate plot of fitted data
def plot_rcs_fitted(cvs, sigmas, model, params, param_covar, ylabel, ylimits, path, cols = ['r', 'b', 'c']):
    """
    Plots data with central values cvs and error sigmas, and the best fit band after subtracting off
    discretization artifacts.
    """
    with sns.plotting_context('paper'):
        # n_plts = cvs.shape[0]    # pass in list of plots
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        # overload to plot multiple
        _, caps, _ = plt.errorbar(x_axis, cvs, sigmas, fmt = '.', c = cols[0], label = 'data', \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        params0 = [0.0]
        params0.extend(params[1:])
        subtracted_pts = [cvs[ii] - model.F(params0)(xx)[0] for ii, xx in enumerate(x_axis)]
        # TODO make sigmas include errors on fit parameters
        print(subtracted_pts)
        _, caps, _ = plt.errorbar(x_axis, subtracted_pts, sigmas, fmt = '.', c = cols[1], label = 'sub', \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        # plt.axhline(y = params[0], color = cols[2], linestyle = '-', label = '$Z_0$')
        x_band = np.linspace(xlimits[0], xlimits[1])
        x_band_reg = np.linspace(0.1, xlimits[1])
        extrap_fit = np.array([model.F(params)(xx) for xx in x_band_reg])
        plt.plot(x_band_reg, extrap_fit, c = 'g', label = 'extrap', alpha = 0.5)
        plt.fill_between(x_band, params[0] + np.sqrt(param_covar[0, 0]), params[0] - np.sqrt(param_covar[0, 0]), color = cols[2], alpha = 0.5, \
                linewidth = 0.0, label = '$Z_0$')
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
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Plot ' + ylabel + ' saved at: \n   ' + path)


# package together ZqAV
# ZqAV_data = [Zq, ZV, ZA]
# ZqAV_mu = [Zq_mu, ZV_mu, ZA_mu, ZqbyZV_mu]
# ZqAV_sigma = [Zq_sigma, ZV_sigma, ZA_sigma, ZqbyZV_sigma]
# ZqAV_ranges = [Zq_range, ZV_range, ZA_range, ZqbyZV_range]
# ZqAV_labels = ['$\mathcal{Z}_q^\mathrm{RI}$', '$\mathcal{Z}_V$', '$\mathcal{Z}_A$', '$\mathcal{Z}_q / \mathcal{Z}_V$']
# ZqAV_stems = ['Zq_RI', 'ZV', 'ZA', 'ZqbyZV']
ZqAV_data = [Zq, ZV, ZA]
ZqAV_mu = [Zq_mu, ZV_mu, ZA_mu]
ZqAV_sigma = [Zq_sigma, ZV_sigma, ZA_sigma]
ZqAV_ranges = [Zq_range, ZV_range, ZA_range]
ZqAV_labels = ['$\mathcal{Z}_q / \mathcal{Z}_V$', '$\mathcal{Z}_V$', '$\mathcal{Z}_A$']
ZqAV_stems = ['ZqbyZV', 'ZV', 'ZA']

Zlabels = [['Z' + str(n + 1) + str(m + 1) + 'byZVsq' for m in range(5)] for n in range(5)]
# Zlabels = [['Z' + str(n + 1) + str(m + 1) for m in range(5)] for n in range(5)]
# ZbyZVsqlabels = [['Z' + str(n + 1) + str(m + 1) + 'byZVsq' for m in range(5)] for n in range(5)]
if plot_raw:
    if RIsMOM:
        for ii in range(len(ZqAV_data)):
            plot_rcs_raw(ZqAV_mu[ii], ZqAV_sigma[ii], ZqAV_labels[ii], ZqAV_ranges[ii], stem + 'raw/ZqVA/' + ZqAV_stems[ii] + '.pdf')
    for n, m in itertools.product(range(5), repeat = 2):
        plot_rcs_raw(Z_mu[n, m], Z_sigma[n, m], '$\mathcal{Z}_{' + str(n + 1) + str(m + 1) + '} / \mathcal{Z}_V^2$', Z_range[n, m], \
                        stem + 'raw/Zops/' + Zlabels[n][m] + '.pdf')
        # plot_rcs_raw(ZbyZVsq_mu[n, m], ZbyZVsq_sigma[n, m], '$\mathcal{Z}_{' + str(n + 1) + str(m + 1) + '} / \mathcal{Z}_V^2$', ZbyZVsq_range[n, m],\
        #                 stem + 'raw/ZopsbyZVsq/' + ZbyZVsqlabels[n][m] + '.pdf')

# Start fitting. Use fit forms here.
fout = h5py.File(foutpath, 'w')
ap2 = Model(lambda params : lambda apsq : params[0] * apsq, 1, '(x)', 'c1')
ap4 = Model(lambda params : lambda apsq : params[0] * (apsq ** 2), 1, '(x^2)', 'c2')
ap6 = Model(lambda params : lambda apsq : params[0] * (apsq ** 3), 1, '(x^3)', 'c3')
ap8 = Model(lambda params : lambda apsq : params[0] * (apsq ** 4), 1, '(x^4)', 'c4')
ap10 = Model(lambda params : lambda apsq : params[0] * (apsq ** 5), 1, '(x^5)', 'c5')
ap12 = Model(lambda params : lambda apsq : params[0] * (apsq ** 6), 1, '(x^6)', 'c6')
apm2 = Model(lambda params : lambda apsq : params[0] / apsq, 1, '(1/x)', 'c-1')
logap = Model(lambda params : lambda apsq : params[0] * np.log(apsq), 1, 'log(x)', 'c6')
aplogap = Model(lambda params : lambda apsq : params[0] * apsq * np.log(apsq), 1, '(x*log(x))', 'c7')
# all_fit_forms = [ap2, ap4, ap6, ap8, aplogap]
# all_fit_forms = [ap2, ap4, ap6, ap8, ap10, ap12, apm2]
# all_fit_forms = [ap2, ap4, ap6, ap8, ap10]
all_fit_forms = [ap2, ap4, ap6, apm2]
print('Functional forms: Z0, ' + str(all_fit_forms))

dist_dir = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/RIsMOM/fit_distributions'
all_subsets = [list(range(n, 8)) for n in range(2, 6)]
# loop over subsets: figure out which models to choose at each step. Then look at the distribution of Z0 for each one and save it to dist_dir

# Fit Zq, ZV, ZA, Zq / ZV
lam = 0.0                                   # shrinkage parameter, may want to loop over it.
if RIsMOM:
    ZqAV_fit_vals = []                          # best fit coefficients for Z0
    ZqAV_fit_sigmas = []                        # error for Z0 coefficient
    ZqAV_fit_chi2_ndof = []                     # chi^2 / ndof
    for ii, ZZ in enumerate(ZqAV_data):
        underline_print('\nFitting ' + ZqAV_stems[ii])
        ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ))
        ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
        ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(x_axis, ZqAV_mu[ii], ZZ_cov_shrunk, all_fit_forms)
        plot_rcs_fitted(ZqAV_mu[ii], ZqAV_sigma[ii], ZZ_model, ZZ_params, ZZ_fit_covar, ZqAV_labels[ii], ZqAV_ranges[ii], \
                        stem + 'fits/ZqVA/' + ZqAV_stems[ii] + '.pdf')
        ZqAV_fit_vals.append(ZZ_params[0])
        ZqAV_fit_sigmas.append(np.sqrt(ZZ_fit_covar[0, 0]))
        print(ZqAV_stems[ii] + ' value = ' + export_float_latex(ZZ_params[0], np.sqrt(ZZ_fit_covar[0, 0]), sf = 2))
        ZqAV_fit_chi2_ndof.append(ZZ_chi2 / ZZ_dof)
        fout[ZqAV_stems[ii] + '/fit_params'] = ZZ_params
        fout[ZqAV_stems[ii] + '/fit_param_covar'] = ZZ_fit_covar
        fout[ZqAV_stems[ii] + '/chi2'] = ZZ_chi2
        fout[ZqAV_stems[ii] + '/ndof'] = ZZ_dof
        fout[ZqAV_stems[ii] + '/model'] = str(ZZ_model)

# Fit Znm
Z_fit_vals = np.zeros((5, 5), dtype = np.float64)
Z_fit_sigmas = np.zeros((5, 5), dtype = np.float64)
Z_fit_chi2_ndof = np.zeros((5, 5), dtype = np.float64)
for n, m in itertools.product(range(5), repeat = 2):
    underline_print('\nFitting ' + Zlabels[n][m])
    ZZ = Z[n, m]
    ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ))
    ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
    ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(x_axis, Z_mu[n, m], ZZ_cov_shrunk, all_fit_forms)
    plot_rcs_fitted(Z_mu[n, m], Z_sigma[n, m], ZZ_model, ZZ_params, ZZ_fit_covar, \
        '$\mathcal{Z}_{' + str(n + 1) + str(m + 1) + '}/ \mathcal{Z}_V^2$', Z_range[n, m], stem + 'fits/Zops/' + Zlabels[n][m] + '.pdf')
    Z_fit_vals[n, m] = ZZ_params[0]
    Z_fit_sigmas[n, m] = np.sqrt(ZZ_fit_covar[0, 0])
    print(Zlabels[n][m] + ' value = ' + export_float_latex(ZZ_params[0], np.sqrt(ZZ_fit_covar[0, 0]), sf = 2))
    Z_fit_chi2_ndof[n, m] = ZZ_chi2 / ZZ_dof
    fout[Zlabels[n][m] + '/fit_params'] = ZZ_params
    fout[Zlabels[n][m] + '/fit_param_covar'] = ZZ_fit_covar
    fout[Zlabels[n][m] + '/chi2'] = ZZ_chi2
    fout[Zlabels[n][m] + '/ndof'] = ZZ_dof
    fout[Zlabels[n][m] + '/model'] = str(ZZ_model)

print('Scaling errors by sqrt(chi^2 / dof)')
Z_fit_sigmas *= np.sqrt(Z_fit_chi2_ndof)

# # Fit Znm/ZV^2
# ZbyZVsq_fit_vals = np.zeros((5, 5), dtype = np.float64)
# ZbyZVsq_fit_sigmas = np.zeros((5, 5), dtype = np.float64)
# ZbyZVsq_fit_chi2_ndof = np.zeros((5, 5), dtype = np.float64)
# for n, m in itertools.product(range(5), repeat = 2):
#     underline_print('\nFitting ' + Zlabels[n][m] + '/ZV^2')
#     ZZ = ZbyZVsq[n, m]
#     ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ))
#     ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
#     ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(x_axis, ZbyZVsq_mu[n, m], ZZ_cov_shrunk, all_fit_forms)
#     plot_rcs_fitted(ZbyZVsq_mu[n, m], ZbyZVsq_sigma[n, m], ZZ_model, ZZ_params, ZZ_fit_covar, '$\mathcal{Z}_{'+str(n+1)+str(m+1)+'}/\mathcal{Z}_V^2$',\
#                     ZbyZVsq_range[n, m], stem + 'fits/ZopsbyZVsq/' + ZbyZVsqlabels[n][m] + '.pdf')
#     ZbyZVsq_fit_vals[n, m] = ZZ_params[0]
#     ZbyZVsq_fit_sigmas[n, m] = np.sqrt(ZZ_fit_covar[0, 0])
#     ZbyZVsq_fit_chi2_ndof[n, m] = ZZ_chi2 / ZZ_dof
#     fout[ZbyZVsqlabels[n][m] + '/fit_params'] = ZZ_params
#     fout[ZbyZVsqlabels[n][m] + '/fit_param_covar'] = ZZ_fit_covar
#     fout[ZbyZVsqlabels[n][m] + '/chi2'] = ZZ_chi2
#     fout[ZbyZVsqlabels[n][m] + '/ndof'] = ZZ_dof
#     fout[ZbyZVsqlabels[n][m] + '/model'] = str(ZZ_model)

fout.close()

# print results
if RIsMOM:
    print('\n\n')
    underline_print('Results for Zq/ZV, ZV, ZA')
    for ii in range(len(ZqAV_data)):
        print(ZqAV_stems[ii] + ' = ' + export_float_latex(ZqAV_fit_vals[ii], ZqAV_fit_sigmas[ii], sf = 2))
        print('Chi^2 / ndof = ' + str(ZqAV_fit_chi2_ndof[ii]))

print('\n\n')
underline_print('Results for Znm/ZV^2')
Znm_print = np.array([[export_float_latex(Z_fit_vals[n, m], Z_fit_sigmas[n, m], sf = 2) for m in range(5)] for n in range(5)])
print('Znm = ')
print(Znm_print)
print('Chi^2 / ndof = ')
print(Z_fit_chi2_ndof)

# print('\n\n')
# underline_print('Results for Znm / ZV^2')
# ZnmbyZVsq_print = np.array([[export_float_latex(ZbyZVsq_fit_vals[n, m], ZbyZVsq_fit_sigmas[n, m], sf = 2) for m in range(5)] for n in range(5)])
# print('Znm / ZV^2 = ')
# print(ZnmbyZVsq_print)
# print('Chi^2 / ndof = ')
# print(ZbyZVsq_fit_chi2_ndof)
