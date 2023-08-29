################################################################################
# Fits discretization artifacts for nnpp renormalization for ZV and ZA.        #
# Values for ZV, ZA from 1610.04545, 1611.07452:
# ZA = 0.867(43)
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
# subset_idxs = None
# subset_idxs = list(range(6, 12))
# subset_idxs = list(range(4, 10))
# subset_idxs = list(range(10, 14))
# subset_idxs = list(range(11, 14))
# subset_idxs = list(range(3, 14))
subset_idxs = list(range(8, 14))

# finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_gamma.h5'
finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_113400/Z_gamma.h5'
foutpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/fitsZVA.h5'
# stem = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/ZVA/'
stem = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/ZVA/ptilde/'
ZV_range = np.array([0.56, 0.88])
ZA_range = np.array([0.7, 0.95])

F = h5py.File(finpath, 'r')
Lat = Lattice(32, 48)
a = 0.145                                   # fm, placeholder for now
ainv = hbarc / a
mpi = 0.8                                   # MeV, placeholder for now
k_list = F['momenta'][()]
mom_list = np.array([Lat.to_linear_momentum(k, datatype=np.float64) for k in k_list])
mu_list = np.array([get_energy_scale(q, a, Lat) for q in k_list])
print('Energy scales:')
print(mu_list)
apsq_list = np.array([square(k) for k in mom_list])
apsq_tilde_list = np.real(np.array([square(Lat.to_lattice_momentum(k)) for k in mom_list]))

print('apsq:')
print(apsq_list)

print('apsq_tilde:')
print(apsq_tilde_list)

ZV = np.real(F['ZV'][()])
ZA = np.real(F['ZA'][()])
n_momenta, n_boot = ZV.shape[0], ZV.shape[1]

if subset_idxs != None:
    k_list = k_list[subset_idxs]
    mom_list = mom_list[subset_idxs]
    mu_list = mu_list[subset_idxs]
    apsq_list = apsq_list[subset_idxs]
    apsq_tilde_list = apsq_tilde_list[subset_idxs]
    print('Fitting on momentum subset: ' + str(apsq_list))
    ZV = ZV[subset_idxs]
    ZA = ZA[subset_idxs]

# Try everything as a function of \hat{p}^2

ZV_mu = np.mean(ZV, axis = 1)
ZV_sigma = np.std(ZV, axis = 1, ddof = 1)
ZA_mu = np.mean(ZA, axis = 1)
ZA_sigma = np.std(ZA, axis = 1, ddof = 1)

x_axis = apsq_list if apsq else mu_list

# TODO Try using apsq_tilde as the x-axis
# x_axis = apsq_tilde_list

xlabel = '$(ap)^2$' if apsq else '$\\mu\;(\\mathrm{GeV})$'
# xlimits = [0.0, 6.5] if apsq else [1.0, 4.1]
# xlimits = [0.0, 18.0] if apsq else [1.0, 4.1]
# xlimits = [0.0, 24.0] if apsq else [1.0, 4.1]
xlimits = [0.0, 1.0]
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

# generate plot of fitted data
def plot_rcs_fitted(cvs, sigmas, model, params, param_covar, ylabel, ylimits, path, cols = ['r', 'b', 'c'], rng = None):
    """
    Plots data with central values cvs and error sigmas, and the best fit band after subtracting off
    discretization artifacts.
    """
    with sns.plotting_context('paper'):
        # n_plts = cvs.shape[0]    # pass in list of plots
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        # overload to plot multiple

        global x_axis
        if rng:
            this_x_axis = x_axis[rng]
        _, caps, _ = plt.errorbar(this_x_axis, cvs, sigmas, fmt='.', c=cols[0], label='data',
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        params0 = [0.0]
        params0.extend(params[1:])
        subtracted_pts = [cvs[ii] - model.F(params0)(xx)[0] for ii, xx in enumerate(this_x_axis)]
        # TODO make sigmas include errors on fit parameters
        print(subtracted_pts)
        _, caps, _ = plt.errorbar(this_x_axis, subtracted_pts, sigmas, fmt = '.', c = cols[1], label = 'sub',
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


# Try Bayesian fit with ZA

def make_data_boots(corr, domain, lam = 0):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T). Here lam is the shrinkage parameter λ, which is 
    set to 1 (fully correlated) by default. 
    """
    d = {t: corr[t, :] for t in domain}
    df = pd.DataFrame(d)
    cv = np.mean(corr, axis = 0)[domain]
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)      # get covariance
    return domain, gv.gvar(cv, cov)

# fit function
def f(x, p):
   return sum(pn * x ** n for n, pn in enumerate(p))

# 91-parameter prior for the fit
prior = gv.gvar(50 * ['0(1)'])
print(ZA.shape)
_, ZA_gvar = make_data_boots(ZA, list(range(len(x_axis))))
fit = lsqfit.nonlinear_fit(data=(x_axis, ZA_gvar), prior=prior, fcn=f)
print(fit.format(maxline=True))


# # package together ZAV
# ZAV_data = [ZV, ZA]
# ZAV_mu = [ZV_mu, ZA_mu]
# ZAV_sigma = [ZV_sigma, ZA_sigma]
# ZAV_ranges = [ZV_range, ZA_range]
# ZAV_labels = ['$\mathcal{Z}_V$', '$\mathcal{Z}_A$']
# ZAV_stems = ['ZV', 'ZA']

# if plot_raw:
#     for ii in range(len(ZAV_data)):
#         plot_rcs_raw(ZAV_mu[ii], ZAV_sigma[ii], ZAV_labels[ii], ZAV_ranges[ii], stem + 'raw/' + ZAV_stems[ii] + '.pdf')
    
# # Start fitting. Use fit forms here.
# # fout = h5py.File(foutpath, 'w')
# ap2 = Model(lambda params : lambda apsq : params[0] * apsq, 1, '(p2)', 'c1')
# ap4 = Model(lambda params : lambda apsq : params[0] * (apsq ** 2), 1, '(p2^2)', 'c2')
# ap6 = Model(lambda params : lambda apsq : params[0] * (apsq ** 3), 1, '(p2^3)', 'c3')
# ap8 = Model(lambda params : lambda apsq : params[0] * (apsq ** 4), 1, '(p2^4)', 'c4')
# ap10 = Model(lambda params : lambda apsq : params[0] * (apsq ** 5), 1, '(p2^5)', 'c5')
# ap12 = Model(lambda params : lambda apsq : params[0] * (apsq ** 6), 1, '(p2^6)', 'c6')
# apm2 = Model(lambda params : lambda apsq : params[0] / apsq, 1, '(1/p2)', 'c-1')
# logap = Model(lambda params : lambda apsq : params[0] * np.log(apsq), 1, 'log(p2)', 'c6')
# aplogap = Model(lambda params : lambda apsq : params[0] * apsq * np.log(apsq), 1, '(p2*log(p2))', 'c7')
# # all_fit_forms = [ap2, ap4, ap6, ap8, aplogap, apm2]
# # all_fit_forms = [ap2, ap4, ap6, ap8, aplogap, logap]
# # all_fit_forms = [ap2, ap4, ap6, ap8, aplogap]
# # all_fit_forms = [ap2, ap4, ap6, ap8, ap10, apm2]
# # all_fit_forms = [ap2, ap4, ap6, ap8]
# # all_fit_forms = [ap2, ap4, ap6, apm2]
# # all_fit_forms = [ap2, ap4]
# all_fit_forms = [ap2]
# print('Functional forms: Z0, ' + str(all_fit_forms))

# dist_dir = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/RIsMOM/fit_distributions'
# all_subsets = [list(range(n, 6)) for n in range(5)]
# print(all_subsets)
# # loop over subsets: figure out which models to choose at each step. Then look at the distribution of Z0 for each one and save it to dist_dir

# # shrinkage parameter, may want to loop over it.
# lam = 0.5
# ZAV_fit_vals = [[], []]                          # best fit coefficients for Z0
# ZAV_fit_sigmas = [[], []]                        # error for Z0 coefficient
# ZAV_fit_chi2_ndof = [[], []]                     # chi^2 / ndof
# for ii, ZZ in enumerate(ZAV_data):
#     for jj, rng in enumerate(all_subsets):
#         id_stem = ZAV_stems[ii]
#         underline_print('\nFitting ' + id_stem)
#         print(rng)
#         print('Data points: ' + str(x_axis[rng]))
#         ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ[rng]))
#         ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
#         ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(
#             x_axis[rng], ZAV_mu[ii][rng], ZZ_cov_shrunk, all_fit_forms)
#         plot_rcs_fitted(ZAV_mu[ii][rng], ZAV_sigma[ii][rng], ZZ_model, ZZ_params, ZZ_fit_covar, ZAV_labels[ii], ZAV_ranges[ii],
#                         stem + 'fits/' + id_stem + '_' + str(rng[0]) + '_' + str(rng[1]) + '.pdf', rng = rng)
#         ZAV_fit_vals[ii].append(ZZ_params[0])
#         ZAV_fit_sigmas[ii].append(np.sqrt(ZZ_fit_covar[0, 0]))
#         print(id_stem + ' value = ' +
#             export_float_latex(ZZ_params[0], np.sqrt(ZZ_fit_covar[0, 0]), sf=2))
#         ZAV_fit_chi2_ndof[ii].append(ZZ_chi2 / ZZ_dof)
#     # fout[id_stem + '/fit_params'] = ZZ_params
#     # fout[id_stem + '/fit_param_covar'] = ZZ_fit_covar
#     # fout[id_stem + '/chi2'] = ZZ_chi2
#     # fout[id_stem + '/ndof'] = ZZ_dof
#     # fout[id_stem + '/model'] = str(ZZ_model)

# print('Ranges fit at: ')
# print(all_subsets)

# print('Fit values for ZV: ')
# print(ZAV_fit_vals[0])
# print('Chi^2 values for ZV: ')
# print(ZAV_fit_chi2_ndof[0])

# print('Fit values for ZA: ')
# print(ZAV_fit_vals[1])
# print('Chi^2 values for ZA: ')
# print(ZAV_fit_chi2_ndof[1])

# # Fit ZV, ZA
# # lam = 0.5                                   # shrinkage parameter, may want to loop over it.
# # ZAV_fit_vals = []                          # best fit coefficients for Z0
# # ZAV_fit_sigmas = []                        # error for Z0 coefficient
# # ZAV_fit_chi2_ndof = []                     # chi^2 / ndof
# # for ii, ZZ in enumerate(ZAV_data):
# #     id_stem = ZAV_stems[ii]
# #     underline_print('\nFitting ' + id_stem)
# #     ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ))
# #     ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
# #     ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(x_axis, ZAV_mu[ii], ZZ_cov_shrunk, all_fit_forms)
# #     plot_rcs_fitted(ZAV_mu[ii], ZAV_sigma[ii], ZZ_model, ZZ_params, ZZ_fit_covar, ZAV_labels[ii], ZAV_ranges[ii], \
# #                     stem + 'fits/' + id_stem + '.pdf')
# #     ZAV_fit_vals.append(ZZ_params[0])
# #     ZAV_fit_sigmas.append(np.sqrt(ZZ_fit_covar[0, 0]))
# #     print(id_stem + ' value = ' + export_float_latex(ZZ_params[0], np.sqrt(ZZ_fit_covar[0, 0]), sf = 2))
# #     ZAV_fit_chi2_ndof.append(ZZ_chi2 / ZZ_dof)
# #     fout[id_stem + '/fit_params'] = ZZ_params
# #     fout[id_stem + '/fit_param_covar'] = ZZ_fit_covar
# #     fout[id_stem + '/chi2'] = ZZ_chi2
# #     fout[id_stem + '/ndof'] = ZZ_dof
# #     fout[id_stem + '/model'] = str(ZZ_model)

# # fout.close()

# # print results
# # print('\n\n')
# # underline_print('Results for ZV, ZA')
# # for ii in range(len(ZAV_data)):
# #     print(ZAV_stems[ii] + ' = ' + export_float_latex(ZAV_fit_vals[ii], ZAV_fit_sigmas[ii], sf = 2))
# #     print('Chi^2 / ndof = ' + str(ZAV_fit_chi2_ndof[ii]))
