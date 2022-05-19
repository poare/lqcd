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

# inputs to toggle
apsq = True                                             # x axis to plot against. If apsq, plots against (ap)^2, else plots against mu^2.
RIsMOM = True                                           # True to fit RI/sMOM data, False to fit MSbar data.

if RIsMOM:
    finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_gamma.h5'
    foutpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/fit_dists_RI.h5'
    stem = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/RIsMOM/fit_distributions'
    Zq_range = np.array([0.66, 0.86])                   # Ranges for plotting
    ZV_range = np.array([0.56, 0.82])
    ZA_range = np.array([0.7, 0.87])
    ZqbyZV_range = np.array([1.05, 1.16])
    Z_range = np.array([
        [[0.42, 0.70], [-0.12, 0.0], [-0.015, 0.0], [0.0, 0.05], [0.005, 0.022]],
        [[-0.10, 0.0], [0.4, 0.70], [0.06, 0.16], [-0.10, 0.0], [0.0, 0.008]],
        [[0.0, 0.015], [0.0, 0.02], [0.2, 0.7], [0.0, 0.16], [-0.005, 0.0]],
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
else:
    finpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_gamma.h5'
    foutpath = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/fit_dists_MS.h5'
    stem = '/Users/theoares/Dropbox (MIT)/research/0nubb/nnpp/plots/MSbar/fit_distributions'
    Zq_range = np.array([0.69, 0.86])                   # Ranges for plotting
    ZV_range = np.array([0.6, 0.82])
    ZA_range = np.array([0.7, 0.87])
    ZqbyZV_range = np.array([1.05, 1.16])
    Z_range = np.array([
        [[0.45, 0.70], [-0.12, 0.0], [-0.012, 0.0], [0.0, 0.05], [0.005, 0.022]],
        [[-0.10, 0.0], [0.45, 0.70], [0.06, 0.14], [-0.10, 0.0], [0.0, 0.006]],
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
ZbyZVsq = np.einsum('ijqb,qb->ijqb', Z, 1 / (ZV ** 2))
ZqbyZV = Zq / ZV

Zq_mu = np.mean(Zq, axis = 1)
Zq_sigma = np.std(Zq, axis = 1, ddof = 1)
ZV_mu = np.mean(ZV, axis = 1)
ZV_sigma = np.std(ZV, axis = 1, ddof = 1)
ZA_mu = np.mean(ZA, axis = 1)
ZA_sigma = np.std(ZA, axis = 1, ddof = 1)
ZqbyZV_mu = np.mean(ZqbyZV, axis = 1)
ZqbyZV_sigma = np.std(ZqbyZV, axis = 1, ddof = 1)
Z_mu = np.mean(Z, axis = 3)
Z_sigma = np.std(Z, axis = 3, ddof = 1)
ZbyZVsq_mu = np.mean(ZbyZVsq, axis = 3)
ZbyZVsq_sigma = np.std(ZbyZVsq, axis = 3, ddof = 1)

x_axis = apsq_list if apsq else mu_list
xlabel = '$(ap)^2$' if apsq else '$\\mu\;(\\mathrm{GeV})$'
xlimits = [0.0, 6.5] if apsq else [1.0, 4.1]
asp_ratio = 4/3

# generate plot of fitted data
def plot_rcs_fitted(cvs, sigmas, model, params, param_covar, ylabel, ylimits, path, cols = ['r', 'b', 'c'], x_pts = x_axis):
    """
    Plots data with central values cvs and error sigmas, and the best fit band after subtracting off
    discretization artifacts.
    """
    with sns.plotting_context('paper'):
        # n_plts = cvs.shape[0]    # pass in list of plots
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        # overload to plot multiple
        _, caps, _ = plt.errorbar(x_pts, cvs, sigmas, fmt = '.', c = cols[0], label = 'data', \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        params0 = [0.0]
        params0.extend(params[1:])
        subtracted_pts = [cvs[ii] - model.F(params0)(xx)[0] for ii, xx in enumerate(x_pts)]
        # TODO make sigmas include errors on fit parameters
        print(subtracted_pts)
        _, caps, _ = plt.errorbar(x_pts, subtracted_pts, sigmas, fmt = '.', c = cols[1], label = 'sub', \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        # plt.axhline(y = params[0], color = cols[2], linestyle = '-', label = '$Z_0$')
        x_band = np.linspace(xlimits[0], xlimits[1])
        x_band_reg = np.linspace(0.1, xlimits[1])
        extrap_fit = np.array([model.F(params)(xx) for xx in x_band_reg])
        plt.plot(x_band_reg, extrap_fit, c = 'g', label = 'extrap', alpha = 0.5)
        # print(np.sqrt(param_covar[0, 0]))
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

def plot_dist(Z0_dist, Z0_sigma, xtick_labels, ylabel, path):
    """
    Plots distribution of fit parameters for a variety of fit ranges.
    """
    with sns.plotting_context('paper'):
        fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
        plt.figure(figsize = fig_size)
        xx = list(range(1, len(Z0_dist) + 1))
        _, caps, _ = plt.errorbar(xx, Z0_dist, Z0_sigma, fmt = '.', c = 'r', \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('Fit range', fontsize = style['fontsize'])
        plt.ylabel('Z0', fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(ticks = xx, labels = xtick_labels, fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Distribution plot ' + ylabel + ' saved at: \n   ' + path)

def plot_dist_chi2(Z0_dist, Z0_sigma, chi2_ndof, xtick_labels, ylabel, path, coloring):
    """
    Plots distribution of fit parameters for a variety of fit ranges, and the chi^2 / ndof.
    Generates a double-panel figure where the x-axis is the fit index; the top panel shows
    the Z0 value and error, and the bottom panel shows the chi^2 / ndof.
    """
    if coloring == None:
        coloring = [[len(Z0_dist), 'r']]
    with sns.plotting_context('paper'):
        from matplotlib import gridspec
        from matplotlib import rc
        rc('text', usetex = True)
        fig_size = (style['colwidth'], 2 * style['colwidth'] / asp_ratio)
        fig = plt.figure(figsize = fig_size)
        gs = gridspec.GridSpec(2, 1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        xx = list(range(1, len(Z0_dist) + 1))
        last_col_idx = 0
        for col_pair in coloring:
            _, caps, _ = ax0.errorbar(xx[last_col_idx : last_col_idx + col_pair[0]], Z0_dist[last_col_idx : last_col_idx + col_pair[0]], \
                    Z0_sigma[last_col_idx : last_col_idx + col_pair[0]], fmt = '.', c = col_pair[1], capsize = style['endcaps'], markersize = style['markersize'], \
                    elinewidth = style['ebar_width'])
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
            ax1.scatter(xx[last_col_idx : last_col_idx + col_pair[0]], chi2_ndof[last_col_idx : last_col_idx + col_pair[0]], c = col_pair[1], marker = '.', \
                    s = style['markersize'])
            last_col_idx += col_pair[0]
        for spine in spinedirs:
            ax0.spines[spine].set_linewidth(style['axeswidth'])
            ax1.spines[spine].set_linewidth(style['axeswidth'])
        ax0.xaxis.set_tick_params(width = style['tickwidth'], length = 2 * style['ticklength'], bottom=True, top=True, \
                                    labelbottom=True, labeltop=False, direction = 'inout')
        ax1.xaxis.set_tick_params(width = style['tickwidth'], length = 2 * style['ticklength'], bottom=True, top=False, \
                                    labelbottom=True, labeltop=False, direction = 'inout')
        ax1.set_xlabel('Fit index', fontsize = style['fontsize'])
        ax0.set_ylabel(ylabel, fontsize = style['fontsize'])
        ax1.set_ylabel('$\chi^2$/ndof', fontsize = style['fontsize'])
        plt.setp(ax0.get_xticklabels(), visible = False)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        print('Distribution plot ' + ylabel + ' saved at: \n   ' + path)

# package together ZqAV
# ZqAV_data = [Zq, ZV, ZA, ZqbyZV]
ZqAV_data = [Zq, ZV, ZA]
ZqAV_mu = [Zq_mu, ZV_mu, ZA_mu, ZqbyZV_mu]
ZqAV_sigma = [Zq_sigma, ZV_sigma, ZA_sigma, ZqbyZV_sigma]
ZqAV_ranges = [Zq_range, ZV_range, ZA_range, ZqbyZV_range]
ZqAV_labels = ['$\mathcal{Z}_q^\mathrm{RI}$', '$\mathcal{Z}_V$', '$\mathcal{Z}_A$', '$\mathcal{Z}_q / \mathcal{Z}_V$']
ZqAV_stems = ['Zq', 'ZV', 'ZA', 'ZqbyZV']
Zlabels = [['Z' + str(n + 1) + str(m + 1) for m in range(5)] for n in range(5)]
ZbyZVsqlabels = [['Z' + str(n + 1) + str(m + 1) + 'byZVsq' for m in range(5)] for n in range(5)]

# Start fitting. Use fit forms here.
fout = h5py.File(foutpath, 'w')
ap2 = Model(lambda params : lambda apsq : params[0] * apsq, 1, '(x)', 'c1')
ap4 = Model(lambda params : lambda apsq : params[0] * (apsq ** 2), 1, '(x^2)', 'c2')
ap6 = Model(lambda params : lambda apsq : params[0] * (apsq ** 3), 1, '(x^3)', 'c3')
ap8 = Model(lambda params : lambda apsq : params[0] * (apsq ** 4), 1, '(x^4)', 'c4')
apm2 = Model(lambda params : lambda apsq : params[0] / apsq, 1, '(1/x)', 'c5')

# loop over subsets: figure out which models to choose at each step. Then look at the distribution of Z0 for each one and save it to dist_dir
all_subsets = [list(range(n, 8)) for n in range(2, 6)]
subs_labels = ['p' + str(subs[0]) + 'to' + str(subs[-1]) for subs in all_subsets]
subset_colors = ['r', 'g', 'b', 'c']
# all_forms = [
#     [ap2, ap4, ap6, ap8],
#     [ap2, ap4, ap6, ap8],
#     [ap2, ap4, ap6],
#     [ap2, ap4]
# ]
all_forms = [
    [ap2, ap4, ap6, apm2],
    [ap2, ap4, ap6, apm2],
    [ap2, ap4, apm2],
    [ap2, ap4, apm2]
]
k_list_subsets = [k_list[subset] for subset in all_subsets]
mom_list_subsets = [mom_list[subset] for subset in all_subsets]
if apsq:
    x_axis_subsets = [apsq_list[subset] for subset in all_subsets]
else:
    x_axis_subsets = [mu_list[subset] for subset in all_subsets]

# Fit Zq, ZV, ZA, Zq / ZV
lam = 0.0                                   # shrinkage parameter, may want to loop over it.
ZqAV_fit_vals = [[], [], []]                    # [Zq..., ZV..., ZA...]
ZqAV_fit_sigmas = [[], [], []]                    # [Zq..., ZV..., ZA...]
ZqAV_fit_chi2_ndof = [[], [], []]                    # [Zq..., ZV..., ZA...]
for ii, ZZ in enumerate(ZqAV_data):
    underline_print('\n\nFitting ' + ZqAV_stems[ii])
    pt_colors = []
    for subset_idx, subset in enumerate(all_subsets):
        underline_print('Fitting momentum subset ' + str(subset))
        subs_stem = subs_labels[subset_idx]
        ZZ_sub = ZZ[subset]
        ZZ_mu = ZqAV_mu[ii][subset]
        ZZ_sigma = ZqAV_sigma[ii][subset]
        ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ_sub))
        ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
        # ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(x_axis_subsets[subset_idx], ZZ_mu, ZZ_cov_shrunk, all_forms[subset_idx])
        # ZZ_chi2_ndof = ZZ_chi2 / ZZ_dof
        # ZZ_fit_covar *= ZZ_chi2_ndof                        # scale fit covariance by chi2 / ndof
        # plot_rcs_fitted(ZZ_mu, ZZ_sigma, ZZ_model, ZZ_params, ZZ_fit_covar, ZqAV_labels[ii], ZqAV_ranges[ii], \
        #                 stem + '/' + ZqAV_stems[ii] + '/' + subs_stem + '.pdf', x_pts = x_axis_subsets[subset_idx])
        # ZqAV_fit_vals[ii].append(ZZ_params[0])
        # ZqAV_fit_sigmas[ii].append(np.sqrt(ZZ_fit_covar[0, 0]))
        # ZqAV_fit_chi2_ndof[ii].append(ZZ_chi2 / ZZ_dof)
        # fout[ZqAV_stems[ii] + subs_stem + '/fit_params'] = ZZ_params
        # fout[ZqAV_stems[ii] + subs_stem + '/fit_param_covar'] = ZZ_fit_covar
        # fout[ZqAV_stems[ii] + subs_stem + '/chi2'] = ZZ_chi2
        # fout[ZqAV_stems[ii] + subs_stem + '/ndof'] = ZZ_dof
        # fout[ZqAV_stems[ii] + subs_stem + '/model'] = str(ZZ_model)
        ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_all(x_axis_subsets[subset_idx], ZZ_mu, ZZ_cov_shrunk, all_forms[subset_idx])
        ZqAV_fit_vals[ii].extend([ZZ_params[k][0] for k in range(len(ZZ_params))])
        ZqAV_fit_sigmas[ii].extend([np.sqrt(ZZ_fit_covar[k][0, 0]) for k in range(len(ZZ_fit_covar))])
        ZqAV_fit_chi2_ndof[ii].extend(list(ZZ_chi2 / ZZ_dof))
        pt_colors.append(len(ZZ_params))
    print('Fit chi^2 / ndofs are: ' + str(ZqAV_fit_chi2_ndof[ii]))
    # plot_dist(ZqAV_fit_vals[ii], ZqAV_fit_sigmas[ii], subs_labels, ZqAV_labels[ii], stem + '/' + ZqAV_stems[ii] + '/dist.pdf')
    plot_dist_chi2(ZqAV_fit_vals[ii], ZqAV_fit_sigmas[ii], ZqAV_fit_chi2_ndof[ii], subs_labels, ZqAV_labels[ii], stem + '/' + ZqAV_stems[ii] + '/dist.pdf', \
                    coloring = zip(pt_colors, subset_colors))


# # Fit Znm
# Z_fit_vals = np.zeros((5, 5), dtype = np.float64)
# Z_fit_sigmas = np.zeros((5, 5), dtype = np.float64)
# Z_fit_chi2_ndof = np.zeros((5, 5), dtype = np.float64)
# for n, m in itertools.product(range(5), repeat = 2):
#     underline_print('\nFitting ' + Zlabels[n][m])
#     ZZ = Z[n, m]
#     ZZ_cov = get_covariance(np.einsum('kb->bk', ZZ))
#     ZZ_cov_shrunk = shrinkage(ZZ_cov, lam)
#     ZZ_params, ZZ_chi2, ZZ_dof, ZZ_fit_covar, ZZ_model = process_fit_forms_AIC_all(x_axis, Z_mu[n, m], ZZ_cov_shrunk, all_fit_forms)
#     plot_rcs_fitted(Z_mu[n, m], Z_sigma[n, m], ZZ_model, ZZ_params, ZZ_fit_covar, '$\mathcal{Z}_{' + str(n + 1) + str(m + 1) + '}$', Z_range[n, m], \
#                     stem + 'fits/Zops/' + Zlabels[n][m] + '.pdf')
#     Z_fit_vals[n, m] = ZZ_params[0]
#     Z_fit_sigmas[n, m] = np.sqrt(ZZ_fit_covar[0, 0])
#     Z_fit_chi2_ndof[n, m] = ZZ_chi2 / ZZ_dof
#     fout[Zlabels[n][m] + '/fit_params'] = ZZ_params
#     fout[Zlabels[n][m] + '/fit_param_covar'] = ZZ_fit_covar
#     fout[Zlabels[n][m] + '/chi2'] = ZZ_chi2
#     fout[Zlabels[n][m] + '/ndof'] = ZZ_dof
#     fout[Zlabels[n][m] + '/model'] = str(ZZ_model)

fout.close()

# # print results
# print('\n\n')
# underline_print('Results for Zq, ZV, ZA')
# for ii in range(len(ZqAV_data)):
#     print(ZqAV_stems[ii] + ' = ' + export_float_latex(ZqAV_fit_vals[ii], ZqAV_fit_sigmas[ii], sf = 2))
#     print('Chi^2 / ndof = ' + str(ZqAV_fit_chi2_ndof[ii]))
#
# print('\n\n')
# underline_print('Results for Znm')
# Znm_print = np.array([[export_float_latex(Z_fit_vals[n, m], Z_fit_sigmas[n, m], sf = 2) for m in range(5)] for n in range(5)])
# print('Znm = ')
# print(Znm_print)
# print('Chi^2 / ndof = ')
# print(Z_fit_chi2_ndof)
