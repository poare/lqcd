from __future__ import print_function

######################################################################################
# This script fits the excited states for the effective matrix element data          #
# #$O_k^{eff}(t)$. The script takes as input the ensemble index {0 : 24I/ml0p01,     #
# 1 : 24I/ml0p005, 2 : 32I/ml0p008, 3 : 32I/ml0p006, 4 : 32I/ml0p004}, and the fit   #
# model index {0 : 'f3', 1 : 'f4', 2 : 'f5', 3 : 'f6'}. It plots the resulting fit,  #
# a comparison to the constant fit result, and the stability plot for the fit model. #
# To call the script, run:                                                           #
# > python3 exc_state_fits.py ens_idx fit_idx                                        #
# or run the corresponding shell script scripts/exc_state_fits.sh to run all         #
# ensembles and fit models.                                                          # 
#                                                                                    #
# With fit form 'f6', this is the production code for the paper.                     #
######################################################################################

# use CMU Serif
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['text.usetex'] = True

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D

import numpy as np
import seaborn as sns
import pandas as pd
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
# style = styles['prd_twocol']
style = styles['notebook']

import gvar as gv
import lsqfit
import corrfitter as cf

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

######################################################################################
################################### Set parameters ###################################
######################################################################################

# Set the ensemble index and fit form.
ens_idx = int(sys.argv[1])
ff_idx = int(sys.argv[2])

ensemble = ['24I/ml_0p01', '24I/ml_0p005',
            '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
ens_path = ['24I/ml0p01', '24I/ml0p005', '32I/ml0p008',
            '32I/ml0p006', '32I/ml0p004'][ens_idx]
fitform = ['f3', 'f4', 'f5', 'f6'][ff_idx]
f_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/bare_matrix_elements/' + \
    ensemble + '/fit_params.h5'
n_ops = 5

print('Fitting ensemble ' + str(ensemble) + ' with fit form ' + fitform)

# read in input to plot
f = h5py.File(f_path, 'r')
data_slice = f['data_slice'][()]
c = f['c'][()]
sigmac = f['sigmac'][()]
plot_domain = f['plot_domain'][()]
f.close()

data_plot_mu = np.mean(data_slice, axis=0)
data_plot_sigma = np.std(data_slice, axis=0, ddof=1)

# ranges to fit
if fitform == 'f3' or fitform == 'f5' or fitform == 'f6':
    exc_domain = [
        # [np.arange(5, 33), np.arange(5, 33), np.arange(8, 33), np.arange(5, 33), np.arange(5, 33)],
        [np.arange(6, 33), np.arange(6, 33), np.arange(9, 33), np.arange(6, 33), np.arange(6, 33)], 
        [np.arange(6, 33), np.arange(6, 33), np.arange(9, 33), np.arange(6, 33), np.arange(6, 33)], 
        [np.arange(6, 33), np.arange(6, 33), np.arange(9, 33), np.arange(6, 33), np.arange(6, 33)], 
        [np.arange(6, 33), np.arange(6, 33), np.arange(9, 33), np.arange(6, 33), np.arange(6, 33)], 
#         [np.arange(6, 33), np.arange(6, 33), np.arange(8, 33), np.arange(6, 33), np.arange(6, 33)]
        [np.arange(6, 33), np.arange(6, 33), np.arange(9, 33), np.arange(6, 33), np.arange(6, 33)]
    ][ens_idx]
else:    # then we have f4
    exc_domain = [
        [np.arange(8, 31), np.arange(8, 31), np.arange(11, 30), np.arange(8, 31), np.arange(8, 31)], 
        [np.arange(8, 31), np.arange(8, 31), np.arange(8, 33), np.arange(8, 31), np.arange(8, 31)], 
        [np.arange(8, 31), np.arange(8, 31), np.arange(8, 33), np.arange(8, 31), np.arange(8, 31)], 
        [np.arange(8, 31), np.arange(8, 31), np.arange(8, 33), np.arange(8, 31), np.arange(8, 31)], 
        [np.arange(8, 31), np.arange(8, 31), np.arange(8, 33), np.arange(8, 31), np.arange(8, 31)], 
        [np.arange(8, 31), np.arange(8, 31), np.arange(8, 33), np.arange(8, 31), np.arange(8, 31)], 
    ][ens_idx]

# set specific ranges and labels
ytick_labels = [
    [['-5.8', '-5.4', '-5.0'], ['-1.0', '-0.9', '-0.8'], ['3.4', '3.6', '3.8'],
        ['-1.9', '-1.7', '-1.5'], ['2.1', '2.3', '2.5']],         # 24I/ml0p01
    [['-5.0', '-4.7', '-4.4'], ['-8.5', '-8.0', '-7.5'], ['1.70', '1.85', '2.00'],
        ['-1.55', '-1.45', '-1.35'], ['1.80', '1.95', '2.10']],      # 24I/ml0p005
    [['-1.90', '-1.75', '-1.60'], ['-2.9', '-2.7', '-2.5'], ['8.00', '8.75', '9.50'],
        ['-5.8', '-5.3', '-4.8'], ['6.2', '6.7', '7.2']],      # 32I/ml0p008
    [['-1.70', '-1.55', '-1.40'], ['-2.7', '-2.5', '-2.3'], ['5.7', '6.2', '6.7'],
        ['-5.5', '-5.0', '-4.5'], ['5.8', '6.2', '6.6']],      # 32I/ml0p006
    [['-1.55', '-1.45', '-1.35'], ['-2.40', '-2.25', '-2.10'], ['3.4', '3.8',
                                                                '4.2'], ['-4.9', '-4.5', '-4.1'], ['5.2', '5.6', '6.0']],      # 32I/ml0p004
][ens_idx]
pwr = [
    [-3, -2, -4, -2, -3],
    [-3, -3, -4, -2, -3],
    [-3, -3, -5, -3, -4],
    [-3, -3, -5, -3, -4],
    [-3, -3, -5, -3, -4]
][ens_idx]              # think about doing using \\text{-}
yrangep = [
    [[-0.0060, -0.0048], [-0.0105, -0.0075], [0.00033, 0.00039], [-0.02, -0.014], [0.0020, 0.0026]],
    [[-0.0051, -0.0043], [-0.00875, -0.00725], [0.000165, 0.000205], [-0.016, -0.013], [0.00175, 0.00215]],
    [[-0.002, -0.0015], [-0.0030, -0.0024], [0.0000775, 0.0000975], [-0.0061, -0.0045], [0.00059, 0.00075]],
    [[-0.00175, -0.00135], [-0.0028, -0.0022], [0.000054, 0.00007], [-0.00575, -0.00425], [0.00056, 0.00068]],
    [[-0.0016, -0.0013], [-0.00245, -0.00205], [0.000032, 0.000044], [-0.005, -0.004], [0.0005, 0.00062]]
][ens_idx]
ytick_locs = [[round(float(ytick_labels[k][i]) * (10 ** pwr[k]), np.abs(pwr[k]) + 2)
               for i in range(len(ytick_labels[k]))] for k in range(n_ops)]
# yaxis_labels = ['$' + latex_labels[ii] + '^{\\mathrm{eff}} \\hspace\{-1.0mm\} \\times \\hspace{-0.5mm} 10^{' + str(pwr[ii]) + '}$' \
#                   for ii in range(len(latex_labels))]

latex_labels = [r'\mathcal{O}_1', r'\mathcal{O}_2', r'\mathcal{O}_3', r'\mathcal{O}_{1^\prime}', r'\mathcal{O}_{2^\prime}']
yaxis_labels = [r'$a^4\langle\pi^+ | ' + latex_labels[ii] + r'|\pi^-\rangle \hspace{-1.0mm} \times \hspace{-0.5mm} 10^{' + str(
    pwr[ii]) + r'}$' for ii in range(len(latex_labels))]

######################################################################################
################################# Utility functions ##################################
######################################################################################

asp_ratio = 4/3
def plot_data(cvs, stds, yaxis_label, ylims = None, yt_locs = None, yt_labels = None, mrk = '.', col = 'r'):
    fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
    dom = plot_domain[:len(cvs)]
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        _, caps, _ = plt.errorbar(dom, cvs, yerr = stds, fmt = mrk, c = col, \
                     capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('$t / a$', fontsize = style['fontsize'])
        plt.ylabel(yaxis_label, fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        if yt_locs:
            ax.set_yticks(yt_locs)
            ax.set_yticklabels(yt_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        #plt.xlim(0, max(list(plot_domain)) // 2 + 1.5)
        if ylims:
            plt.ylim(ylims[0], ylims[1])
        plt.tight_layout()
        plt.close()

def plot_fns(dom, fns, yaxis_label, ylims = None, yt_locs = None, yt_labels = None, mrk = '.', cols = None):
    """Plots data and a function on the same figure to get an idea of what the fit parameters should tune to."""
    fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        for ii, fn in enumerate(fns):
            col = cols[ii] if cols else 'b'
            plt.plot(dom, [fn(t) for t in dom], c = col)
        plt.xlabel('$t / a$', fontsize = style['fontsize'])
        plt.ylabel(yaxis_label, fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        if yt_locs:
            ax.set_yticks(yt_locs)
            ax.set_yticklabels(yt_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        #plt.xlim(0, max(list(plot_domain)) // 2 + 1.5)
        if ylims:
            plt.ylim(ylims[0], ylims[1])
        plt.tight_layout()
        plt.close()

def plot_data_fn(cvs, stds, fn, yaxis_label, ylims = None, yt_locs = None, yt_labels = None, mrk = '.', col = 'r'):
    """Plots data and a function on the same figure to get an idea of what the fit parameters should tune to."""
    fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
    dom = plot_domain[:len(cvs)]
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        _, caps, _ = plt.errorbar(dom, cvs, yerr = stds, fmt = mrk, c = col, \
                     capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.plot(dom, [fn(t) for t in dom], c = 'b')
        plt.xlabel('$t / a$', fontsize = style['fontsize'])
        plt.ylabel(yaxis_label, fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        if yt_locs:
            ax.set_yticks(yt_locs)
            ax.set_yticklabels(yt_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        #plt.xlim(0, max(list(plot_domain)) // 2 + 1.5)
        if ylims:
            plt.ylim(ylims[0], ylims[1])
        plt.tight_layout()
        plt.close()

def plot_fit(cvs, stds, fit_x, fit_lower, fit_upper, yaxis_label, ylims = None, yt_locs = None, yt_labels = None, \
             mrk = '.', col = 'r', Oeff_mu = None, Oeff_sigma = None, const_mu = None, const_sigma = None, \
             chi2_dof = None, saveat = None):
    """
    cvs: Central value of data points.
    stds: Standard deviation of data points.
    fit_x: x-domain to plot the fit band against (domain of the fit)
    fit_lower / fit_upper: Lower / upper band of the fit.
    ...
    Oeff_mu / Oeff_std: Mean / std value of the matrix element, extracted as fit.p['c0'].mean / .std
    const_mu / const_sigma: Mean / std value of the matrix element from a constant fit.
    """
    fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
    dom = plot_domain[:len(cvs)]    # in case plot_domain needs truncation
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        if const_mu is not None:
            plt.fill_between(dom, const_mu - const_sigma, const_mu + const_sigma, color = 'silver', alpha = 0.5, \
                             linewidth = 0.0, label = 'const $O^{\mathrm{eff}}$')
        if Oeff_mu is not None:
            plt.fill_between(fit_x, Oeff_mu - Oeff_sigma, Oeff_mu + Oeff_sigma, color = 'k', alpha = 0.5, \
                             linewidth = 0.0, label = 'exc $O^{\mathrm{eff}}$')
        _, caps, _ = plt.errorbar(dom, cvs, yerr = stds, fmt = mrk, c = col, \
                     capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        #plt.fill_between(fit_x, fit_lower, fit_upper, color = col, alpha = 0.3, linewidth = 0.0)
        plt.fill_between(fit_x, fit_lower, fit_upper, color = col, alpha = 0.5, linewidth = 0.0, label = 'extrap')
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        if chi2_dof:
            plt.title('$\chi^2/\mathrm{dof} = $' + str(round(chi2_dof, 2)), fontsize = style['fontsize'])
        plt.xlabel('$t / a$', fontsize = style['fontsize'])
        plt.ylabel(yaxis_label, fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        if yt_locs:
            ax.set_yticks(yt_locs)
            ax.set_yticklabels(yt_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        plt.xlim(0, max(list(plot_domain)) // 2 + 1.5)
        if ylims:
            plt.ylim(ylims[0], ylims[1])
        if const_mu is not None:
            plt.legend()
        plt.tight_layout()
        if saveat:
            plt.savefig(saveat, bbox_inches='tight')
        plt.close()

def plot_fit_paper(k, cvs, stds, fit_x, fit_lower, fit_upper, yaxis_label, ylims = None, yt_locs = None, yt_labels = None, \
             mrk = '.', col = 'r', Oeff_mu = None, Oeff_sigma = None, saveat = None):
    fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
    dom = plot_domain[:len(cvs)]    # in case plot_domain needs truncation
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        if Oeff_mu is not None:
            plt.fill_between(dom, Oeff_mu - Oeff_sigma, Oeff_mu + Oeff_sigma, color = 'k', alpha = 0.2, \
                             linewidth = 0.0, label = r'$a^4\langle\pi^+ | \mathcal{O}_{' + idx_labels[k] + \
                             r'} |\pi^- \rangle$')
        _, caps, _ = plt.errorbar(dom, cvs, yerr = stds, fmt = mrk, c = col, label = 'Data', \
                     capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
        plt.fill_between(fit_x, fit_lower, fit_upper, color = col, alpha = 0.4, linewidth = 0.0, label = 'Extrapolation')
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('$t / a$', fontsize = style['fontsize'])
        plt.ylabel(yaxis_label, fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        if yt_locs:
            ax.set_yticks(yt_locs)
            ax.set_yticklabels(yt_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        plt.xlim(0, max(list(plot_domain)) // 2 + 1.5)
        if ylims:
            plt.ylim(ylims[0], ylims[1])
        plt.legend()
        plt.tight_layout()
        if saveat:
            plt.savefig(saveat, bbox_inches='tight')
        plt.close()

def const_fit_band(cv, std, xlims = (0, max(list(plot_domain)) // 2 + 1.5)):
    xx = np.linspace(xlims[0], xlims[1], 500)
    lower_band = np.full(xx.shape, cv - std)
    upper_band = np.full(xx.shape, cv + std)
    return xx, lower_band, upper_band

def get_fit_band(params, fcn, xlims):
    xx = np.linspace(xlims[0], xlims[1], 500)
    fx = fcn(xx, params)
    fx_lower = gv.mean(fx) - gv.sdev(fx)
    fx_upper = gv.mean(fx) + gv.sdev(fx)
    return xx, fx_lower, fx_upper

def perform_fit(k, tmin, tmax, lam, c_priors, dE_prior):
    domain = np.arange(tmin, tmax)
    fit_data = data_slice[:, k, :]
    t_dom, Oeff = make_data(fit_data, domain, lam)
    prior = make_prior(c_priors, dE_prior)
    fit = lsqfit.nonlinear_fit(data = (t_dom, Oeff), fcn = fcn, prior = prior, p0 = p0)
    return fit

stab_formats = ['.', 'x', '+', 'v']
stab_colors = ['r', 'b', 'g', 'c']
def plot_stability(k, fit_dict, key_labels = ['tmin', 'tmax', 'lambda', 'prior'], band_cv = None, band_std = None, saveat = None):
    n_keys = len(key_labels)
    legend_elems = [Line2D([0], [0], color = stab_colors[ii], lw = 4, label = key_labels[ii]) for ii in range(n_keys)]
    asp_ratio = 2.0
    fig_size = (style['colwidth'] * asp_ratio, style['colwidth'])
    Tmin_rg = tmin_rg_O3 if k == 2 else tmin_rg
    start, doms = 0, []
    for key in key_labels:
        dint = len(fit_dict[k][key])
        doms.append(np.arange(start, start + dint))
        start += dint
    with sns.plotting_context('paper'):
        all_labels = []
        all_doms = np.array([])
        plt.figure(figsize = fig_size)
        for ii, param_key in enumerate(key_labels):
            c0_dat = fit_data[k][param_key]
            if len(c0_dat) == 0:        # if this variation has no accepted fits
                continue
            labels = c0_dat.keys()
            cvs = np.array([c0_dat[jj].p['c0'].mean for jj in labels]) * (10 ** (-pwr[k]))
            stds = np.array([c0_dat[jj].p['c0'].sdev for jj in labels]) * (10 ** (-pwr[k]))
            _, caps, _ = plt.errorbar(doms[ii], cvs, yerr = stds, fmt = stab_formats[ii], c = stab_colors[ii], capsize = style['endcaps'], \
                         markersize = style['markersize'], elinewidth = style['ebar_width'])
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
            plt.ylabel(yaxis_labels[k], fontsize = style['fontsize'])
            all_doms = np.append(all_doms, doms[ii])
            all_labels.extend(labels)
        xlims = (-1, all_doms[-1] + 1)
        if band_cv:
            band_cv *= (10 ** (-pwr[k]))
            band_std *= (10 ** (-pwr[k]))
            plt.fill_between(xlims, band_cv + band_std, band_cv - band_std, color = 'b', alpha = 0.15, linewidth = 0.0, label = 'Fiducial')
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.set_xticks(all_doms)
        ax.set_xticklabels(all_labels)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        ax.legend(handles = legend_elems, loc = 'upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 16})
        ax.set_xlim(xlims)
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        # plt.title('Fit stability for $' + latex_labels[k] + '$', fontsize = 16)
        plt.tight_layout()
        if saveat:
            plt.savefig(saveat, bbox_inches='tight')
        plt.close()

######################################################################################
#################################### Process data ####################################
######################################################################################

# Use this code block if I make this notebook into production code
f3pt_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/analysis_output/' + \
    ensemble + '/SD_output.h5'

f = h5py.File(f3pt_path, 'r')
L, T = f['L'][()], f['T'][()]
vol = (L**3)  # * T
C2pt_tavg = f['pion-00WW'][()]
# C2WW = f['C2pt'][()]
C2_pion00WP = np.real(f['pion-00WP'][()]) / vol
C3pt_tavg = f['C3pt'][()]
Cnpt = f['Cnpt'][()]
R_boot = f['R'][()]
mpi_boot = f['mpi'][()]
f.close()

mpi_mu = np.mean(mpi_boot)
mpi_std = np.std(mpi_boot, ddof=1)

# fold 2-point WW. Note that we need this for the R-ratio, but the WP gives a cleaner 2-point signal for mpi fits
C2_fold = np.real(fold(C2pt_tavg, T))
# C2_mu = np.mean(C2_fold, axis = 0)
# C2_sigma = np.std(C2_fold, axis = 0, ddof = 1)

# get WP pion mass
meff = get_cosh_effective_mass(C2_pion00WP)
C2_folded = np.real(fold(C2_pion00WP, T))
C2_mu = np.mean(C2_folded, axis=0)
C2_sigma = np.std(C2_folded, axis=0, ddof=1)
meff_folded = fold_meff(meff, T)
meff_mu = np.mean(meff_folded, axis=0)
meff_sigma = np.std(meff_folded, axis=0, ddof=1)

# process data points and make ready to fit
R_mu = np.mean(R_boot, axis=0)
R_sigma = np.std(R_boot, axis=0, ddof=1)
data_slice = np.zeros((n_boot, n_ops, T), dtype=np.float64)
plot_domain = range(T)
for i in range(n_ops):
    for sep in range(T):
        if sep % 2 == 0:
            data_slice[:, i, sep] = np.real(R_boot[:, i, sep // 2, sep])
        else:
            data_slice[:, i, sep] = np.real((R_boot[:, i, sep // 2, sep] + R_boot[:, i, sep // 2 + 1, sep]) / 2)
data_plot_mu = np.mean(data_slice, axis=0)
data_plot_sigma = np.std(data_slice, axis=0, ddof=1)

######################################################################################
#################################### Perform fit #####################################
######################################################################################

# assemble data into correct form
def make_data(corr, domain, lam=1):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T). Here lam is the shrinkage parameter λ, which is 
    set to 1 (fully correlated) by default. 
    """
    d = {t: corr[:, t] for t in domain}
    df = pd.DataFrame(d)
    mean = np.array(df.mean())
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)    # try shrinkage
    #cov = np.array(df.cov())
    return domain, gv.gvar(mean, cov)

# f3
if fitform == 'f3':
    def make_prior(c_widths=[0.1, 0.1, 0.1, 0.1], dE_width=0.4):
        """
        Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
        In the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential
        """
        prior = gv.BufferDict()
        prior['c0'] = gv.gvar(0.0, c_widths[0])
        prior['c1'] = gv.gvar(0.0, c_widths[1])
        prior['c2'] = gv.gvar(0.0, c_widths[2])
        # be more judicious and make this have a larger gap
        prior['log(dE)'] = np.log(gv.gvar(0.1, dE_width))
        prior['log(m)'] = np.log(gv.gvar(mpi_mu, mpi_std))
        return prior

    def fcn(t, p):
        """Single exponential fitting function + backpropagating state."""
        c0 = p['c0']
        c1 = p['c1']
        c2 = p['c2']
        dE = p['dE']
        m = p['m']
        return (c0 + c1*np.exp(-dE*t) + c2 * np.exp(-(m + dE)*(T - 2*t)))
    c_width = [0.1, 0.1, 0.1]
    prior_strs = ['c0', 'c1', 'c2', 'dE']

# f4
if fitform == 'f4':
    def make_prior(c_widths=[0.1, 0.1, 0.1, 0.1], dE_width=0.4):
        """
        Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
        In the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential
        """
        prior = gv.BufferDict()
        prior['c0'] = gv.gvar(0.0, c_widths[0])
        prior['c1'] = gv.gvar(0.0, c_widths[1])
        prior['c2'] = gv.gvar(0.0, c_widths[2])
        prior['c3'] = gv.gvar(0.0, c_widths[3])
        # be more judicious and make this have a larger gap
        prior['log(dE)'] = np.log(gv.gvar(0.1, dE_width))
        prior['log(m)'] = np.log(gv.gvar(mpi_mu, mpi_std))
        return prior

    def fcn(t, p):
        """Single exponential fitting function + backpropagating state."""
        c0 = p['c0']
        c1 = p['c1']
        c2 = p['c2']
        c3 = p['c3']
        dE = p['dE']
        m = p['m']
        return (c0 + c1*np.exp(-dE*t) + c2 * np.exp(-(m + dE)*(T - 2*t))) / (1+c3*np.exp(-(m + dE)*T + (2*m+dE)*2*t))
    c_width = [0.1, 0.1, 0.1, 0.1]
    prior_strs = ['c0', 'c1', 'c2', 'c3', 'dE']

# f5
if fitform == 'f5':
    def make_prior(c_widths=[0.1, 0.1, 0.1, 0.1], dE_width=0.4):
        """
        Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
        In the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential.
        """
        prior = gv.BufferDict()
        prior['c0'] = gv.gvar(0.0, c_widths[0])
        prior['c1'] = gv.gvar(0.0, c_widths[1])
        prior['c2'] = gv.gvar(0.0, c_widths[2])
        prior['c3'] = gv.gvar(0.0, c_widths[3])
        # be more judicious and make this have a larger gap
        prior['log(dE)'] = np.log(gv.gvar(0.1, dE_width))
        prior['log(m)'] = np.log(gv.gvar(mpi_mu, mpi_std))
        return prior

    def fcn(t, p):
        """Single exponential fitting function + backpropagating state."""
        c0 = p['c0']
        c1 = p['c1']
        c2 = p['c2']
        c3 = p['c3']
        dE = p['dE']
        m = p['m']
        return (c0 + c1*np.exp(-dE*t) + c2 * np.exp(-(m + dE)*(T - 2*t))) / (1 + c3*np.exp(-2*dE*t))
    c_width = [0.1, 0.1, 0.1, 0.1]
    prior_strs = ['c0', 'c1', 'c2', 'c3', 'dE']

# f6 (Taylor's version)
if fitform == 'f6':
    def make_prior(c_widths = [0.1, 0.1, 0.1, 0.1, 0.1], dE_width = 0.1):
        """
        Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
        In the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential.
        """
        prior = gv.BufferDict()
        prior['c0'] = gv.gvar(0.0, c_widths[0])
        prior['c1'] = gv.gvar(0.0, c_widths[1])
        prior['c2'] = gv.gvar(0.0, c_widths[2])
        prior['c3'] = gv.gvar(0.0, c_widths[3])
        prior['c4'] = gv.gvar(0.0, c_widths[4])
        #prior['log(dE)'] = np.log(gv.gvar(0.1, dE_width))
        #prior['log(dE)'] = np.log(gv.gvar(mpi_mu, 0.1))
        prior['log(dE)'] = np.log(gv.gvar(mpi_mu, dE_width))
        prior['log(m)'] = np.log(gv.gvar(mpi_mu, mpi_std))
        return prior
    def fcn(t, p):
        """Single exponential fitting function + backpropagating state."""
        c0 = p['c0']
        c1 = p['c1']
        c2 = p['c2']
        c3 = p['c3']
        c4 = p['c4']
        dE = p['dE']
        m = p['m']
        return c0 + c1*np.exp(-dE*t) + c2 * np.exp(-(m + dE)*(T - 2*t)) - c3*np.exp(-2*dE*t) - c4*np.exp(-(m + dE)*T + (2*m+dE)*2*t)
    fitform = 'f6'
    c_width = [0.1, 0.1, 0.1, 0.1, 0.1]
    prior_strs = ['c0', 'c1', 'c2', 'c3', 'c4', 'dE']

# Fit data_slice[b, k, t].
fit_outs = []
fit_chi2_dof = []
Oeff_cv = []
Oeff_std = []
# lam = 0.95
lam = 0.9
# lam = 1.0
for k in range(n_ops):
    print('Fitting ' + op_labels[k])
    domain = exc_domain[k]
    fit_data = data_slice[:, k, :]
    t_dom, Oeff = make_data(fit_data, domain, lam)
    prior = make_prior()
    p0 = None
    fit = lsqfit.nonlinear_fit(data=(t_dom, Oeff), fcn=fcn, prior=prior, p0=p0)

    # print fit and save parameters
    print(fit)
    fit_outs.append(fit)
    fit_chi2_dof.append(fit.chi2 / fit.dof)
    Oeff_cv.append(fit.p['c0'].mean)
    Oeff_std.append(fit.p['c0'].sdev)
Oeff_cv = np.array(Oeff_cv)
Oeff_std = np.array(Oeff_std)

style = styles['notebook']
for k in range(n_ops):
    fit_x, fit_lower, fit_upper = get_fit_band(fit_outs[k].p, fcn, xlims=(exc_domain[k][0], exc_domain[k][-1]))
    O_mu, O_std = fit_outs[k].p['c0'].mean, fit_outs[k].p['c0'].sdev
    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/eff_matelems/' + fitform + '/comparisons/' \
        + ens_path + '/' + op_labels[k] + '.pdf'
    plot_fit(data_plot_mu[k], data_plot_sigma[k], fit_x, fit_lower, fit_upper, yaxis_labels[k], ylims=yrangep[k], yt_locs=ytick_locs[k], \
        yt_labels=ytick_labels[k], col=colors[k], Oeff_mu=Oeff_cv[k], Oeff_sigma=Oeff_std[k], const_mu=c[k], const_sigma=sigmac[k], \
        chi2_dof=fit_chi2_dof[k], saveat=path)

tmin0, tmin_rg = exc_domain[0][0], [6, 7, 8]
tmin0_O3, tmin_rg_O3 = exc_domain[2][0], [9, 10, 11]        # O3 has more excited state contamination
tmax0, tmax_rg = exc_domain[0][-1], [31, 32, 33]
lam0, lam_rg = 0.9, np.arange(0.75, 1.0, 0.05)
ll_strs = ['0.75', '0.8', '0.85', '0.9', '0.95']
# lam0, lam_rg = 0.9, np.arange(0.8, 1.0, 0.05)
# ll_strs = ['0.8', '0.85', '0.9', '0.95']

dE_width = 0.4
prior_widths = copy(c_width)
prior_widths.append(dE_width)
n_priors = len(prior_widths)
key_labels = ['tmin', 'tmax', 'lambda', 'prior']
n_keys = len(key_labels)

# vary t_{min} and see how the posterior for the extracted matrix element reacts. Note we only accept fits with p-value geq pcut
fit_data = [{'tmin': {}, 'tmax': {}, 'lambda': {}, 'prior': {}} for k in range(n_ops)]
prior_vary = [[10 * prior_widths[i] if i == j else prior_widths[i] for i in range(n_priors)] for j in range(n_priors)]
p0 = None
p_cut = 0.01
for k in range(n_ops):
    print('Stability for fit form ' + fitform + ' for O' + str(k))
    Tmin0 = tmin0_O3 if k == 2 else tmin0
    Tmin_rg = tmin_rg_O3 if k == 2 else tmin_rg
    for tm in Tmin_rg:
        tmp = perform_fit(k, tm, tmax0, lam0, c_width, dE_width)
        if tmp.Q > p_cut:
            fit_data[k]['tmin'][str(tm)] = tmp
    for tp in tmax_rg:
        tmp = perform_fit(k, Tmin0, tp, lam0, c_width, dE_width)
        if tmp.Q > p_cut:
            fit_data[k]['tmax'][str(tp)] = tmp
    for il, ll in enumerate(lam_rg):
        ll_key = ll_strs[il]
        tmp = perform_fit(k, Tmin0, tmax0, ll, c_width, dE_width)
        if tmp.Q > p_cut:
            fit_data[k]['lambda'][ll_key] = tmp
    for ip in range(n_priors):
        tmp = perform_fit(k, Tmin0, tmax0, lam0, prior_vary[ip][:n_priors-1], prior_vary[ip][n_priors - 1])
        if tmp.Q > p_cut:
            fit_data[k]['prior'][prior_strs[ip]] = tmp

# unpack fit_data
cut = 0.1
all_cvs, all_stds = [[] for k in range(n_ops)], [[] for k in range(n_ops)]
all_pvals = [[] for k in range(n_ops)]
all_weights = [[] for k in range(n_ops)]
for k in range(n_ops):
    for tmp in fit_data[k].values():
        for ff in tmp.values():
            c0_mu, c0_std = ff.p['c0'].mean, ff.p['c0'].sdev
            pval = ff.Q        # Note ff.Q = scipy.stats.chi2.sf(ff.chi2, ff.dof)
            if pval > cut:
                all_cvs[k].append(c0_mu)
                all_stds[k].append(c0_std)
                all_pvals[k].append(pval)
                all_weights[k].append(pval * (c0_std ** (-2)))
    all_cvs[k], all_stds[k] = np.array(all_cvs[k]), np.array(all_stds[k])
    all_pvals[k], all_weights[k] = np.array(all_pvals[k]), np.array(all_weights[k])
    all_weights[k] = all_weights[k] / np.sum(all_weights[k])        # normalize to 1

# take weighted average
c0_means = []
c0_stds = []
for k in range(n_ops):
    c0bar = np.sum(all_weights[k] * all_cvs[k])
    dc0_stat_sq = np.sum(all_weights[k] * (all_stds[k] ** 2))
    dc0_sys_sq = np.sum(all_weights[k] * ((all_cvs[k] - c0bar)**2))
    c0_sigma = np.sqrt(dc0_stat_sq + dc0_sys_sq)
    c0_means.append(c0bar)
    c0_stds.append(c0_sigma)

fiducial = True
if fiducial:
    fit_cv = Oeff_cv
    fit_std = Oeff_std
else:
    fit_cv = c0_means
    fit_std = c0_stds

style = styles['notebook']
for k in range(n_ops):
    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/eff_matelems/' + fitform + '/stability/' \
        + ens_path + '/' + op_labels[k] + '.pdf'
    plot_stability(k, fit_data, band_cv = fit_cv[k], band_std = fit_std[k], saveat = path)

# plot average
style = styles['prd_twocol']
for k in range(n_ops):
    fit_x, fit_lower, fit_upper = get_fit_band(fit_outs[k].p, fcn, xlims = (exc_domain[k][0], exc_domain[k][-1]))
    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/eff_matelems/' + fitform + '/' + ens_path + \
           '/' + op_labels[k] + '.pdf'
    plot_fit_paper(k, data_plot_mu[k], data_plot_sigma[k], fit_x, fit_lower, fit_upper, yaxis_labels[k], ylims = \
                   yrangep[k], yt_locs = ytick_locs[k], yt_labels = ytick_labels[k], col = colors[k], \
                   Oeff_mu = fit_cv[k], Oeff_sigma = fit_std[k], saveat = path)

print('Done fitting and plotting for ensemble ' + str(ensemble) + ' and fit form ' + str(fitform))

# Make data for bootstrap analysis. Use each bootstrap with the sample covariance matrix to preserve correlations.
def make_data_boots(b_idx, corr, domain, lam=1):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T). Here lam is the shrinkage parameter λ, which is 
    set to 1 (fully correlated) by default. 
    """
    d = {t: corr[:, t] for t in domain}
    df = pd.DataFrame(d)
    cv = corr[b_idx, domain]
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)      # get covariance
    return domain, gv.gvar(cv, cov)

# Fit data_slice[b, k, t].
Oeff_cv = []
Oeff_std = []
# lam = 0.95
lam = 0.9
# lam = 1.0
fits = np.zeros((n_ops, n_boot), dtype = np.float64)
for k in range(n_ops):
    print('Fitting ' + op_labels[k])
    domain = exc_domain[k]
    fit_data = data_slice[:, k, :]
    prior = make_prior()
    p0 = None
    for b in range(n_boot):
        t_dom, Oeff = make_data_boots(b, fit_data, domain, lam)
        fit = lsqfit.nonlinear_fit(data=(t_dom, Oeff), fcn=fcn, prior=prior, p0=p0)
        fit_val = fit.p['c0'].mean
        print('Chi^2/dof = ' + str(fit.chi2 / fit.dof) + ', <O_k> = ' + str(fit_val))          # print fit info
        fits[k, b] = fit_val
print('Value of <O_k> from full fit: ' + export_float_latex(fit_cv, fit_std, sf = 2))
print('Value of <O_k> from bootstrapped fit: ' + export_float_latex(np.mean(fits, axis = 1), np.std(fits, axis = 1, ddof = 1), sf = 2))

# Save output
out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/bare_matrix_elements/' \
                + ensemble + '/fit_params_exc_' + fitform + '.h5'
fout = h5py.File(out_file, 'w')
fout['fits'] = fits.T
fout['data_slice'] = data_slice
fout['c'] = np.array(fit_cv)
fout['sigmac'] = np.array(fit_std)
fout['plot_domain'] = plot_domain
for k in range(n_ops):
    fout['fit_domain/' + str(op_labels[k])] = np.array(exc_domain[k], dtype = np.float64)
fout.close()
print('Results output to: ' + out_file)