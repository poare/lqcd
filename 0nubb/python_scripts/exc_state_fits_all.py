from __future__ import print_function

######################################################################################
# This script fits the excited states for the effective matrix element data          #
# #$O_k^{eff}(t)$ for the fiducial mode f3 and generates the stability plot. The     #
# script takes as input the ensemble index {0 : 24I/ml0p01, 1 : 24I/ml0p005, 2 :     #
# 32I/ml0p008, 3 : 32I/ml0p006, 4 : 32I/ml0p004}.                                    #
# To call the script, run:                                                           #
# > python3 exc_state_fits_paper.py ens_idx                                          #
#                                                                                    #
# This is the production code for using the model-averaged excited state fits,       #
# however will likely not make its way into the paper as we'll instead be Taylor     #
# expanding the original model.                                                      #
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
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

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

ensemble = ['24I/ml_0p01', '24I/ml_0p005',
            '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
ens_path = ['24I/ml0p01', '24I/ml0p005', '32I/ml0p008',
            '32I/ml0p006', '32I/ml0p004'][ens_idx]
fitforms = ['f3', 'f4', 'f5']
n_models = len(fitforms)
model_labels = ['Model A', 'Model B', 'Model C']
f_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/bare_matrix_elements/' + \
    ensemble + '/fit_params.h5'
n_ops = 5

# read in input to plot
f = h5py.File(f_path, 'r')
data_slice = f['data_slice'][()]
c = f['c'][()]
sigmac = f['sigmac'][()]
plot_domain = f['plot_domain'][()]
f.close()

data_plot_mu = np.mean(data_slice, axis=0)
data_plot_sigma = np.std(data_slice, axis=0, ddof=1)

# set specific ranges and labels
ytick_labels = [
    [['-5.8', '-5.4', '-5.0'], ['-1.0', '-0.9', '-0.8'], ['3.4', '3.6', '3.8'], ['-1.9', '-1.7', '-1.5'], ['2.1', '2.3', '2.5']],               # 24I/ml0p01
    [['-5.0', '-4.7', '-4.4'], ['-8.5', '-8.0', '-7.5'], ['1.70', '1.85', '2.00'], ['-1.55', '-1.45', '-1.35'], ['1.80', '1.95', '2.10']],      # 24I/ml0p005
    [['-1.90', '-1.75', '-1.60'], ['-2.9', '-2.7', '-2.5'], ['8.00', '8.75', '9.50'], ['-5.8', '-5.3', '-4.8'], ['6.2', '6.7', '7.2']],         # 32I/ml0p008
    [['-1.70', '-1.55', '-1.40'], ['-2.7', '-2.5', '-2.3'], ['5.7', '6.2', '6.7'], ['-5.5', '-5.0', '-4.5'], ['5.8', '6.2', '6.6']],            # 32I/ml0p006
    [['-1.55', '-1.45', '-1.35'], ['-2.40', '-2.25', '-2.10'], ['3.4', '3.8', '4.2'], ['-4.9', '-4.5', '-4.1'], ['5.2', '5.6', '6.0']],         # 32I/ml0p004
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
# yaxis_labels = [r'$' + latex_labels[ii] + r'^{\mathrm{eff}} \hspace{-1.0mm} \times \hspace{-0.5mm} 10^{' + str(pwr[ii]) + r'}$' for ii in range(len(latex_labels))]
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
        # plt.fill_between(fit_x, fit_lower, fit_upper, color = col, alpha = 0.4, linewidth = 0.0, label = 'Extrapolation')
        plt.fill_between(fit_x, fit_lower, fit_upper, color = col, alpha = 0.4, linewidth = 0.0, label = 'Fiducial fit')
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
stab_colors = ['r', 'b', 'g', 'k']
stab_leg_labels = [r'$t_\mathrm{min}$', r'$t_\mathrm{max}$', r'$\lambda$', r'Prior']
def plot_stability_all(k, fit_dict, key_labels = ['tmin', 'tmax', 'lambda', 'prior'], band_cv = None, band_std = None, saveat = None):
    n_keys = len(key_labels)
    legend_elems = [Line2D([0], [0], color = stab_colors[ii], lw = 4, label = stab_leg_labels[ii]) for ii in range(n_keys)]
    # TODO add band_cv to legend if we need to
    asp_ratio = 2.0
    fig_size = (style['colwidth'] * asp_ratio, style['colwidth'])
    start, doms = 0, [[] for i in range(len(model_labels))]
    for im, lbl in enumerate(model_labels):
        for key in key_labels:
            dint = len(fit_dict[lbl][k][key])
            doms[im].append(np.arange(start, start + dint))
            start += dint
    with sns.plotting_context('paper'):
        all_labels = []
        all_doms = np.array([])
        fig = plt.figure(figsize = fig_size)

        # ax = host_subplot(111, axes_class = AA.Axes, figure = fig)
        # plt.subplots_adjust(bottom = 0.2)
        # ax2 = ax.twiny()
        # offset = -40
        # new_fixed_axis = ax2.get_grid_helper().new_fixed_axis
        # ax2.axis['bottom'] = new_fixed_axis(loc = 'bottom',
        #                                     axes = ax2,
        #                                     offset = (0, offset))
        # ax2.axis['bottom'].toggle(all = True)

        for im, lbl in enumerate(model_labels):
            for ii, param_key in enumerate(key_labels):
                c0_dat = fit_dict[lbl][k][param_key]
                if len(c0_dat) == 0:        # if this variation has no accepted fits
                    continue
                labels = c0_dat.keys()    # TODO make this more versatile?
                cvs = np.array([c0_dat[jj].p['c0'].mean for jj in labels]) * (10 ** (-pwr[k]))
                stds = np.array([c0_dat[jj].p['c0'].sdev for jj in labels]) * (10 ** (-pwr[k]))
                _, caps, _ = plt.errorbar(doms[im][ii], cvs, yerr = stds, fmt = stab_formats[ii], c = stab_colors[ii], capsize = style['endcaps'], \
                            markersize = style['markersize'], elinewidth = style['ebar_width'])
                # _, caps, _ = ax.errorbar(doms[im][ii], cvs, yerr = stds, fmt = stab_formats[ii], c = stab_colors[ii], capsize = style['endcaps'], \
                #             markersize = style['markersize'], elinewidth = style['ebar_width'])
                for cap in caps:
                    cap.set_markeredgewidth(style['ecap_width'])
                plt.ylabel(yaxis_labels[k], fontsize = style['fontsize'])
                # ax.set_ylabel(yaxis_labels[k], fontsize = style['fontsize'])
                all_doms = np.append(all_doms, doms[im][ii])
                all_labels.extend(labels)
        xlims = (-1, all_doms[-1] + 1)
        if band_cv:
            band_cv *= (10 ** (-pwr[k]))
            band_std *= (10 ** (-pwr[k]))
            plt.fill_between(xlims, band_cv + band_std, band_cv - band_std, color = 'b', alpha = 0.15, linewidth = 0.0, label = 'Fiducial')
            # ax.fill_between(xlims, band_cv + band_std, band_cv - band_std, color = 'b', alpha = 0.15, linewidth = 0.0, label = 'Fiducial')
        
        ax = plt.gca()
        ax.set_xlim(xlims)
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'] * 2/3, labelrotation=90)
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'] * 2/3)
        # ax.set_yticks(ytick_locs[k])
        # ax.set_yticklabels(ytick_labels[k], fontdict = {'fontsize' : style['fontsize'] * 2/3})
        ax.set_xticks(all_doms, fontsize = style['fontsize'], rotation = 90)
        ax.set_xticklabels(all_labels, fontdict = {'fontsize' : style['fontsize'] * 2/3}, rotation = 90)
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        ax.legend(handles = legend_elems, loc = 'upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 16})

        # original code
        ax2 = ax.twiny()
        ax2.spines["bottom"].set_position(("axes", -0.10))
        ax2.set_xlim(xlims)
        ax2.tick_params('both', length=0, width=0, which='minor')
        ax2.tick_params('both', direction='in', which='major')
        ax2.xaxis.set_tick_params(width = style['tickwidth'], length = 2*style['ticklength'])
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # set major tick positions
        dtick = lambda idx : 1 if (idx == len(model_labels) - 1) else 0.5
        # last_ticks = [doms[im][-1][-1] + dtick(im) for im in range(len(model_labels))]
        last_ticks = [max([max(x) if len(x) > 0 else -1 for x in doms[im]]) + dtick(im) for im in range(len(model_labels))]
        # print(last_ticks)
        major_tick_locs = [-1.]
        major_tick_locs.extend(last_ticks)
        major_tick_locs = np.array(major_tick_locs, dtype = np.float64)
        ax2.set_xticks(major_tick_locs)
        ax2.xaxis.set_major_formatter(ticker.NullFormatter())

        # set major tick labels
        major_label_locs = [(major_tick_locs[i + 1] - major_tick_locs[i]) / 2. + major_tick_locs[i] for i in range(len(model_labels))]
        ax2.xaxis.set_minor_locator(ticker.FixedLocator(major_label_locs))
        ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(model_labels))

        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        # ax.set_xticks(fontsize = style['fontsize'])
        # ax.set_yticks(fontsize = style['fontsize'])
        plt.tight_layout()
        if saveat:
            plt.savefig(saveat, bbox_inches='tight')
        plt.close()

weight_formats = ['x', '+', 'v']
weight_colors = ['r', 'g', 'k']
# weight_leg_labels = [r'$t_\mathrm{min}$', r'$t_\mathrm{max}$', r'$\lambda$', r'Prior']
weight_leg_labels = model_labels
def weight_plot(k, weights, fit_cvs, fit_stds, band_cv = None, band_std = None, saveat = None):
    """
    All data input (weights, fit_cvs, fit_stds) should be a list of n_models lists, where each inner list are the 
    accepted fits for the given model.
    Parameters
    ----------
    fit_cvs : [[]]
        
    """
    legend_elems = [Line2D([0], [0], color = weight_colors[ii], lw = 4, label = weight_leg_labels[ii]) for ii in range(n_models)]
    asp_ratio = 2.0
    fig_size = (style['colwidth'] * asp_ratio, style['colwidth'])

    # iterate over fits, get weights and assign a color
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        for ii, lbl in enumerate(model_labels):
            cvs = np.array(fit_cvs[ii], dtype = np.float64) * (10 ** (-pwr[k]))
            stds = np.array(fit_stds[ii], dtype = np.float64) * (10 ** (-pwr[k]))
            _, caps, _ = plt.errorbar(weights[ii], cvs, yerr = stds, fmt = weight_formats[ii], c = weight_colors[ii], \
                capsize = style['endcaps'], markersize = style['markersize'], elinewidth = style['ebar_width'])
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('Weight', fontsize = style['fontsize'])
        plt.ylabel(yaxis_labels[k], fontsize = style['fontsize'])
        min_weight, max_weight = min([min(w) for w in weights]), max([max(w) for w in weights])
        xlims = (min_weight * 0.95, max_weight * 1.05)
        if band_cv:
            band_cv *= (10 ** (-pwr[k]))
            band_std *= (10 ** (-pwr[k]))
            plt.fill_between(xlims, band_cv + band_std, band_cv - band_std, color = 'b', alpha = 0.15, linewidth = 0.0, label = 'Weighted Average')
        ax = plt.gca()
        ax.set_xlim(xlims)
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'] * 2/3)
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'] * 2/3)
        # ax.set_yticks(ytick_locs[k])
        # ax.set_yticklabels(ytick_labels[k], fontdict = {'fontsize' : style['fontsize'] * 2/3})
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        ax.legend(handles = legend_elems, loc = 'upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 16})

        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
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

def init_fit_params(fitform):
    # ranges to fit. TODO make sure O3 is picked correctly. In particular right now it breaks on Model B for 24I/ml0p005
    if fitform == 'f3' or fitform == 'f5':
        exc_domain = [
            # [np.arange(5, 33), np.arange(5, 33), np.arange(8, 33), np.arange(5, 33), np.arange(5, 33)],
            [np.arange(6, 33), np.arange(6, 33), np.arange(8, 33), np.arange(6, 33), np.arange(6, 33)], 
            [np.arange(6, 33), np.arange(6, 33), np.arange(8, 33), np.arange(6, 33), np.arange(6, 33)], 
            [np.arange(6, 33), np.arange(6, 33), np.arange(8, 33), np.arange(6, 33), np.arange(6, 33)], 
            [np.arange(6, 33), np.arange(6, 33), np.arange(8, 33), np.arange(6, 33), np.arange(6, 33)], 
            [np.arange(6, 33), np.arange(6, 33), np.arange(9, 33), np.arange(6, 33), np.arange(6, 33)], 
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
        # prior_strs = ['c0', 'c1', 'c2', 'dE']
        prior_strs = [r'$\langle\mathcal{O}_k\rangle$', r'$A_1$', r'$A_2$', r'$\Delta$']

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
        # prior_strs = ['c0', 'c1', 'c2', 'c3', 'dE']
        prior_strs = [r'$\langle\mathcal{O}_k\rangle$', r'$B_1$', r'$B_2$', r'$B_3$', r'$\Delta$']

    # f5 (Model C)
    if fitform == 'f5':
        def make_prior(c_widths = [0.1, 0.1, 0.1, 0.1], dE_width = 0.4):
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
        # prior_strs = ['c0', 'c1', 'c2', 'c3', 'dE']
        prior_strs = [r'$\langle\mathcal{O}_k\rangle$', r'$C_1$', r'$C_2$', r'$C_3$', r'$\Delta$']
    return exc_domain, make_prior, fcn, c_width, prior_strs

def get_key(tm, tp, lam):
    return '(' + str(tm) + ', ' + str(tp) + '; ' + str(lam) + ')'

# loop over all fit forms
all_fits = {}
fid_idx = 0                                                 # for comparison, use Model A
fid_fits = [0 for k in range(n_ops)]                        # fiducial fit values
# acc_cvs, acc_stds = [[] for k in range(n_ops)], [[] for k in range(n_ops)], [[] for k in range(n_ops)]
acc_fits, acc_weights, acc_fit_model = [[] for k in range(n_ops)], [[] for k in range(n_ops)], [[] for k in range(n_ops)]
acc_cvs, acc_stds = [[] for k in range(n_ops)], [[] for k in range(n_ops)]
for ff_idx, fitform in enumerate(fitforms):
    model_label = model_labels[ff_idx]
    exc_domain, make_prior, fcn, c_width, prior_strs = init_fit_params(fitform)
    
    # TODO fit data_slice[b, k, t].
    fit_outs, fit_chi2_dof = [], []
    Oeff_cv, Oeff_std = [], []
    lam = 0.9
    for k in range(n_ops):
        print('Fitting ' + op_labels[k])
        domain = exc_domain[k]
        fit_data = data_slice[:, k, :]
        t_dom, Oeff = make_data(fit_data, domain, lam)
        prior = make_prior()
        p0 = None
        fit = lsqfit.nonlinear_fit(data=(t_dom, Oeff), fcn=fcn, prior=prior, p0=p0)
        print(fit)
        fit_outs.append(fit)
        fit_chi2_dof.append(fit.chi2 / fit.dof)
        Oeff_cv.append(fit.p['c0'].mean)
        Oeff_std.append(fit.p['c0'].sdev)
        if ff_idx == fid_idx:            # save fiducial fit
            fid_fits[k] = fit
            fid_fcn = fcn
            print('Fiducial value for ' + op_labels[k] + ': ' + export_float_latex(fid_fits[k].p['c0'].mean, fid_fits[k].p['c0'].sdev, sf = 2))
    Oeff_cv = np.array(Oeff_cv)
    Oeff_std = np.array(Oeff_std)

    # tmin0, tmin_rg = exc_domain[0][0], [5, 6, 7, 8]
    tmin0, tmin_rg = exc_domain[0][0], [6, 7, 8]
    tmin0_O3, tmin_rg_O3 = exc_domain[2][0], [9, 10, 11]        # O3 has more excited state contamination
    tmax0, tmax_rg = exc_domain[0][-1], [31, 32, 33]
    # lam0, lam_rg = 0.9, np.arange(0.8, 1.0, 0.05)
    # lam0_str, ll_strs = '0.9', ['0.8', '0.85', '0.9', '0.95']
    # lam0, lam_rg = 0.9, np.arange(0.85, 1.0, 0.05)
    lam0, lam_rg = 0.9, np.array([0.85, 0.9, 0.95])
    lam0_str, ll_strs = '0.9', ['0.85', '0.9', '0.95']

    dE_width = 0.4
    prior_widths = copy(c_width)
    prior_widths.append(dE_width)
    n_priors = len(prior_widths)
    key_labels = ['tmin', 'tmax', 'lambda', 'prior']
    n_keys = len(key_labels)

    # vary t_{min} and see how the posterior for the extracted matrix element reacts. Note we only accept fits with p-value geq pcut
    all_fits[model_label] = [{'tmin': {}, 'tmax': {}, 'lambda': {}, 'prior': {}} for k in range(n_ops)]
    prior_vary = [[10 * prior_widths[i] if i == j else prior_widths[i] for i in range(n_priors)] for j in range(n_priors)]
    p0 = None
    p_cut = 0.01
    # could also cut on weights
    for k in range(n_ops):
        print('Stability (fiducial) for fit form ' + fitform + ' for O' + str(k))
        Tmin0 = tmin0_O3 if k == 2 else tmin0
        Tmin_rg = tmin_rg_O3 if k == 2 else tmin_rg
        for tm in Tmin_rg:
            tmp = perform_fit(k, tm, tmax0, lam0, c_width, dE_width)
            if tmp.Q > p_cut:
                all_fits[model_label][k]['tmin'][str(tm)] = tmp
                # all_fits[model_label][k]['tmin'][get_key(tm, tmax0, lam0_str)] = tmp
                # acc_cvs[k].append(tmp.p['c0'].mean)
                # acc_stds[k].append(tmp.p['c0'].sdev)
                # acc_weights[k].append(tmp.Q * (tmp.p['c0'].sdev**(-2)))
        for tp in tmax_rg:
            tmp = perform_fit(k, Tmin0, tp, lam0, c_width, dE_width)
            if tmp.Q > p_cut:
                all_fits[model_label][k]['tmax'][str(tp)] = tmp
                # all_fits[model_label][k]['tmax'][get_key(Tmin0, tp, lam0_str)] = tmp
                # acc_cvs[k].append(tmp.p['c0'].mean)
                # acc_stds[k].append(tmp.p['c0'].sdev)
                # acc_weights[k].append(tmp.Q * (tmp.p['c0'].sdev**(-2)))
        for il, ll in enumerate(lam_rg):
            ll_key = ll_strs[il]
            tmp = perform_fit(k, Tmin0, tmax0, ll, c_width, dE_width)
            if tmp.Q > p_cut:
                all_fits[model_label][k]['lambda'][ll_key] = tmp
                # all_fits[model_label][k]['lambda'][get_key(Tmin0, tmax0, ll_key)] = tmp
                # acc_cvs[k].append(tmp.p['c0'].mean)
                # acc_stds[k].append(tmp.p['c0'].sdev)
                # acc_weights[k].append(tmp.Q * (tmp.p['c0'].sdev**(-2)))
        for ip in range(n_priors):
            tmp = perform_fit(k, Tmin0, tmax0, lam0, prior_vary[ip][:n_priors-1], prior_vary[ip][n_priors - 1])
            if tmp.Q > p_cut:
                all_fits[model_label][k]['prior'][prior_strs[ip]] = tmp
                # acc_cvs[k].append(tmp.p['c0'].mean)
                # acc_stds[k].append(tmp.p['c0'].sdev)
                # acc_weights[k].append(tmp.Q * (tmp.p['c0'].sdev**(-2)))
        
        print('Stability (all domains) for fit form ' + fitform + ' for O' + str(k))
        all_doms = []                       # should make all_doms --> list of elements [tmin, tmax, lambda, [c_priors], dE_prior]
        for tmin in Tmin_rg:
            for tmax in tmax_rg:
                for lam in lam_rg:
                    for c_ind in itertools.product(range(2), repeat = n_priors - 1):   # i.e. (0, 1, 1) means [c0, 10c1, 10c2]
                        for dE_ind in range(2):
                            dE_scale = 1 if dE_ind == 0 else 10
                            cur_dom = [tmin, tmax, lam]
                            c_dom = []
                            for c_idx, ind in enumerate(c_ind):
                                c_scale = 1 if ind == 0 else 10
                                c_dom.append(c_width[c_idx] * c_scale)
                            cur_dom.append(c_dom)
                            cur_dom.append(prior_widths[-1] * dE_scale)
                            all_doms.append(cur_dom)
        print('Number of domains: ' + str(len(all_doms)))
        # print('Example: ' + str(all_doms[0]))
        for dom in all_doms:
            tmp = perform_fit(k, *dom)
            if tmp.Q > p_cut:
                # TODO give acc_cvs and such another index for ff_idx
                acc_fits[k].append(tmp)
                # acc_cvs[k].append(tmp.p['c0'].mean)
                # acc_stds[k].append(tmp.p['c0'].sdev)
                acc_weights[k].append(tmp.Q * (tmp.p['c0'].sdev**(-2)))
                acc_fit_model[k].append(ff_idx)             # tracker for what model it came from
        print('Accepted domains: ' + str(len(acc_fits[k])))
    print('Done iterating for model: ' + model_label)

use_fiducial = False
if use_fiducial:            # Use fiducial model and vary parameters.
    Oeff_final_cvs = [fid_fits[k].p['c0'].mean for k in range(n_ops)]
    Oeff_final_stds = [fid_fits[k].p['c0'].sdev for k in range(n_ops)]
else:                       # Loop over a large domain and use weighted average
    Oeff_final_cvs = []
    Oeff_final_stds = []
    for k in range(n_ops):
        acc_cvs[k] = np.array([fit.p['c0'].mean for fit in acc_fits[k]], dtype = np.float64)
        acc_stds[k] = np.array([fit.p['c0'].sdev for fit in acc_fits[k]], dtype = np.float64)
        # acc_cvs[k] = np.array(acc_cvs[k], dtype = np.float64)
        # acc_stds[k] = np.array(acc_stds[k], dtype = np.float64)
        acc_weights[k] = np.array(acc_weights[k], dtype = np.float64)
        acc_weights[k] /= np.sum(acc_weights[k])
        # c0_weighted_mean = np.sum(acc_weights[k] * acc_cvs[k])
        # dc0_stat_sq = np.sum(acc_weights[k] * (acc_stds[k]**2))
        # dc0_sys_sq = np.sum(acc_weights[k] * ((acc_cvs[k] - c0_weighted_mean)**2))
        c0_weighted_mean = np.sum(acc_weights[k] * acc_cvs[k])
        dc0_stat_sq = np.sum(acc_weights[k] * (acc_stds[k]**2))
        dc0_sys_sq = np.sum(acc_weights[k] * ((acc_cvs[k] - c0_weighted_mean)**2))
        c0_weighted_std = np.sqrt(dc0_stat_sq + dc0_sys_sq)
        print('Weighted average for ' + op_labels[k] + ': ' + export_float_latex(c0_weighted_mean, c0_weighted_std, sf = 2))
        Oeff_final_cvs.append(c0_weighted_mean)
        Oeff_final_stds.append(c0_weighted_std)

######################################################################################
############################ Generate plots and save data ############################
######################################################################################

# comparison plots
style = styles['notebook']
for k in range(n_ops):
    fit_x, fit_lower, fit_upper = get_fit_band(fid_fits[k].p, fid_fcn, xlims=(exc_domain[k][0], exc_domain[k][-1]))
    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/eff_matelems/fit_plots/comparisons/' \
        + ens_path + '/' + op_labels[k] + '.pdf'
    plot_fit(data_plot_mu[k], data_plot_sigma[k], fit_x, fit_lower, fit_upper, yaxis_labels[k], ylims = yrangep[k], yt_locs = ytick_locs[k], \
        yt_labels = ytick_labels[k], col = colors[k], Oeff_mu = Oeff_final_cvs[k], Oeff_sigma = Oeff_final_stds[k], const_mu = c[k], const_sigma = sigmac[k], \
        saveat = path)
        # chi2_dof = fit_chi2_dof[k], saveat = path)

# individual plots for paper
style = styles['prd_twocol']
for k in range(n_ops):
    fit_x, fit_lower, fit_upper = get_fit_band(fid_fits[k].p, fid_fcn, xlims = (exc_domain[k][0], exc_domain[k][-1]))
    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/final_eff_matelems/' + ens_path + '/' + op_labels[k] + '.pdf'
    plot_fit_paper(k, data_plot_mu[k], data_plot_sigma[k], fit_x, fit_lower, fit_upper, yaxis_labels[k], ylims = \
                yrangep[k], yt_locs = ytick_locs[k], yt_labels = ytick_labels[k], col = colors[k], \
                Oeff_mu = Oeff_final_cvs[k], Oeff_sigma = Oeff_final_stds[k], saveat = path)

# notebook-style stability plots
style = styles['notebook']
for k in range(n_ops):
    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/eff_matelems/fit_plots/stability/' \
        + ens_path + '/' + op_labels[k] + '.pdf'
    plot_stability_all(k, all_fits, saveat = path, band_cv = Oeff_final_cvs[k], band_std = Oeff_final_stds[k])

# Weighted average plots for paper. 
style = styles['prd_twocol']
for k in range(n_ops):
    # rearrange fit_cvs, etc. for plotting
    plot_weights, plot_cvs, plot_stds = [[] for _ in range(n_models)], [[] for _ in range(n_models)], [[] for _ in range(n_models)]
    for ii, ff_idx in enumerate(acc_fit_model[k]):
        plot_weights[ff_idx].append(acc_weights[k][ii])
        plot_cvs[ff_idx].append(acc_cvs[k][ii])
        plot_stds[ff_idx].append(acc_stds[k][ii])

    path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/final_weighted_avg/' + ens_path + '/' + op_labels[k] + '.pdf'
    weight_plot(k, plot_weights, plot_cvs, plot_stds, saveat = path, band_cv = Oeff_final_cvs[k], band_std = Oeff_final_stds[k])

# Save data

print('Done fitting and plotting for ensemble ' + str(ensemble))