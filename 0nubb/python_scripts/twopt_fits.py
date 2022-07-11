from __future__ import print_function

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

# Set the ensemble index.
ens_idx = int(sys.argv[1])
ensemble = ['24I/ml_0p01', '24I/ml_0p005', '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
ens_path = ['24I/ml0p01', '24I/ml0p005', '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
ens_label = ['24Iml0p01', '24Iml0p005', '32Iml0p008', '32Iml0p006', '32Iml0p004'][ens_idx]
f_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/bare_matrix_elements/' + ensemble + '/fit_params.h5'
n_ops = 5
lam = 0.9
twopt_parent = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/twopt_fits/'

# set priors and fit domains for each ensemble
mpi_domain = [np.arange(10, 25), np.arange(10, 25), np.arange(10, 25), np.arange(10, 25), np.arange(10, 25)][ens_idx]
backprop_domain = [np.arange(6, 30), np.arange(6, 30), np.arange(6, 30), np.arange(6, 30), np.arange(6, 25)][ens_idx]
dE_domain = [np.arange(2, 30), np.arange(2, 30), np.arange(2, 30), np.arange(2, 30), np.arange(2, 30)][ens_idx]

m_prior = [gv.gvar(0.2, 0.1), gv.gvar(0.2, 0.1), gv.gvar(0.2, 0.1), gv.gvar(0.2, 0.1), gv.gvar(0.2, 0.1)][ens_idx]
dE_prior = [gv.gvar(0.5, 0.5), gv.gvar(0.5, 0.5), gv.gvar(0.5, 0.5), gv.gvar(0.5, 0.5), gv.gvar(0.5, 0.5)][ens_idx]

asp_ratio = 4/3
def plot_data(cvs, stds, yaxis_label, ylims = None, yt_locs = None, yt_labels = None, mrk = '.', col = 'r', saveat = None):
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
        if saveat:
            plt.savefig(saveat, bbox_inches='tight')
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

# Use this code block if I make this notebook into production code
f3pt_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/analysis_output/' +ensemble+ '/SD_output.h5'

f = h5py.File(f3pt_path, 'r')
L, T = f['L'][()], f['T'][()]
plot_domain = range(T)
vol = (L**3)# * T
C2pt_tavg = f['pion-00WW'][()]
# C2WW = f['C2pt'][()]
C2_pion00WP = np.real(f['pion-00WP'][()]) / vol
C3pt_tavg = f['C3pt'][()]
Cnpt = f['Cnpt'][()]
R_boot = f['R'][()]
mpi_boot = f['mpi'][()]
f.close()

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

# get WP pion mass
C2_fold = np.real(fold(C2pt_tavg, T))
meff = get_cosh_effective_mass(C2_pion00WP)
C2_folded = np.real(fold(C2_pion00WP, T))
C2_mu = np.mean(C2_folded, axis = 0)
C2_sigma = np.std(C2_folded, axis = 0, ddof = 1)
meff_folded = fold_meff(meff, T)
meff_mu = np.mean(meff_folded, axis = 0)
meff_sigma = np.std(meff_folded, axis = 0, ddof = 1)

# plot 2-points
plot_data(C2_mu, C2_sigma, '$C_{2\mathrm{pt}}$', saveat = twopt_parent + 'data/' + ens_label + 'C2.pdf')
plot_data(meff_mu, meff_sigma, '$am_{\mathrm{eff}}$', saveat = twopt_parent + 'data/' + ens_label + 'eff_mass.pdf')

# assemble data into correct form
print('Fitting effective mass.')
def make_data(corr, domain, lam = 1.0):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T).
    """
    d = {t : corr[:, t] for t in domain}
    df = pd.DataFrame(d)
    mean = np.array(df.mean())
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)    # try shrinkage
    return domain, gv.gvar(mean, cov)

def make_prior():
    """
    Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
    Note that in the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential
    """
    prior = gv.BufferDict()
    prior['m'] = gv.gvar(0.0, 1.0)
    return prior

def fcn(t, p):
    """Constant fitting function f(t) = m."""
    m = p['m']
    return m + 0*t

t_dom, meff = make_data(meff_folded, mpi_domain, lam = lam)
prior = make_prior()
p0 = None
fit = lsqfit.nonlinear_fit(data = (t_dom, meff), fcn = fcn, prior = prior, p0 = p0)
print(fit)
mean_c, std_c = fit.p['m'].mean, fit.p['m'].sdev

fit_x, fit_lower, fit_upper = get_fit_band(fit.p, fcn, xlims = (mpi_domain[0], mpi_domain[-1]))
plot_fit(meff_mu, meff_sigma, fit_x, fit_lower, fit_upper, '$(am_{\mathrm{eff}})^{\mathrm{const}}$', \
    saveat = twopt_parent + 'const/' + ens_label + 'meff_fit.pdf')

# Fit C2pt
print('Fitting two point correlator with one state.')
def make_data(corr, domain, lam = 1.0):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T).
    """
    d = {t : corr[:, t] for t in domain}
    df = pd.DataFrame(d)
    mean = np.array(df.mean())
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)    # try shrinkage
    return mpi_domain, gv.gvar(mean, cov)

def make_prior():
    """
    Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
    Note that in the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential
    """
    prior = gv.BufferDict()
    prior['Z'] = gv.gvar(40, 40)
    prior['log(m)'] = np.log(m_prior)
    return prior

def fcn(t, p):
    """Single exponential fitting function f(t) = Z e^{-mt}."""
    Z = p['Z']
    m = p['m']
    return Z * np.exp(-m * t)

# mpi_domain = np.arange(10, 16)    # domain to modify
t_dom, C2 = make_data(C2_folded, mpi_domain, lam = lam)
prior = make_prior()
p0 = None
fit = lsqfit.nonlinear_fit(data = (t_dom, C2), fcn = fcn, prior = prior, p0 = p0)
print(fit)
mean_c, std_c = fit.p['m'].mean, fit.p['m'].sdev
Z_mu, Z_std = fit.p['Z'].mean, fit.p['Z'].sdev

fit_x, fit_lower, fit_upper = get_fit_band(fit.p, fcn, xlims = (mpi_domain[0], mpi_domain[-1]))
plot_fit(C2_mu, C2_sigma, fit_x, fit_lower, fit_upper, '$C_{2\mathrm{pt}}$', ylims = (0, 2), \
    saveat = twopt_parent + 'const/' + ens_label + 'corr_fit.pdf')

# backprop exponential
print('Fitting backpropagating model.')
def make_data(corr, domain, lam = 1.0):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T).
    """
    d = {t : corr[:, t] for t in domain}
    df = pd.DataFrame(d)
    mean = np.array(df.mean())
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)    # try shrinkage
    return backprop_domain, gv.gvar(mean, cov)

def make_prior():
    """
    Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
    Note that in the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential
    """
    prior = gv.BufferDict()
    prior['Z0'] = gv.gvar(40, 40)
    prior['Z1'] = gv.gvar(40, 40)
    prior['log(m)'] = np.log(m_prior)
    return prior

def fcn(t, p):
    """Exponential fitting function f(t) = Z0 e^{-mt} + Z1 e^{-m(T - t)}."""
    Z0 = p['Z0']
    Z1 = p['Z1']
    m = p['m']
    return Z0 * np.exp(-m * t) + Z1 * np.exp(-m * (T - t))

t_dom, C2 = make_data(C2_folded, backprop_domain, lam = lam)
prior = make_prior()
p0 = None
fit = lsqfit.nonlinear_fit(data = (t_dom, C2), fcn = fcn, prior = prior, p0 = p0)
print(fit)
mean_c, std_c = fit.p['m'].mean, fit.p['m'].sdev
Z0_mu, Z0_std = fit.p['Z0'].mean, fit.p['Z0'].sdev
Z1_mu, Z1_std = fit.p['Z1'].mean, fit.p['Z1'].sdev

fit_x, fit_lower, fit_upper = get_fit_band(fit.p, fcn, xlims = (backprop_domain[0], backprop_domain[-1]))
plot_fit(C2_mu, C2_sigma, fit_x, fit_lower, fit_upper, '$C_{2\mathrm{pt}}$', ylims = (0, 5), \
    saveat = twopt_parent + 'const/' + ens_label + 'corr_backprop_fit.pdf')

# Excited state fits
print('Fitting two point correlator data with one excited state.')
def make_data(corr, domain, lam = 1.0):
    """
    Makes data for lsqfit. corr is an np.array of shape (n_boot, T).
    """
    d = {t : corr[:, t] for t in domain}
    df = pd.DataFrame(d)
    mean = np.array(df.mean())
    full_cov = np.array(df.cov())
    cov = shrinkage(full_cov, lam)    # try shrinkage
    return dE_domain, gv.gvar(mean, cov)

def make_prior():
    """
    Make priors for fit parameters. For the constant ground state fit, the only prior is n the parameter a. 
    Note that in the examples, a is a vector [a[0], ..., a[N]], where a[i] is the amplitude for the ith exponential
    """
    prior = gv.BufferDict()
    prior['Z0'] = gv.gvar(40, 40)
    prior['Z1'] = gv.gvar(40, 40)
    #prior['Z2'] = gv.gvar(20, 40)
    prior['Z2'] = gv.gvar(0, 20)
    prior['log(m)'] = np.log(m_prior)
    prior['log(dE)'] = np.log(dE_prior)
    return prior

def fcn(t, p):
    """Exponential fitting function f(t) = Z0 e^{-mt} + Z1 e^{-m(T - t)}."""
    Z0 = p['Z0']
    Z1 = p['Z1']
    Z2 = p['Z2']
    m = p['m']
    dE = p['dE']
    return Z0 * np.exp(-m * t) + Z1 * np.exp(-m * (T - t)) + Z2 * np.exp(-(m + dE) * t)

# mpi_domain = np.arange(5, 25)    # domain to modify
# mpi_domain = np.arange(4, 25)    # domain to modify
t_dom, C2 = make_data(C2_folded, dE_domain, lam = lam)
prior = make_prior()
p0 = None
fit = lsqfit.nonlinear_fit(data = (t_dom, C2), fcn = fcn, prior = prior, p0 = p0)
print(fit)
mean_c, std_c = fit.p['m'].mean, fit.p['m'].sdev
Z0_mu, Z0_std = fit.p['Z0'].mean, fit.p['Z0'].sdev
Z1_mu, Z1_std = fit.p['Z1'].mean, fit.p['Z1'].sdev

fit_x, fit_lower, fit_upper = get_fit_band(fit.p, fcn, xlims = (dE_domain[0], dE_domain[-1]))
plot_fit(C2_mu, C2_sigma, fit_x, fit_lower, fit_upper, '$C_{2\mathrm{pt}}$', ylims = (0, 100), \
    saveat = twopt_parent + 'exc/' + ens_label + 'C2_twolevel.pdf')