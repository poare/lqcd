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
from formattools import *
style = styles['prd_twocol']

# Read the ensemble index. Can also manually set it to choose the ensemble to run on.
try:
    ens_idx = int(sys.argv[1])
    print('Plotting bare matrix elements for ensemble: ' + str(ensemble))
except:
    ens_idx = 0
    print("No input read. Defaulting to ensemble 0.")
# ens_idx = int(sys.argv[1])
ensemble = ['24I/ml_0p01', '24I/ml_0p005', '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
ens_path = ['24I/ml0p01', '24I/ml0p005', '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
f_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/bare_matrix_elements/' + ensemble + '/fit_params.h5'
n_ops = 5

# set specific ranges and labels
ytick_labels = [
    [['-5.8', '-5.4', '-5.0'], ['-1.0', '-0.9', '-0.8'], ['3.4', '3.6', '3.8'], ['-1.9', '-1.7', '-1.5'], ['2.1', '2.3', '2.5']],         # 24I/ml0p01
    [['-5.0', '-4.7', '-4.4'], ['-8.5', '-8.0', '-7.5'], ['1.70', '1.85', '2.00'], ['-1.55', '-1.45', '-1.35'], ['1.80', '1.95', '2.10']],      # 24I/ml0p005
    [['-1.90', '-1.75', '-1.60'], ['-2.9', '-2.7', '-2.5'], ['8.00', '8.75', '9.50'], ['-5.8', '-5.3', '-4.8'], ['6.2', '6.7', '7.2']],      # 32I/ml0p008
    [['-1.70', '-1.55', '-1.40'], ['-2.7', '-2.5', '-2.3'], ['5.7', '6.2', '6.7'], ['-5.5', '-5.0', '-4.5'], ['5.8', '6.2', '6.6']],      # 32I/ml0p006
    [['-1.55', '-1.45', '-1.35'], ['-2.40', '-2.25', '-2.10'], ['3.4', '3.8', '4.2'], ['-4.9', '-4.5', '-4.1'], ['5.2', '5.6', '6.0']],      # 32I/ml0p004
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
ytick_locs = [[round(float(ytick_labels[k][i]) * (10 ** pwr[k]), np.abs(pwr[k]) + 2) for i in range(len(ytick_labels[k]))] for k in range(n_ops)]
# yaxis_labels = ['$' + latex_labels[ii] + '^{\\mathrm{eff}} \\hspace\{-1.0mm\} \\times \\hspace{-0.5mm} 10^{' + str(pwr[ii]) + '}$' for ii in range(len(latex_labels))]

latex_labels = [r'O_1', r'O_2', r'O_3', r'O_{1^\prime}', r'O_{2^\prime}']
yaxis_labels = [r'$' + latex_labels[ii] + r'^{\mathrm{eff}} \hspace{-1.0mm} \times \hspace{-0.5mm} 10^{' + str(pwr[ii]) + r'}$' for ii in range(len(latex_labels))]

# read in input to plot
f = h5py.File(f_path, 'r')
data_slice = f['data_slice'][()]
c = f['c'][()]
sigmac = f['sigmac'][()]
plot_domain = f['plot_domain'][()]
f.close()

data_plot_mu = np.mean(data_slice, axis = 0)
data_plot_sigma = np.std(data_slice, axis = 0, ddof = 1)

def safe_arange(start, stop, step):
    """This is needed because np.arange returns floats with round-off error."""
    return step * np.arange(start / step, stop / step)

if ens_idx == 0:
    box_alpha = 0.5
    def safe_arange(start, stop, step):
        return step * np.arange(start / step, stop / step)
    with sns.plotting_context('paper'):
        asp_ratio = 5/3
        fig, ax = plt.subplots(figsize=[style['colwidth'], style['colwidth'] / asp_ratio])
        for i in range(n_ops):
            _, caps, _ = ax.errorbar(plot_domain, data_plot_mu[i], yerr = data_plot_sigma[i], fmt = markers[i], c = colors[i], \
                        label = '$' + latex_labels[i] + '$', elinewidth = style['ebar_width'], markersize = style['markersize'])
            ax.fill_between(plot_domain, c[i] - sigmac[i], c[i] + sigmac[i], color = colors[i], alpha = 0.2, linewidth = 0.0)
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
        ax.set_xlabel('$t / a$', fontsize = style['fontsize'] * 2/3)
        ax.set_ylabel('$\\langle \mathcal{O}_k^{\mathrm{eff.}} \\rangle$', fontsize = style['fontsize'] * 2/3)#, rotation = 0)
        ax.set_xlim((0, max(list(plot_domain)) // 2 + 1))
        ax.set_ylim((-0.16 / 8., 0.18 / 8.))
        #ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))    # y axis sci notation
        ax.set_xticklabels(np.arange(0, 31, 5), fontdict = {'fontsize' : style['fontsize'] * 2/3})
        yticks = safe_arange(-0.015, 0.016, 0.005)
        ytick_str = ['0']
        for ii in range(len(yticks)):
            ytick_str.append("{:.3f}".format(yticks[ii]))
        ax.set_yticklabels(ytick_str, fontdict = {'fontsize' : style['fontsize'] * 2/3})
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        ax.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', prop={'size': style['fontsize'] * 0.4})

        # create inset axes
        axins = inset_axes(ax, width="30%", height="30%", loc='upper left',
                           bbox_to_anchor=(0.06, -0.02, 2.3, 1.), bbox_transform=ax.transAxes)

        for spine in spinedirs:
            axins.spines[spine].set_alpha(box_alpha)
            axins.spines[spine].set_linewidth(style['axeswidth'])
        axins.tick_params(axis='x', color = [0, 0, 0, box_alpha])
        axins.tick_params(axis='y', color = [0, 0, 0, box_alpha])
        for i in range(n_ops):
            _, caps, _ = axins.errorbar(plot_domain, data_plot_mu[i], yerr = data_plot_sigma[i], fmt = '.', c = colors[i], \
                           label = '$' + latex_labels[i] + '$', capsize = style['endcaps'], \
                           elinewidth = style['ebar_width'], markersize = style['markersize'])
            axins.fill_between(plot_domain, c[i] - sigmac[i], c[i] + sigmac[i], color = colors[i], alpha = 0.35)
            for cap in caps:
                cap.set_markeredgewidth(style['ecap_width'])
        axins.set_xlim((9.5, 29.5))
        axins.set_ylim((-0.0475 / 8., -0.0445 / 8.))
        axins.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        axins.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        plt.xticks(visible = False)
        plt.yticks(visible = False)
        axins.set_facecolor('none')

        mark_inset(ax, axins, loc1 = 3, loc2 = 4, fc = 'none', ec='0.2', alpha = box_alpha)

        plt.draw()
        plt.tight_layout()
        inset_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/24Iml0p01_fits.pdf'
        plt.savefig(inset_path, bbox_inches='tight')
        print('Saved inset figure at ' + inset_path + '.')

# iterate over fit ranges and save figure
# efflabels = ['$\\langle ' + label + '^{\mathrm{eff. (0)}} \\rangle$' for label in latex_labels]
# efflabels = ['$\\langle ' + label + '^{\mathrm{eff.}} \\rangle$' for label in latex_labels]
efflabels = ['$' + label + '^{\mathrm{eff}}$' for label in latex_labels]
# yrange = [
#     [[-0.05, -0.038], [-0.085, -0.065], [0.0026, 0.0032], [-0.15, -0.118], [0.0165, 0.021]],
#     [[-0.041, -0.034], [-0.07, -0.058], [0.0013, 0.0017], [-0.125, -0.1070], [0.0144, 0.0171]],
#     [[-0.0155, -0.0120], [-0.024, -0.019], [0.000625, 0.00080], [-0.05, -0.036], [0.0046, 0.0062]],
#     [[-0.0136, -0.0112], [-0.0215, -0.0180], [0.00043, 0.00055], [-0.043, -0.034], [0.0045, 0.0055]],
#     [[-0.01225, -0.01025], [-0.0195, -0.0165], [0.00026, 0.00035], [-0.039, -0.032], [0.0041, 0.0048]]
# ][ens_idx]
asp_ratio = 4/3
fig_size = (style['colwidth'], style['colwidth'] / asp_ratio)
for i in range(n_ops):
    print('Plotting operator ' + str(op_labels[i]))
    # assert yrange[i][0] < yrange[i][1], 'Invalid yrange entered'
    with sns.plotting_context('paper'):
        plt.figure(figsize = fig_size)
        _, caps, _ = plt.errorbar(plot_domain, data_plot_mu[i], yerr = data_plot_sigma[i], fmt = markers[i], c = colors[i], \
                     label = efflabels[i], capsize = style['endcaps'], markersize = style['markersize'], \
                     elinewidth = style['ebar_width'])
        plt.fill_between(plot_domain, c[i] - sigmac[i], c[i] + sigmac[i], color = colors[i], alpha = 0.3, linewidth = 0.0)
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        plt.xlabel('$t / a$', fontsize = style['fontsize'])
        # plt.ylabel(efflabels[i], fontsize = style['fontsize'])
        plt.ylabel(yaxis_labels[i], fontsize = style['fontsize'])
        ax = plt.gca()
        ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        ax.set_yticks(ytick_locs[i])
        ax.set_yticklabels(ytick_labels[i])
        for spine in spinedirs:
            ax.spines[spine].set_linewidth(style['axeswidth'])
        plt.xticks(fontsize = style['fontsize'])
        plt.yticks(fontsize = style['fontsize'])
        # plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))    # y axis sci notation
        plt.xlim(0, max(list(plot_domain)) // 2 + 1.5)
        # plt.ylim(yrange[i][0] / 8., yrange[i][1] / 8.)
        plt.ylim(yrangep[i][0], yrangep[i][1])
        plt.tight_layout()
        f2path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/bare_eff_mat_elems/' + ens_path + '/' + op_labels[i] + '.pdf'
        plt.savefig(f2path, bbox_inches='tight')
        print('Operator ' + str(op_labels[i]) + ' saved at: \n   ' + f2path)

# Make stacked image for 24I/ml0p01 as well
if ens_idx == 0:
    from matplotlib import gridspec
    from matplotlib import rc
    rc('text', usetex = True)
    # ytick_labels = [['-5.8', '-5.4', '-5.0'], ['-1.0', '-0.9', '-0.8'], ['3.4', '3.6', '3.8'], ['-1.9', '-1.7', '-1.5'], ['2.1', '2.3', '2.5']]
    # pwr = [-3, -2, -4, -2, -3]              # think about doing using \\text{-}
    # ytick_locs = [[round(float(ytick_labels[k][i]) * (10 ** pwr[k]), np.abs(pwr[k]) + 2) for i in range(len(ytick_labels[k]))] for k in range(n_ops)]
    # yaxis_labels = ['$' + latex_labels[ii] + '^{\mathrm{eff}}\\hspace{-1.0mm}\\times \\hspace{-0.5mm} 10^{' + str(pwr[ii]) + '}$' for ii in range(len(latex_labels))]
    print('Generating stacked image.')
    # yrangep = [[-0.0060, -0.0048], [-0.0105, -0.0075], [0.00033, 0.00039], [-0.02, -0.014], [0.0020, 0.0026]]
    reindex = [4, 2, 0, 1, 3]           # reindex to stack them in order of value
    asp_ratio = 3.
    fig = plt.figure(figsize = (style['colwidth'], style['colwidth'] / asp_ratio * n_ops))
    gs = gridspec.GridSpec(n_ops, 1)
    all_axes = []
    for j in range(n_ops):
        i = reindex[j]
        if j == 0:
            axi = plt.subplot(gs[j])
        else:
            axi = plt.subplot(gs[j], sharex = all_axes[0])
        _, caps, _ = axi.errorbar(plot_domain, data_plot_mu[i], yerr = data_plot_sigma[i], fmt = markers[i], c = colors[i],\
                             label = efflabels[i], capsize = style['endcaps'], markersize = style['markersize'], \
                             elinewidth = style['ebar_width'])
        axi.fill_between(plot_domain, c[i] - sigmac[i], c[i] + sigmac[i], color = colors[i], alpha = 0.3, linewidth = 0.0)
        for cap in caps:
            cap.set_markeredgewidth(style['ecap_width'])
        for spine in spinedirs:
            axi.spines[spine].set_linewidth(style['axeswidth'])
        axi.set_ylim((yrangep[i][0], yrangep[i][1]))
        axi.set_ylabel(yaxis_labels[i], fontsize = style['fontsize'])
        axi.set_xticks([0, 10, 20, 30])
        if i < 4:
            axi.xaxis.set_tick_params(width = style['tickwidth'], length = 2 * style['ticklength'], bottom=True, top=True, \
                                        labelbottom=True, labeltop=False, direction = 'inout')
        else:
            axi.xaxis.set_tick_params(width = style['tickwidth'], length = 2 * style['ticklength'], bottom=True, top=False, \
                                        labelbottom=True, labeltop=False, direction = 'inout')
        axi.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'])
        axi.set_yticks(ytick_locs[i])
        axi.set_yticklabels(ytick_labels[i])
        all_axes.append(axi)
    all_axes[4].set_xlim((0, max(list(plot_domain)) // 2 + 0.5))
    all_axes[4].set_xlabel('$t / a$', fontsize = style['fontsize'])
    for i in range(n_ops - 1):
        plt.setp(all_axes[i].get_xticklabels(), visible = False)

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace = 0.0)
    stack_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/paper/plots/24Iml0p01_fits_stacked.pdf'
    plt.savefig(stack_path, bbox_inches = 'tight')
    print('Stacked image saved at: ' + stack_path)

print('Done plotting on ' + ensemble)
