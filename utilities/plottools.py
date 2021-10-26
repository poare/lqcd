
################################################################################
# This script saves a list of all common plotting functions that I may need to #
# use in my python code. To import the script, simply add the path with sys    #
# before importing. For example, if lqcd/ is in /Users/theoares:               #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from plottools import *                                                      #
################################################################################

from __main__ import *

# use CMU Serif
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
# cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
cmfont = font_manager.FontProperties('/Users/theoares/Library/Fonts/cmunrm.otf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Ellipse
from matplotlib import cm
import seaborn as sns

def plot_1d_function(ax, xvals, dat, colors = ['r', 'c'], title = 'Plot of function', ax_label = ['x', 'y'], \
                    fn_label = 'f', legend = True, t_size = 16, ax_size = 24):
    """
    Plots a function of one real variable, f, which can either be real or complex-valued.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to use for the figure.
    xvals : np.array[np.float64]
        X values to plot at function at.
    dat : np.array[np.float64 or np.complex64]
        Data to plot, should the same dimensions as xvals. Can either be np.float or np.complex data types.
    colors : [string] or [string, string]
        Colors to plot the functions with. If f is a real function, then use the first color given in the list. If f is
        complex, then pass in a pair of colors [c1, c2].
    title : string
        Title of the plot.
    ax_label : [string, string]
        Labels for the x and y axes.
    fn_label : string
        Label for the function in the plot legend.
    legend : bool
        True if the legend should be shown, False otherwise.
    t_size : int
        Font size for title.
    ax_size : int
        Font size for axis labels.

    Returns
    -------
    """
    if np.iscomplexobj(dat):
        ax.plot(xvals, np.real(dat), color = colors[0], label = 'Re[' + fn_label + ']')
        ax.plot(xvals, np.imag(dat), color = colors[1], label = 'Im[' + fn_label + ']')
    else:
        ax.plot(xvals, dat, color = colors[0], label = fn_label)
    ax.set_title(title, fontsize = t_size)
    ax.set_xlabel(ax_label[0], fontsize = ax_size)
    ax.set_ylabel(ax_label[1], fontsize = ax_size)
    if legend:
        ax.legend()
    return ax

def add_line(ax, val, orientation = 'v', **kwargs):
    """
    Adds a line to a plot of a 1 variable function.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to use for the figure.
    val : float
        Value to add the line at
    orientation : 'h' or 'v'
        'h' for horizontal line, 'v' for vertical line
    style :

    Returns
    -------
    matplotlib.Axes
        Axes which the line is added to
    """
    if orientation == 'v':
        lims = ax.get_ylim()
        ax.axvline(val, lims[0], lims[1], **kwargs)
    elif orientation == 'h':
        lims = ax.get_xlim()
        ax.axhline(val, lims[0], lims[1], **kwargs)
    else:
        raise Exception('Orientation must be h or v')
    return ax

def plot_complex3D(ax, redom, imdom, dat, colmap = 'hot', c = '', title = 'Plot of function', ax_label = ['\nRe[z]', '\nIm[z]'], \
                fn_label = 'f', zbounds = (0, 1), **kwargs):
    """
    Plots a real valued complex function f : C --> R as a surface in R^3. To add the colorbar, use plt.colorbar(graph)
    after calling this function

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
        3D Axes to use for the figure.
    redom : np.array[np.float64]
        Real part of domain values (subset of the complex plane) to plot at.
    imdom : np.array[np.float64]
        Imaginary part of domain values (subset of the complex plane) to plot at.
    dat : np.array[np.float64], or function handle
        Array of data (of same size as zdomain). Can also be a function handle which will be evaluated before plotting.
    col : string
        Color scheme to use. Defaults to 'Spectral'.
    title : string
        Title of plot to show.
    ax_label : [string, string]
        Axes labels to show. First value is real axis, second value is imaginary axis

    Returns
    -------
    mpl_toolkits.mplot3d.Axes3D
        Axes object the plot is shown on.
    graph
        Surface plot.
    """
    XX, YY = np.meshgrid(redom, imdom)
    if callable(dat):
        zdom = np.array([[redom[i] + (1j) * imdom[j] for j in range(len(redom))] for i in range(len(imdom))], dtype = np.complex64)
        dat = np.array(list(map(dat, zdom)))
    if c == '':
        graph = ax.plot_surface(XX, YY, dat, cmap = colmap, **kwargs)
    else:
        graph = ax.plot_surface(XX, YY, dat, color = c, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(ax_label[0])
    ax.set_ylabel(ax_label[1])
    ax.set_zlabel('\n' + fn_label + '(z)')
    ax.set_zlim(zbounds)
    return ax, graph

def plot_complex2D(ax, redom, imdom, dat, col = 'hot', title = 'Plot of function', ax_label = ['\nRe[z]', '\nIm[z]'], \
                fn_label = 'f', **kwargs):
    """
    Plots a real valued complex function f : C --> R as a heat map in R^2. To add the colorbar, use plt.colorbar(graph)
    after calling this function.

    Parameters
    ----------
    ax : matplotlib.Axes
        2D Axes to use for the figure.
    redom : np.array[np.float64]
        Real part of domain values (subset of the complex plane) to plot at.
    imdom : np.array[np.float64]
        Imaginary part of domain values (subset of the complex plane) to plot at.
    dat : np.array[np.float64], or function handle
        Array of data (of same size as zdomain). Can also be a function handle which will be evaluated before plotting.
    col : string
        Color scheme to use. Defaults to 'Spectral'.
    title : string
        Title of plot to show.
    ax_label : [string, string]
        Axes labels to show. First value is real axis, second value is imaginary axis

    Returns
    -------
    matplotlib.Axes
        Axes object the plot is shown on.
    graph
        Surface plot.
    """
    XX, YY = np.meshgrid(redom, imdom)
    if callable(dat):
        zdom = np.array([[redom[i] + (1j) * imdom[j] for j in range(len(redom))] for i in range(len(imdom))], dtype = np.complex64)
        dat = np.array(list(map(dat, zdom)))
    graph = ax.pcolormesh(XX, YY, dat, cmap = col, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(ax_label[0])
    ax.set_ylabel(ax_label[1])
    return ax, graph

def add_points(ax, redom, imdom, dat, col = 'r'):
    if type(ax) == Axes3D:
        # ax.plot(redom, imdom, dat, marker = 'x', markersize = 5, c = col)    # this has a problem rendering
        for ii in range(len(dat)):
            add_point_3D(ax, redom[ii], imdom[ii], dat[ii], c = col)
    else: # then it's a 2D axes object
        ax.scatter(redom, imdom, dat, marker = 'x', c = col, linewidth = 8)

def add_point_3D(ax, x, y, z, c = 'r', radius = 0.05):
    xy_len, z_len = ax.get_figure().get_size_inches()
    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                      'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                      'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width = radius, height = radius * ratio, fc = c, ec = c)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)
    return ax
