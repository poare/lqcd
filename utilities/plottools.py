################################################################################
# This script saves a list of all common plotting functions that I may need to #
# use in my python code. To import the script, simply add the path with sys    #
# before importing. For example, if lqcd/ is in /Users/theoares:               #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from plottools import *                                                      #
#                                                                              #
# Author: Patrick Oare                                                         #
################################################################################

from __main__ import *

import numpy as np
import seaborn as sns
from formattools import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.font_manager as font_manager

def set_font():
    """
    Sets the font used by matplotlib to CMU Serif.
    """
    mpl.rcParams['font.family']='serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif']=cmfont.get_name()
    mpl.rcParams['mathtext.fontset']='cm'
    mpl.rcParams['axes.unicode_minus']=False
    mpl.rcParams['axes.formatter.use_mathtext'] = True
set_font()

################################################################################
############################ REAL-VALUED FUNCTIONS #############################
################################################################################

def plot_1d_function(xvals, dat, ax = None, col = ['r', 'c'], ax_label = ['x', 'y'], title = None, fn_label = 'f', legend = False, \
                    style = default_style, fig_size = None, tight_layout = True, saveat_path = None):
    """
    Plots a function of one real variable, f, which can either be real or complex-valued. Either creates a new axis if 
    one is not passed in, or uses an existing axis.

    Parameters
    ----------
    xvals : np.array[np.float64]
        X values to plot at function at.
    dat : np.array[np.float64 or np.complex64]
        Data to plot, should the same dimensions as xvals. Can either be np.float or np.complex data types.
    ax : matplotlib.Axes (default = None)
        Axes to use for the figure. If None, creates a new axis.
    col : [string] or [string, string]
        Colors to plot the functions with. If f is a real function, then use the first color given in the list or just 
        pass a single color. If f is complex, then pass in a pair of colors [c1, c2].
    ax_label : [string, string]
        Labels for the x and y axes.
    title : string (default = None)
        Title of the plot. If None, no title is generated.
    fn_label : string
        Label for the function in the plot legend.
    legend : bool (default = False)
        True if the legend should be shown, False otherwise.
    style : dict (default = default_style)
        Element of the formattools.styles dictionary, which specifies font sizes and other hyperparameters. 
    fig_size : float (default = None)
        Figure size to generate. If None, defaults to (style['col_width], style['col_width'] / style['asp_ratio'])
    tight_layout : bool (default = True)
        True if set the layout to tight. 
    saveat_path : string (default = None)
        Path to save figure at. If None, does not save the figure. 

    Returns
    -------
    plt.figure
        Figure which is plotted on.
    plt.Axes
        Axes which are plotted on.
    """
    with sns.plotting_context('talk'):
        if not fig_size:
            fig_size = (style['colwidth'], style['colwidth'] / style['asp_ratio'])
        if ax is None:
            fig, ax = plt.subplots(1, figsize = fig_size)
        else:
            fig = ax.get_figure()
        if np.iscomplexobj(dat):
            ax.plot(xvals, np.real(dat), color = col[0], label = 'Re[' + fn_label + ']')
            ax.plot(xvals, np.imag(dat), color = col[1], label = 'Im[' + fn_label + ']')
        else:
            color = col[0] if type(col) is list else col
            ax.plot(xvals, dat, color = color, label = fn_label)
        if title:
            ax.set_title(title, fontsize = style['fontsize'])
        ax.set_xlabel(ax_label[0], fontsize = style['fontsize'])
        ax.set_ylabel(ax_label[1], fontsize = style['fontsize'])
        
        # TODO later, add option for setting tick positions + labels
        stylize_axis(ax, style)

        fig.subplots_adjust(
            bottom = style['bottom_pad'], top = style['top_pad'], left = style['left_pad'], right = style['right_pad']
        )
        if legend:
            ax.legend()
        if tight_layout:
            plt.tight_layout()
        if saveat_path:
            plt.savefig(saveat_path, bbox_inches = 'tight')
    return fig, ax

# TODO change other functions so they create fig, ax inside

def plot_1d_data(ax, xvals, cvs, sigmas, colors = ['r', 'c'], title = 'Plot of function', ax_label = ['x', 'y'], \
                    legend = True, style = default_style):
    """
    Plots a function of one real variable, f, which can either be real or complex-valued.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to use for the figure.
    xvals : np.array[np.float64]
        X values to plot at function at.
    cvs : np.array[np.float64 or np.complex64]
        Central values of data to plot, should the same dimensions as xvals. Can be np.float or np.complex data types.
    sigmas : np.array[np.float64 or np.complex64]
        Standard deviations of data to plot.
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
    style : dict
        Element of the formattools.styles dictionary, which specifies font sizes and other hyperparameters. 

    Returns
    -------
    """
    if np.iscomplexobj(cvs):
        _, caps1, _ = ax.errorbar(xvals, np.real(cvs), yerr = np.real(sigmas), color = colors[0], \
            label = 'Re[' + fn_label + ']', capsize = style['endcaps'], markersize = style['markersize'], \
            elinewidth = style['ebar_width'])
        _, caps2, _ = ax.errorbar(xvals, np.imag(cvs), yerr = np.imag(sigmas), color = colors[1], \
            label = 'Im[' + fn_label + ']', capsize = style['endcaps'], markersize = style['markersize'], \
            elinewidth = style['ebar_width'])
        for cap in caps1:
            cap.set_markeredgewidth(style['ecap_width'])
        for cap in caps2:
            cap.set_markeredgewidth(style['ecap_width'])
    else:
        ax.errorbar(xvals, cvs, yerr = sigmas, color = colors[0], label = fn_label)
    ax.set_title(title, fontsize = t_size)
    ax.set_xlabel(ax_label[0], fontsize = style['fontsize'])
    ax.set_ylabel(ax_label[1], fontsize = style['fontsize'])
    if legend:
        ax.legend()
    return ax

def add_line(ax, val, orientation = 'v', **kwargs):
    """
    Adds a line to a 2-dimensional plot.

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
        ax.axvline(val, **kwargs)
    elif orientation == 'h':
        ax.axhline(val, **kwargs)
    else:
        raise Exception('Orientation must be h or v')
    return ax

def plot_stacked_fig(tmp, vert = True, show_join_axis = False):
    """
    Plots a stacked 2D figure, joined along either the horizontal or vertical axis. 

    Parameters
    ----------
    TODO other params
    vert : bool (default: True)
        True if we want to stack vertically, False if horizontally.
    show_join_axis: bool (default : False)
        False if we want to show labels on the interior axes.
    """
    return

################################################################################
########################### COMPLEX-VALUED FUNCTIONS ###########################
################################################################################

def plot_data_CR_3D(redom, imdom, dat, ax = None, colmap = 'hot', c = None, title = 'Plot of function', ax_label = ['\nRe[z]', '\nIm[z]'], \
                fn_label = 'f', zbounds = (0, 1), fig_size = None, style = default_style, **kwargs):
    """
    Plots a real valued complex function f : C --> R as a surface in R^3. To add the colorbar, use plt.colorbar(graph)
    after calling this function.

    Parameters
    ----------
    redom : np.array[np.float64]
        Real part of domain values (subset of the complex plane) to plot at.
    imdom : np.array[np.float64]
        Imaginary part of domain values (subset of the complex plane) to plot at.
    dat : np.array[np.float64], or function handle
        Array of data (of same size as zdomain). Can also be a function handle which will be evaluated before plotting.
    ax : mpl_toolkits.mplot3d.Axes3D (default = None)
        3D Axes to use for the figure. If None, generates Axes and returns in function.
    col : string
        Color scheme to use. Defaults to 'Spectral'.
    title : string
        Title of plot to show.
    ax_label : [string, string]
        Axes labels to show. First value is real axis, second value is imaginary axis

    Returns
    -------
    plt.figure
        Figure which is plotted on.
    mpl_toolkits.mplot3d.Axes3D
        Axes object the plot is shown on.
    graph
        Surface plot.
    """
    with sns.plotting_context('talk'):
        if ax is None:
            if fig_size is None:
                fig_size = (style['colwidth'], style['colwidth'] / style['asp_ratio'])
            fig = plt.figure(figsize = fig_size)
            ax = Axes3D(fig)
        else:
            fig = ax.get_figure()
        XX, YY = np.meshgrid(redom, imdom)
        # if callable(dat):
        #     zdom = np.array([[redom[i] + (1j) * imdom[j] for j in range(len(redom))] for i in range(len(imdom))], dtype = np.complex64)
        #     dat = np.array(list(map(dat, zdom)))
        if c is None:
            graph = ax.plot_surface(XX, YY, dat, cmap = colmap, **kwargs)
        else:
            graph = ax.plot_surface(XX, YY, dat, color = c, **kwargs)
        ax.set_title(title)
        ax.set_xlabel(ax_label[0])
        ax.set_ylabel(ax_label[1])
        ax.set_zlabel('\n' + fn_label + '(z)')
        ax.set_zlim(zbounds)
    return fig, ax, graph

def plot_data_CR_2D(redom, imdom, dat, ax = None, colmap = 'viridis', ax_label = ['\nRe[z]', '\nIm[z]'], \
                fn_label = 'f', fig_size = None, fname = r'f', use_title = True, style = styles['notebook']):
    """
    Plots a complex-valued data f : C --> C as a heat map in R^2. To add the colorbar, use plt.colorbar(graph)
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
    with sns.plotting_context('talk'):
        if not fig_size:
            fig_size = (style['colwidth'], style['colwidth'] / style['asp_ratio'])
        if ax is None:
            fig, ax = plt.subplots(1, figsize = fig_size)
        else:
            fig = ax.get_figure()
        
        X, Y = np.meshgrid(redom, imdom)
        Z = X + 1j * Y
        image = func(Z)
        divider1 = make_axes_locatable(ax)
        cax = divider1.append_axes('right', size='5%', pad=0.15)
        graph = ax.pcolormesh(X, Y, dat, cmap = colmap)
        # im = ax.imshow(image, cmap = colmap, vmin = np.min(image), vmax = np.max(image), \
        #     extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin = 'lower')
        ax.set_xlabel(r'$\mathrm{Re}[z]$')
        ax.set_ylabel(r'$\mathrm{Im}[z]$')
        if use_title:
            ax.set_title(r'$' + fname + r'(z)$')
        cbar = fig.colorbar(graph, cax, orientation='vertical')

        stylize_axis(ax, style)
        stylize_colorbar(cbar, style)
        return fig, ax

def plot_data_CC_2D(ax, redom, imdom, dat, col = 'hot', title = 'Plot of function', ax_label = ['\nRe[z]', '\nIm[z]'], \
                fn_label = 'f', **kwargs):
    """
    Plots a complex-valued data f : C --> C as a heat map in R^2. To add the colorbar, use plt.colorbar(graph)
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
    # if callable(dat):
    #     zdom = np.array([[redom[i] + (1j) * imdom[j] for j in range(len(redom))] for i in range(len(imdom))], dtype = np.complex64)
    #     dat = np.array(list(map(dat, zdom)))
    graph = ax.pcolormesh(XX, YY, dat, cmap = col, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(ax_label[0])
    ax.set_ylabel(ax_label[1])
    return ax, graph

def plot_fn_CR_2D(func, ax = None,  fig = None, x_bounds = [-2, 2], y_bounds = [-2, 2], nx = 50, ny = 50, \
    colmap = 'viridis', fig_size = None, fname = r'f', use_title = True, style = styles['notebook']):
    """
    Plots a real-valued complex function in the 2D plane with a heatmap. 

    Parameters
    ----------
    func : np.complex64 -> np.float
        Real-valued complex function to plot. 
    axes : matplotlib.Axes (None)
        Axes to plot the result on. If None, creates a new Axes object.
    fig : matplotlib.Figure (None)
        Figure to plot on. If None, gets figure with ax.get_figure().
    x_bounds : [xmin, xmax] (default = [-2, 2])
        Bounds for the x-axis.
    y_bounds : [ymin, ymax] (default = [-2, 2])
        Bounds for the y-axis.
    nx : int (default = 50)
        Number of points in the x-direction.
    ny : int (default = 50)
        Number of points in the y-direction.
    colmap : string (default = viridis)
        Color maps to use. First component is for the real part, second is for the imaginary part. 
    fname : string (default = r'f(z)')
        Function handle to display.
    use_title : bool (default = True)
        Whether or not to render the plot with a title.
    fig_size : [x_size, y_size] (default = None)
        Figure size to use. If None, defaults to using the size specified by style.
    style : dict (default = formattools.styles['notebook'])
        Style to use. 
    
    Returns
    -------
    matplotlib.Figure
        Figure the plot is rendered with.
    matplotlib.Axes
        Axes for the plot.
    """
    with sns.plotting_context('talk'):
        if not fig_size:
            fig_size = (style['colwidth'], style['colwidth'] / style['asp_ratio'])
        if ax is None:
            fig, ax = plt.subplots(1, figsize = fig_size)
        else:
            fig = ax.get_figure()
        
        X, Y = np.meshgrid(
            np.linspace(x_bounds[0], x_bounds[1], nx),
            np.linspace(y_bounds[0], y_bounds[1], ny)
        )
        Z = X + 1j * Y

        graph = ax.pcolormesh(XX, YY, dat, cmap = col, **kwargs)

        image = func(Z)
        divider1 = make_axes_locatable(ax)
        cax = divider1.append_axes('right', size='5%', pad=0.15)
        im = ax.imshow(image, cmap = colmap, vmin = np.min(image), vmax = np.max(image), \
            extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin = 'lower')
        ax.set_xlabel(r'$\mathrm{Re}[z]$')
        ax.set_ylabel(r'$\mathrm{Im}[z]$')
        if use_title:
            ax.set_title(r'$' + fname + r'(z)$')
        cbar = fig.colorbar(im, cax, orientation='vertical')

        stylize_axis(ax, style)
        stylize_colorbar(cbar, style)
        return fig, ax

def plot_fn_CC_2D(func, axes = None,  fig = None, x_bounds = [-2, 2], y_bounds = [-2, 2], nx = 50, ny = 50, \
    colmap = ['inferno', 'viridis'], fname = r'f(z)', use_title = True, fig_size = None, style = styles['notebook']):
    """
    Plots a complex function on two 2D panels: one for the real part, one for the imaginary part. 

    Parameters
    ----------
    func : np.complex64 -> np.complex64
        Complex-valued function to plot. 
    axes : [matplotlib.Axes, matplotlib.Axes] (default = None)
        Axes to plot the result on. ax[0] (ax[1]) is used for the real (imaginary) part. If None, creates new Axes. 
    fig : matplotlib.Figure (default = None)
        Figure to plot on. If None, gets figure with ax.get_figure().
    x_bounds : [xmin, xmax] (default = [-2, 2])
        Bounds for the x-axis.
    y_bounds : [ymin, ymax] (default = [-2, 2])
        Bounds for the y-axis.
    nx : int (default = 50)
        Number of points in the x-direction.
    ny : int (default = 50)
        Number of points in the y-direction.
    colmap : [string, string] (default = ['inferno', 'viridis'])
        Color maps to use. First component is for the real part, second is for the imaginary part. 
    fname : string (default = r'f(z)')
        Function handle to display.
    use_title : bool (default = True)
        Whether or not to render the plot with a title.
    fig_size : [x_size, y_size] (default = None)
        Figure size to use. If None, defaults to using the size specified by style.
    style : dict (default = formattools.styles['notebook'])
        Style to use. 
    
    Returns
    -------
    matplotlib.Figure
        Figure the plot is rendered with.
    [matplotlib.Axes, matplotlib.Axes]
        Axes of the real and imaginary part of the plot.
    """
    with sns.plotting_context('talk'):
        if not fig_size:
            fig_size = (style['colwidth'], style['colwidth'] / style['asp_ratio'])
        if axes is None:
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = fig_size)
        else:
            fig = axes.get_figure()
        ax1, ax2 = axes
        
        X, Y = np.meshgrid(
            np.linspace(x_bounds[0], x_bounds[1], nx),
            np.linspace(y_bounds[0], y_bounds[1], ny)
        )
        Z = X + 1j * Y
        image = func(Z)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.15)
        im1 = ax1.imshow(np.real(image), cmap = colmap[0], vmin = np.min(np.real(image)), vmax = np.max(np.real(image)), \
            extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin = 'lower')
        ax1.set_xlabel(r'$\mathrm{Re}[z]$')
        ax1.set_ylabel(r'$\mathrm{Im}[z]$')
        if use_title:
            ax1.set_title(r'$\mathrm{Re}[' + fname + r']$')
        cbar1 = fig.colorbar(im1, cax1, orientation='vertical')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.15)
        im2 = ax2.imshow(np.imag(image), cmap = colmap[1], vmin = np.min(np.imag(image)), vmax = np.max(np.imag(image)), \
            extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], origin = 'lower')
        ax2.set_xlabel(r'$\mathrm{Re}[z]$')
        ax2.set_ylabel(r'$\mathrm{Im}[z]$')
        if use_title:
            ax2.set_title(r'$\mathrm{Im}[' + fname + r']$')
        cbar2 = fig.colorbar(im2, cax2, orientation='vertical')

        stylize_axis(ax1, style)
        stylize_colorbar(cbar1, style)
        stylize_axis(ax2, style)
        stylize_colorbar(cbar2, style)

        plt.tight_layout()
    return fig, [ax1, ax2]

def mark_point_2D(axes, x, y, col = 'r', size = 100, mkr = 'x', lw = 3):
    """
    Marks a point on a 2D plane. 

    Parameters
    ----------
    axes : matplotlib.Axes
        Axes to plot on.
    x : float
        x-position of point to mark.
    y : float
        y-position of point to mark.
    col : string (default = 'r')
        Color of marker to use.
    size : int (default = 100)
        Size of point to plot.
    mkr : string (default = 'x')
        Type of marker to plot.
    lw : int (default = 3)
        Width of marker to use. 
    
    Returns
    -------
    matplotlib.Axes
        Axes object which has been modified.
    """
    if type(axes) != list:
        axes.scatter(x, y, s = size, marker = mkr, c = col, linewidth = lw)
    else:
        for ax in axes:
            ax.scatter(x, y, s = size, marker = mkr, c = col, linewidth = lw)
    return axes

def mark_points_2D(axes, xlist, ylist, col = 'r', size = 100, mkr = 'x', lw = 3):
    """
    Marks multiple point on a 2D plane. 

    Parameters
    ----------
    axes : matplotlib.Axes
        Axes to plot on.
    xlist : [float]
        x-positions to mark.
    ylist : [float]
        y-positions to mark.
    col : string (default = 'r')
        Color of marker to use.
    size : int (default = 100)
        Size of point to plot.
    mkr : string (default = 'x')
        Type of marker to plot.
    lw : int (default = 3)
        Width of marker to use. 
    
    Returns
    -------
    matplotlib.Axes
        Axes object which has been modified.
    """
    for ii in range(len(xlist)):
        mark_point_2D(axes, xlist[ii], ylist[ii], col = col, size = size, mkr = mkr, lw = lw)
    return axes

def add_points_3D(ax, redom, imdom, dat, col = 'r', size = 10, lw = 8):
    for ii in range(len(dat)):
        add_point_3D(ax, redom[ii], imdom[ii], dat[ii], c = col)    

################################################################################
################################### 3D PLOTS ###################################
################################################################################

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

def add_colorbar(fig, graphic, style = default_style, ticks = None, tick_labels = None):
    if ticks is None:
        cbar = fig.colorbar(graphic)
    else:
        cbar = fig.colorbar(graphic, ticks = ticks)
        if tick_labels is not None:
            cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize = style['fontsize'])
    return fig

################################################################################
################################## UTILITIES ###################################
################################################################################

def stylize_axis(ax, style = default_style):
    """
    Formats a given axis in an existing style by giving it the appropriate tick and label sizes. 

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to modify style of.
    style : dict (default = default_style)
        Style to use for the axis.
    """
    ax.xaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'])
    ax.yaxis.set_tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'])
    for spine in spinedirs:
        ax.spines[spine].set_linewidth(style['axeswidth'])
    return ax

def stylize_colorbar(cbar, style = default_style):
    """
    Formats a given colorbar in an existing style by giving it the appropriate tick and label sizes. 

    Parameters
    ----------
    cbar : matplotlib.Axes
        Axes to modify style of.
    style : dict (default = default_style)
        Style to use for the axis.
    """
    cbar.ax.tick_params(width = style['tickwidth'], length = style['ticklength'], labelsize = style['fontsize'])
    cbar.outline.set_linewidth(style['axeswidth'])
    for spine in spinedirs:
        cbar.ax.spines[spine].set_linewidth(style['axeswidth'])
    return cbar
