"""
Visualization routines.
This module should contain routines for making figures, mostly in the form of wrappers for seaborn
and matplotlib to play nicely with standard objects encountered in physic analyses (e.g., gvars and
DataFrames).
Routines involving physics or analysis logic typically belong in the module figures.py.
"""
import gvar as gv
import numpy as np
import pylab as plt
import seaborn as sns


def color_palette(*args, **kwargs):
    """ Wrapper for seaborn.color_palette """
    return sns.color_palette(*args, **kwargs)


def subplots(*args, **kwargs):
    """
    Wraps pylab.subplots(...)
    TODO: fix imports
    Backstory:
    pylab is supposed to be deprecated (?) due to its state-machine environment
    Related infor[https://matplotlib.org/faq/usage_faq.html]
    """
    return plt.subplots(*args, **kwargs)


def plot(y, ax=None, **kwargs):
    """Plot y on axis ax."""
    if ax is None:
        _, ax = plt.subplots(1)
    x = range(len(y))
    errorbar(ax, x, y, **kwargs)
    return ax


def errorbar(ax, x, y, bands=False, **kwargs):
    """Wrapper to plot gvars using the matplotlib function errorbar."""
    if hasattr(y, 'values'):
        y = y.values
    if hasattr(x, 'values'):
        x = x.values
    try:
        xerr = gv.sdev(x)
    # sdev throws error w/ x = np.arange(>50)
    except ZeroDivisionError:
        xerr = 0.*np.array(x)  # in case x passed from plot().
    x = gv.mean(x)
    try:
        yerr = gv.sdev(y)
    except ZeroDivisionError:
        yerr = 0.*y
    y = gv.mean(y)
    if bands:
        ax.errorbar(x=x, y=y, **kwargs)
        facecolor = kwargs.get('color', ax.lines[-1].get_color())
        alpha = kwargs.get('alpha', 1.0)
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            facecolor=facecolor,
            alpha=alpha)
    else:
        ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, **kwargs)
    return ax


def mirror(y, x=None, ax=None, label=None, color=None):
    """
    Makes a "mirror" plot, where negative values are mirrored to be positive.
    Positive (negative) values appear with circles (squares) for markers.
    """
    if ax is None:
        _, ax = plt.subplots(1)
    if x is None:
        x = np.arange(len(y))
    if len(x) != len(y):
        raise ValueError(
            f"Size mismatch between x and y: len(x)={len(x)}, len(y)={len(y)}")
    neg = y < 0
    pos = ~neg
    errorbar(ax, x[pos], y[pos], marker='o', fmt='.', color=color, label=label)
    color = ax.lines[-1].get_color()  # match color
    errorbar(ax, x[neg], -y[neg], marker='s', fmt='.', color=color,
             markerfacecolor='none', markeredgewidth=2)
    if neg.any():
        errorbar(ax, x[neg], -y[neg], marker='s', fmt='.', color=color,
                 markerfacecolor='none', markeredgewidth=2)
    errorbar(ax, x, np.sign(y)*y, color=color)
    return ax

def noise_to_signal(ax, y, x=None, label=None, color=None):
    """ Plots the noise-to-signal ratio as a percentage. """
    y = 100 * gv.sdev(y) / gv.mean(y)
    mirror(ax=ax, y=y, x=x, label=label, color=color)
    ax.set_ylabel("n/s (%)")
    return ax


def axhline(ax, y, alpha=None, **kwargs):
    """Wrapper to plot gvars using matplotlib function axhline."""
    if alpha is None:
        alpha = 0.25
    mean = gv.mean(y)
    ax.axhline(mean, **kwargs)
    color = kwargs.get('color', 'k')
    xmin = kwargs.get('xmin', 0.0)
    xmax = kwargs.get('xmax', 1.0)
    axhspan(ax, y, alpha=alpha, color=color, xmin=xmin, xmax=xmax)
    return ax


def axhspan(ax, y, **kwargs):
    """Wrapper to plot gvars using matplotlib function axhspan."""
    mean = gv.mean(y)
    err = gv.sdev(y)
    ax.axhspan(mean - err, mean + err, **kwargs)


def axvline(ax, x, alpha=None, **kwargs):
    """Wrapper to plot gvars using matplotlib function axvline."""
    if alpha is None:
        alpha = 0.25
    mean = gv.mean(x)
    ax.axvline(mean, **kwargs)
    color = kwargs.get('color', 'k')
    axvspan(ax, x, alpha=alpha, color=color)


def axvspan(ax, x, **kwargs):
    """Wrapper to plot gvars using matplotlib function axvspan."""
    mean = gv.mean(x)
    err = gv.sdev(x)
    ax.axvspan(mean - err, mean + err, **kwargs)