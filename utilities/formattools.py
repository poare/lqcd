################################################################################
# This script saves a list of common formatting functions for general data     #
# processing that I may need to use in my python code. To import the script,   #
# simply add the path with sys before importing. For example, if lqcd/ is in   #
# /Users/theoares:                                                             #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from formattools import *                                                    #
#                                                                              #
# Author: Patrick Oare                                                         #
################################################################################

from __main__ import *

import numpy as np
import h5py
import os
from scipy.special import zeta
import time
import re
import itertools
import io
import random
from scipy import optimize
from scipy.stats import chi2

# all lengths are in point unless otherwise specified
"""
styles is a dictionary which contains measurements necessary to create figures for
different styles of papers. Currently implemented are:
  - 'prd_twocol'            : Single-column figure of a two-column PRD paper.
  - 'prd_twocol*'           : Double-column figure of a two-column PRD paper.

The parameters in each style dictionary are (as well as units):
  - 'colwidth'              : Width of a single column in this format (inches).
  - 'textwidth'             : Width of text size on page in this format (inches).
  - 'fontsize'              : Font size of paper (pts)
  - 'tickwidth'             : Width of tick in figure (pts)
  - 'ticklength'            : Length of tick in figure (pts)
  - 'axeswidth'             : Thickness of axes in figure (pts)
  - 'markersize'            : Size of markers (pts)
  - 'ebar_width'            : Thickness of error bars (pts)
  - 'capsize'               : Size of endcaps for error bars (pts)
  - 'ecap_width'            : Width of error bar caps (pts)
  - 'bottom_pad'            : Padding on bottom of frame (pts)
  - 'top_pad'               : Padding on top of frame (pts)
  - 'left_pad'              : Padding on left of frame (pts)
  - 'right_pad'             : Padding on right of frame (pts)
  - 'asp_ratio'             : Aspect ratio of figure.
"""
pts_per_inch = 72.27    # inches to pts
styles = {
    'matplotlib_default' : {

    },
    'prd_twocol' : {
        'colwidth'          : 246.0 / pts_per_inch,         # inches
        'textwidth'         : 510.0 / pts_per_inch,         # inches
        'fontsize'          : 10.0,                         # pts
        'tickwidth'         : 0.5,                          # pts
        'ticklength'        : 2.0,                          # pts
        'spinewidth'        : 0.5,                          # pts
        'axeswidth'         : 0.5,                          # pts
        'markersize'        : 1.0,                          # pts
        'ebar_width'        : 0.5,                          # pts
        'endcaps'           : 1.0,                          # pts
        'ecap_width'        : 0.4,                          # pts
        'bottom_pad'        : 0.2,                          # pts
        'top_pad'           : 0.95,                         # pts
        'left_pad'          : 0.22,                          # pts
        'right_pad'         : 0.95,                         # pts
        'asp_ratio'         : 4/3,
        'linewidth'         : 1.5,
    },
    'prd_twocol*' : {
        'colwidth'          : 510.0 / pts_per_inch,
        'textwidth'         : 510.0 / pts_per_inch,
        'fontsize'          : 10.0,
        'tickwidth'         : 0.75,
        'ticklength'        : 2.0,
        'spinewidth'        : 0.5,
        'axeswidth'         : 0.5,                          # pts
        'markersize'        : 1.0,                          # pts
        'ebar_width'        : 1.0,                          # pts
        'endcaps'           : 1.0,                          # pts
        'ecap_width'        : 0.4,                          # pts
        'bottom_pad'        : 0.5,                          # pts
        'top_pad'           : 1.5,                          # pts
        'left_pad'          : 0.5,                          # pts
        'right_pad'         : 1.5,                          # pts
        'asp_ratio'         : 4/3,
        'linewidth'         : 1.5,
    },
    'notebook' : {
        'colwidth'          : 10,         # inches
        'textwidth'         : 510.0 / pts_per_inch,         # inches
        'fontsize'          : 20.0,                         # pts
        'tickwidth'         : 1.0,                          # pts
        'ticklength'        : 4.0,                          # pts
        'axeswidth'         : 1.0,                          # pts
        'markersize'        : 30.0,                          # pts
        'ebar_width'        : 1.0,                          # pts
        'endcaps'           : 2.0,                          # pts
        'ecap_width'        : 1.0,                          # pts
        'bottom_pad'        : 0.5,                          # pts
        'top_pad'           : 1.5,                          # pts
        'left_pad'          : 0.5,                          # pts
        'right_pad'         : 1.5,                          # pts
        'asp_ratio'         : 2.0,
        'linewidth'         : 3.0,
    },
    'notebook_square' : {
        'colwidth'          : 8,         # inches
        'textwidth'         : 510.0 / pts_per_inch,         # inches
        'fontsize'          : 20.0,                         # pts
        'tickwidth'         : 1.0,                          # pts
        'ticklength'        : 4.0,                          # pts
        'axeswidth'         : 1.0,                          # pts
        'markersize'        : 30.0,                          # pts
        'ebar_width'        : 1.0,                          # pts
        'endcaps'           : 2.0,                          # pts
        'ecap_width'        : 1.0,                          # pts
        'bottom_pad'        : 0.5,                          # pts
        'top_pad'           : 1.5,                          # pts
        'left_pad'          : 0.5,                          # pts
        'right_pad'         : 1.5,                          # pts
        'asp_ratio'         : 1.0,
        'linewidth'         : 1.5,
    },
    'multiplot_nb' : {
        'colwidth'          : 30,         # inches
        'textwidth'         : 1500.0 / pts_per_inch,         # inches
        'fontsize'          : 50.0,                         # pts
        'tickwidth'         : 3.0,                          # pts
        'ticklength'        : 10.0,                          # pts
        'axeswidth'         : 3.0,                          # pts
        'markersize'        : 5.0,                          # pts
        'ebar_width'        : 3.0,                          # pts
        'endcaps'           : 5.0,                          # pts
        'ecap_width'        : 3.0,                          # pts
        'bottom_pad'        : 2.0,                          # pts
        'top_pad'           : 3.0,                          # pts
        'left_pad'          : 1.0,                          # pts
        'right_pad'         : 3.0,                          # pts
        'asp_ratio'         : 2.0,
        'linewidth'         : 1.5,
        'wfontsize'         : 200,
    },
    'talk' : {
        'colwidth'          : 1000.0 / pts_per_inch,
        'textwidth'         : 1000.0 / pts_per_inch,
        'fontsize'          : 50.0,
        'tickwidth'         : 0.5,
        'ticklength'        : 4.0,
        'spinewidth'        : 0.5,
        'axeswidth'         : 0.5,                          # pts
        'markersize'        : 20.0,                         # pts
        'ebar_width'        : 3.0,                          # pts
        'endcaps'           : 5.0,                          # pts
        'ecap_width'        : 2.0,                          # pts
        'bottom_pad'        : 0.5,                          # pts
        'top_pad'           : 1.5,                          # pts
        'left_pad'          : 0.5,                          # pts
        'right_pad'         : 1.5,                          # pts
        'asp_ratio'         : 16/9,
        'linewidth'         : 1.5,
        'tick_fontsize'     : 30.0,
        'leg_fontsize'      : 25.0,
        'wfontsize'         : 200,
    },
    'talk_onecol' : {
        'colwidth'          : 900.0 / pts_per_inch,
        'textwidth'         : 900.0 / pts_per_inch,
        # 'fontsize'          : 20.0,
        'fontsize'          : 50.0,
        'tickwidth'         : 0.5,
        'ticklength'        : 4.0,
        'spinewidth'        : 0.5,
        'axeswidth'         : 0.5,                          # pts
        'markersize'        : 20.0,                         # pts
        'ebar_width'        : 2.0,                          # pts
        'endcaps'           : 5.0,                          # pts
        'ecap_width'        : 3.0,                          # pts
        'bottom_pad'        : 0.5,                          # pts
        'top_pad'           : 1.5,                          # pts
        'left_pad'          : 0.5,                          # pts
        'right_pad'         : 1.5,                          # pts
        'asp_ratio'         : 4/3,
        'linewidth'         : 1.5,
        'tick_fontsize'     : 30.0,
        'leg_fontsize'      : 25.0,
        'wfontsize'         : 150,
    },
}

# initialize empty keys
default_style = styles['notebook']
for sty in styles.values():
    if sty == default_style:
        continue
    for key in default_style:
        if not key in sty:
            sty[key] = default_style[key]

"""List of all spines in matplotlib."""
spinedirs = ['top', 'bottom', 'left', 'right']

def format_float(f, ndec = 2):
    """
    Formats a float f as a string to ndec places. Replaces a decimal point '.' with 'p'.

    Parameters
    ----------
    f : float
        Float to format.
    ndec : int (default = 2)
        Number of decimal points to format the float to.
    
    Returns
    -------
    f_str : string
        f formatted as a string.
    """
    f_str = '{0:.2f}'.format(f)
    f_str_p = [c if c != '.' else 'p' for c in f_str]
    return ''.join(f_str_p)

# TODO change over to using pandas
import pandas as pd

class Entry:

    # enum for possible types an entry can have
    types = {'r' : np.float64, 'c' : np.complex64, 'i' : int, 's' : str, 'f' : float}
    inv_types = {v : k for (k, v) in types.items()}

    def __init__(self, entry, sigma = -1, sf = 2):
        self._entry = entry
        self._T = type(entry)
        self._type = Entry.inv_types[self._T]
        self._has_err = (sigma != -1)
        self._err = sigma
        self._sf = sf
        self._string_rep = self.gen_string_rep()

    def gen_string_rep(self):
        if self._type == 'r' or self._type == 'f':
            string_rep = export_float_latex(self._entry, sigma = self._err, sf = self._sf)
        elif self._type == 'c':
            # TODO implement
            pass
        elif self._type == 's':
            string_rep = self._entry
        elif self._type == 'i':
            string_rep = str(self._entry)
        self._string_rep = string_rep
        return string_rep

class Table:

    def __init__(self):
        self._entries = -1

    def __init__(self, entries, is_entry = True, sigma = None, sf = 2):
        """
        Initializes a table with data passed in.

        Parameters
        ----------
        self : Table
            Table of Entries to create
        entries : np.array[T]
            Entries to pass in. If is_entry, the type T defaults to Entry.
        is_entry : bool
            True if the entries data passed in is type Entry, false otherwise.
        sigma : np.array[T] (default = None)
            Error on data. Should only be an array if is_entry is false. Otherwise, defaults
            to None.
        sf : Sig figs on data.

        Returns
        -------
        """
        if len(entries.shape) == 1:    # one column table
            entries = np.expand_dims(entries, axis = 1)
        if is_entry:
            self._entries = entries
        else:
            Entries = np.empty(entries.shape, dtype = object)
            for i in range(entries.shape[0]):
                for j in range(entries.shape[1]):
                    # Entries[i, j] = Entry(entries[i, j], sf = sf) if sigma is -1 else Entry(entries[i, j], sigma = sigma[i, j], sf = sf)
                    Entries[i, j] = Entry(entries[i, j], sf = sf) if sigma is None else Entry(entries[i, j], sigma = sigma[i, j], sf = sf)
            self._entries = Entries
        self.update()

    def copy(self):
        cp_entries = np.copy(self._entries)
        cp = Table(cp_entries)
        return cp

    def insert_rows(self, rows):
        """
        Inserts rows into Table at row_idxs

        Parameters
        ----------
        self : Table
            Table to insert rows into
        rows : {int : Table}
            List of (idx, row) pairs to insert into table. Each row have the same column size as the original Table.
            The idx value for each pair in the dictionary is the row index to insert the row into. For example, if
            rows = {0 : A, 1 : B, 4 : C} and the original Table is [D, E, F], the insert order is [A, B, D, E, C, F].

        Returns
        -------
        Table
            Updated Table object.
        """
        N = self._dims[0] + len(rows)
        M = self._dims[1]
        new_entries = np.empty((N, M), dtype = object)
        old_idx = 0
        for ii in range(N):
            if ii in rows:
                # row_obj = rows[ii]
                row = rows[ii]._entries
                if row.shape == (1, M):
                    row = row.T
                assert row.shape == (M, 1), 'Inserted row at key ' + str(ii) + ' has wrong length.'
                row = row[:, 0]             # collapse extra row
            else:
                row = self._entries[old_idx, :]
                old_idx += 1
            new_entries[ii] = row
        new_table = Table(new_entries)
        return new_table

    def insert_cols(self, cols):
        cp = self.copy()
        cp._entries = cp._entries.T
        cp.update()
        cp_extend = cp.insert_rows(cols)
        cp_extend._entries = cp_extend._entries.T
        cp_extend.update()
        return cp_extend

    def update(self):
        """
        Updates Table based on self._entries
        """
        self._dims = self._entries.shape
        self._types = np.array([[self._entries[i, j]._type for j in range(self._dims[1])] for i in range(self._dims[0])])
        self._T = np.array([[self._entries[i, j]._T for j in range(self._dims[1])] for i in range(self._dims[0])])
        self._string_rep = self.gen_string_rep()
        return self

    # make an iterator?

    def gen_string_rep(self):
        string_rep = np.empty(self._dims, dtype = object)
        for i in range(self._dims[0]):
            for j in range(self._dims[1]):
                # string_rep[i, j] = export_float_latex(self._entries[i, j])
                string_rep[i, j] = self._entries[i, j].gen_string_rep()
        self._string_rep = np.array(string_rep)
        return string_rep

def export_float_latex(mu, sigma = None, sf = 2):
    """
    Returns a latex string that will render a float x up to sf significant figures
    on the error. If no error is passed in, then renders sf significant figures
    on the matrix itself.

    Parameters
    ----------
    x : np.float64
        Mean of value, or just the value, to print out.
    sigma : np.float64 (default = None)
        Error on value to print out. This parameter is optional. If sigma is None, then will just print the data's mean.
    sf : int
        Number of significant figures to print the error to. If sigma is not entered,
        prints x to sf significant figures. Defaults to 2.

    Returns
    -------
    string
        Formatted latex string.
    """
    if type(mu) is np.ndarray:
        fmt_str = '[\n   '
        for i in range(len(mu)):
            fmt_str += export_float_latex(mu[i], sigma[i], sf)
            fmt_str += '\n   '
        return fmt_str + ']'
    # if sigma is -1:
    if not sigma:
        prec = '.' + str(sf) + 'f'
        return f'{mu:{prec}}'
    sigma_scinot = f'{sigma:e}'
    sigma_tokens = sigma_scinot.split('e')
    sigma_power = int(sigma_tokens[1])
    sigma_rounded = round(sigma, 1 - sigma_power)
    s = f'{sigma_rounded:e}'.split('e')[0].replace('.', '')[:sf]
    if sigma_power >= 0:        # TODO this may break if there is a larger sigma
        s = s[:sigma_power + 1] + '.' + s[sigma_power + 1:]
    n_digits = sf - sigma_power
    mu_scinot = f'{mu:e}'
    mu_tokens = mu_scinot.split('e')
    mu_power = int(mu_tokens[1])
    # mu_fmt = format(mu, '0.' + str(n_digits + mu_power + 1) + 'f')
    # mu_fmt = format(mu, '0.' + str(n_digits + mu_power) + 'f')
    mu_fmt = format(mu, '0.' + str(n_digits - 1) + 'f')
    return mu_fmt + '(' + s + ')'

def underline_print(s):
    """Prints the string s with an underline of the same number of characters."""
    n_chars = len(s)
    print(s)
    print('-' * n_chars)
    return

def mat_to_pmat(M):
    """Formats a matrix as a pmatrix for latex output."""
    n, m = M.shape
    fmtstring = '\\begin{pmatrix}'
    for i in range(n):
        for j in range(m):
            if j == 0:
                fmt_string = fmt_string + ' '
            fmtstring = fmtstring + ' & '

def export_matrix_latex(M, sigmaM = None, sf = 2, epsilon = 1e-8):
    """
    Returns a latex string that will render an n x n matrix M to sf significant figures
    on the error. If no error is passed in, then renders sf significant figures
    on the matrix itself.

    Parameters
    ----------
    M : np.array[np.float64]
        Matrix values, or mean of values, to print out.
    sigmaM : np.array[np.float64]
        Error on matrix values to print out. This parameter is optional.
    sf : int
        Number of significant figures to print the error to. If sigmaM is not entered,
        prints M to sf significant figures. Defaults to 2.
    epsilon : float
        Tolerance for a number to be nonzero.

    Returns
    -------
    string
        Formatted latex string.
    """
    n, m = M.shape
    # if sigmaM is None:
    #     # TODO method stub
    #     fmtstring = a
    # else:
    fmtstring = '\\begin{pmatrix} '
    for i in range(n):
        for j in range(m):
            if np.abs(M[i, j]) < epsilon:
                Mij = '0'
            else:
                if sigmaM is None:
                    sg = -1
                else:
                    sg = sigmaM[i, j]
                # Mij = export_float_latex(M[i, j], sigmaM[i, j], sf)
                Mij = export_float_latex(M[i, j], sg, sf)
            fmtstring += Mij
            if j < m - 1:    # check for end of string
                fmtstring += ' & '
        if i < n - 1:
            fmtstring += ' \\\\ '
    fmtstring += ' \\end{pmatrix}'
    return fmtstring

# TODO move these into the Table class and add other formatting options. Default should be
# fmt = 'latex', but can also use fmt = 'notebook' and add other options as we go

def export_vert_table_latex(data, col_labels, row_labels = np.empty((), dtype = str), header = None, hline_idxs = None):
    """
    Formats a data table in the vertical direction, i.e. the first row is a set of column labels.
    data should either be a Table class instance or an array of string representations of the data.
    Output table will look like:
    col_labels[0]    | col_labels[1]    | ... | col_labels[n]    | col_labels[n + 1] | ... | col_labels[n + m]
    ----------------------------------------------------------------------------------------------------------
    row_labels[0, 0] | row_labels[0, 1] | ... | row_labels[0, n] | data[0, 0]        | ... | data[0, m]
    ----------------------------------------------------------------------------------------------------------
    row_labels[1, 0] | row_labels[1, 1] | ... | row_labels[1, n] | data[1, 0]        | ... | data[1, m]
    ----------------------------------------------------------------------------------------------------------
    ...
    ----------------------------------------------------------------------------------------------------------
    row_labels[k, 0] | row_labels[k, 1] | ... | row_labels[k, n] | data[k, 0]        | ... | data[k, m]
    Note that if the row_labels gets too confusing, one can just ignore then and input everything under
    the first row as data, where data is passed in as a string array.
    """
    if type(data) != str:
        data = data.gen_string_rep()
    n_rows, n_cols = data.shape[0] + 1, len(col_labels)
    n = (1 if len(row_labels.shape) == 1 else row_labels.shape[1])
    m = data.shape[1]
    assert n_cols == n + m
    if header is None:
        header = '\\begin{tabular}{|'
        for ii in range(len(col_labels)):
            header += 'c|'
        header += '} \\hline \\hline '
    if hline_idxs is None:
        hline_idxs = list(range(n_rows))
    all_labels = np.empty((n_rows, n_cols), dtype = object)
    all_labels.fill('')
    all_labels[0, :] = col_labels
    for k in range(1, n_rows):
        all_labels[k, :n] = row_labels[k - 1]
        all_labels[k, n:] = data[k - 1]
    body = ''
    for k in range(n_rows):
        for l in range(n_cols):
            body += all_labels[k, l]
            body += ' \\\\ ' if l == n_cols - 1 else ' & '
        if k in hline_idxs:
            body += ' \\hline '
    body += ' \\hline \\hline \\end{tabular} '
    return header + body

def export_hor_table_latex(data, row_labels, col_labels = np.empty((), dtype = str)):
    """
    Formats a data table in the vertical direction, i.e. the first row is a set of column labels.
    data should either be a Table class instance or an array of string representations of the data.
    Output table will look like:
    row_labels[0]     | col_labels[0, 0] | col_labels[0, 1] | ... | col_labels[0, n] |
    ----------------------------------------------------------------------------------
    row_labels[1]     | col_labels[1, 0] | col_labels[1, 1] | ... | col_labels[1, n] |
    ----------------------------------------------------------------------------------
    ...
    ----------------------------------------------------------------------------------
    row_labels[k]     | col_labels[k, 0] | col_labels[k, 1] | ... | col_labels[k, n] |
    ----------------------------------------------------------------------------------
    row_labels[k + 1] | data[0, 0]       | data[0, 1]       | ... | data[0, n]       |
    ----------------------------------------------------------------------------------
    row_labels[k + 2] | data[1, 0]       | data[1, 1]       | ... | data[1, n]       |
    ----------------------------------------------------------------------------------
    ...
    ----------------------------------------------------------------------------------
    row_labels[k + m] | data[m, 0]       | data[m, 1]       | ... | data[m, n]       |
    ----------------------------------------------------------------------------------
    """
    # TODO reimplement function for this orientation
    if type(data) != str:
        data = data.gen_string_rep()
    n_rows, n_cols = len(row_labels), data.shape[1] + 1
    k, n, m = col_labels.shape[0], data.shape[1], data.shape[0]
    assert n_rows == k + m
    header = '\\begin{tabular}{|'
    for ii in range(n + 1):
        header += 'c|'
    header += '} \\hline '
    all_labels = np.empty((n_rows, n_cols), dtype = object)
    all_labels.fill('')
    all_labels[:, 0] = row_labels
    for l in range(1, n_cols):
        all_labels[:k, l] = col_labels[:, l - 1]
        all_labels[k:, l] = data[:, l - 1]
    body = ''
    for k in range(n_rows):
        for l in range(n_cols):
            body += all_labels[k, l]
            body += ' \\\\ ' if l == n_cols - 1 else ' & '
        body += ' \\hline '
    body += ' \\end{tabular} '
    return header + body
