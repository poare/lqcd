
################################################################################
# This script saves a list of common formatting functions for general data     #
# processing that I may need to use in my python code. To import the script,   #
# simply add the path with sys before importing. For example, if lqcd/ is in   #
# /Users/theoares:                                                             #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from formattools import *                                                    #
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

class FormatData:
    """
    FormatData class for formatting.

    Fields
    ------
    _type : string
        Possibilities are: {'i' : int, 'r' : float, 'c' : np.complex64}
    _data : np.array[_type]
        Mean of value, or just the value, to print out.
    _has_err : bool
        Whether or not this dataset has error.
    _sigma : np.float64
        Error on value to print out. This parameter is optional.
    _dims : np.array[int]
        Dimensions of dataset. Both _data and _sigma (if applicable) should be this
        dimension.
    _string_rep : str
        String representation for the dataset.
    """

    def __init__(self, data, sigma = -1, type = 'r'):
        """
        Initializes a dataset with data of type passed in.

        Parameters
        ----------
        x : np.float64
            Mean of value, or just the value, to print out.
        sigma : np.float64
            Error on value to print out. This parameter is optional.
        sf : int
            Number of significant figures to print the error to. If sigma is not entered,
            prints x to sf significant figures. Defaults to 2.

        Returns
        -------
        string
            Formatted latex string.
        """
        self._type = type
        self._has_err = (sigma is not -1)
        self._data = data
        self._sigma = sigma
        self._dims = data.shape
        self._string_rep = ''
        self.gen_string_rep()

    def gen_string_rep(self, sf = 2):
        """
        Generates a string representation of the dataset.
        """
        string_rep = np.empty(self._dims, dtype = object)
        # string_rep = []
        for i in range(self._dims[0]):
            # string_rep.append([])
            for j in range(self._dims[1]):
                if self._has_err is True:
                    string_rep[i, j] = export_float_latex(self._data[i, j], self._sigma[i, j], sf)
                    # tmp = export_float_latex(self._data[i, j], self._sigma[i, j], sf)
                    # string_rep[i].append(tmp)
                else:
                    string_rep[i, j] = str(self._data[i, j])
        self._string_rep = np.array(string_rep)
        return string_rep

def export_float_latex(mu, sigma, sf = 2):
    """
    Returns a latex string that will render a float x up to sf significant figures
    on the error. If no error is passed in, then renders sf significant figures
    on the matrix itself.

    Parameters
    ----------
    x : np.float64
        Mean of value, or just the value, to print out.
    sigma : np.float64
        Error on value to print out. This parameter is optional.
    sf : int
        Number of significant figures to print the error to. If sigma is not entered,
        prints x to sf significant figures. Defaults to 2.

    Returns
    -------
    string
        Formatted latex string.
    """
    sigma_scinot = f'{sigma:e}'
    sigma_tokens = sigma_scinot.split('e')
    sigma_power = int(sigma_tokens[1])
    sigma_rounded = round(sigma, 1 - sigma_power)
    s = f'{sigma_rounded:e}'.split('e')[0].replace('.', '')[:sf]
    n_digits = sf - sigma_power
    mu_scinot = f'{mu:e}'
    mu_tokens = mu_scinot.split('e')
    mu_power = int(mu_tokens[1])
    # mu_fmt = format(mu, '0.' + str(n_digits + mu_power + 1) + 'f')
    # mu_fmt = format(mu, '0.' + str(n_digits + mu_power) + 'f')
    mu_fmt = format(mu, '0.' + str(n_digits - 1) + 'f')
    return mu_fmt + '(' + s + ')'

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
    if sigmaM is None:
        # TODO method stub
        fmtstring = a
    else:
        fmtstring = '\\begin{pmatrix} '
        for i in range(n):
            for j in range(m):
                if np.abs(M[i, j]) < epsilon:
                    Mij = '0'
                else:
                    Mij = export_float_latex(M[i, j], sigmaM[i, j], sf)
                fmtstring += Mij
                if j < m - 1:    # check for end of string
                    fmtstring += ' & '
            if i < n - 1:
                fmtstring += ' \\\\ '
        fmtstring += ' \\end{pmatrix}'
    return fmtstring

def export_vert_table_latex(data, col_labels, row_labels = np.empty((), dtype = str)):
    """
    Formats a data table in the vertical direction, i.e. the first row is a set of column labels.
    data should either be a FormatData class instance or an array of string representations of the data.
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
    n, m = row_labels.shape[1], data.shape[1]
    assert n_cols == n + m
    header = '\\begin{tabular}{|'
    for ii in range(len(col_labels)):
        header += 'c|'
    header += '} \\hline '
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
        body += ' \\hline '
    body += ' \\end{tabular} '
    return header + body

def export_hor_table_latex(data, row_labels, col_labels = np.empty((), dtype = str)):
    """
    Formats a data table in the vertical direction, i.e. the first row is a set of column labels.
    data should either be a FormatData class instance or an array of string representations of the data.
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
