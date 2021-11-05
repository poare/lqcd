
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

class Entry:

    # enum for possible types an entry can have
    types = {'r' : np.float64, 'c' : np.complex64, 'i' : int, 's' : str, 'f' : float}
    inv_types = {v : k for (k, v) in types.items()}

    def __init__(self, entry, sigma = -1, sf = 2):
        self._entry = entry
        self._T = type(entry)
        self._type = Entry.inv_types[self._T]
        self._has_err = (sigma is not -1)
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

    def __init__(self, entries, is_entry = True, sigma = -1, sf = 2):
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
        sigma : np.array[T] or int
            Error on data. Should only be an array if is_entry is false. Otherwise, defaults
            to -1.
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
                    Entries[i, j] = Entry(entries[i, j], sf = sf) if sigma is -1 else Entry(entries[i, j], sigma = sigma[i, j], sf = sf)
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



# class FormatData:
#     """
#     FormatData class for formatting.
#
#     Fields
#     ------
#     _type : np.array[string]
#         Possibilities are: {'i' : int, 'r' : float, 'c' : np.complex64}
#     _data : np.array[_type]
#         Mean of value, or just the value, to print out.
#     _has_err : np.array[bool]
#         Whether or not this dataset has error.
#     _sigma : np.float64
#         Error on value to print out. This parameter is optional.
#     _dims : np.array[int]
#         Dimensions of dataset. Both _data and _sigma (if applicable) should be this
#         dimension.
#     _string_rep : str
#         String representation for the dataset.
#     """
#
#     def __init__(self, data, sigma = -1, type = 'r'):
#         """
#         Initializes a dataset with data of type passed in.
#
#         Parameters
#         ----------
#         x : np.float64
#             Mean of value, or just the value, to print out.
#         sigma : np.float64
#             Error on value to print out. This parameter is optional.
#         sf : int
#             Number of significant figures to print the error to. If sigma is not entered,
#             prints x to sf significant figures. Defaults to 2.
#
#         Returns
#         -------
#         string
#             Formatted latex string.
#         """
#         self._type = type
#         self._has_err = (sigma is not -1)
#         self._data = data
#         self._sigma = sigma
#         self._dims = data.shape
#         self._string_rep = ''
#         self.gen_string_rep()
#
#     def gen_string_rep(self, sf = 2):
#         """
#         Generates a string representation of the dataset.
#         """
#         string_rep = np.empty(self._dims, dtype = object)
#         # string_rep = []
#         for i in range(self._dims[0]):
#             # string_rep.append([])
#             for j in range(self._dims[1]):
#                 if self._has_err is True:
#                     string_rep[i, j] = export_float_latex(self._data[i, j], self._sigma[i, j], sf)
#                     # tmp = export_float_latex(self._data[i, j], self._sigma[i, j], sf)
#                     # string_rep[i].append(tmp)
#                 else:
#                     string_rep[i, j] = str(self._data[i, j])
#         self._string_rep = np.array(string_rep)
#         return string_rep

def export_float_latex(mu, sigma = -1, sf = 2):
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
    if sigma is -1:
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

# TODO move these into the Table class and add other formatting options. Default should be
# fmt = 'latex', but can also use fmt = 'notebook' and add other options as we go

def export_vert_table_latex(data, col_labels, row_labels = np.empty((), dtype = str)):
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
