################################################################################
# This script saves common conventions for writing out files that I use in     #
# my Python code. To import the script, simply add the path with sys before    # 
# importing. For example, if lqcd/ is in /Users/theoares:                      #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from iotools import *                                                        #
#                                                                              #
# Author: Patrick Oare                                                         #
################################################################################

from __main__ import *

import numpy as np
import h5py
import io
import os
import h5py
import time
import re
import itertools

import xml.etree.ElementTree as ET

def grid_complex(cstring, check_fmt = True):
    """
    Parses a complex number from Grid. Assumes input is of the form (r,i) with no spaces, 
    and returns the complex number r + 1j*i.

    Parameters
    ----------
    cstring : str
        String representing the complex number to parse. Must be of the form (r,i), where 
        r and i are two floats.
    check_fmt : bool (default = True)
        Whether or not to check the input format is the same as Grid's.
    
    Returns
    -------
    z : np.complex128
        Complex number read out.
    """
    if check_fmt:
        assert cstring[0] == '(' and cstring[-1] == ')' and ',' in cstring, 'Invalid input format'
    tokens = cstring[1:-1].split(',')
    if check_fmt:
        assert len(tokens) == 2, 'wrong number of input tokens per line.'
    return np.float64(tokens[0]) + 1j * np.float64(tokens[1])

################################################################################
#################################### Text files ################################
################################################################################

class TextFile:
    """
    Parameters will be stored in text files with the following format (n |represents line number n), 
    for example storing two integers Nomega, Ntau and a complex Nomega-dimensional vector rho.
    File: ex.txt
        1 | Nomega 64
        2 | Ntau 16
        3 | rho
        4 | 1.319127129 0.4923979139
        5 | 2.931097941 0.0123928479
        ...
        6 | 0.212894233 1.1209739128
    """

    def __init__(self, info):
        self._info = info
    
    def __getitem__(self, key):
        return self._info[key]
    
    def __setitem__(self, key, value):
        self._info[key] = value
    
    def write_to_file(fname):

        return

def read_text(fname):
    return


################################################################################
#################################### XML files #################################
################################################################################

class XMLParser:
    """
    Parses an XML File.

    Fields
    ------
    file_path : str
        Path to XML file.
    skeleton  : dict
        Skeleton of the file paths.
    data      : dict
        Data in the file.
    """

    def __init__(self, file_path):
        """
        Constructs an XML Parser. 
        
        Fields
        """
        self._file_path = file_path
        self._skeleton  = {}
        self._data = {}
        self._read_file()
        return
    
    def _read_file(self):
        """Reads a file."""
        tree = ET.parse(self._file_path)
        root = tree.getroot()
        
        xml_data = root[0][0]
        return

    def skeleton(self):
        return self._skeleton

    def data(self):
        return self._data
    
    def reduce(self):
        """Returns a reduced version of data, which has been pruned to remove single-child roots."""
        # reduced = self._data
        # while len(reduced) == 1:
        #     reduced = reduced[]
        #     if len(reduced) == 0:
        #         return {}
        # return reduced

        # TODO method stub
        return

################################################################################
#################################### HDF5 files ################################
################################################################################