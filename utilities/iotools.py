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
#################################### HDF5 files ################################
################################################################################