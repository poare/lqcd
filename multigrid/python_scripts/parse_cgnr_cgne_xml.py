"""
Parses XML output to determine the vector of residuals at each time-slice at each iteration for CGNE and CGNR. 
"""

import os
import sys
import math
from time import time
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

from importlib import reload

sys.path.append('/Users/patrickoare/lqcd/utilities')
import pytools as pyt
import plottools as pt
import iotools as io
import formattools as fmt
pt.set_font()
default_style = fmt.styles['notebook']

########################################################################
########################### READ INPUT FILE ############################
########################################################################

mstr_lst = ['0p01']
# mstr_lst = ['0p00001', '0p0001', '0p001', '0p01', '0p1']

m_lst = [fmt.str_to_float(mstr) for mstr in mstr_lst]

directory = '/Users/patrickoare/Dropbox (MIT)/research/multigrid/grid_out'
# directory = '/Users/patrickoare/Dropbox (MIT)/research/multigrid/grid_out/pt_src'
methods = [
    'cgnr',
    'cgne',
]

# fnames = [f'{directory}/{method}_m{mstr}.xml' for method in methods]
residuals = {}
# print(f'Reading data from {fnames}.')

########################################################################
########################### PARSE INPUT FILE ###########################
########################################################################

for mstr in mstr_lst:
    for idx, method in enumerate(methods):
        fin = f'{directory}/{method}_m{mstr}.xml'
        print(f'Reading data from {fin}.')

        # fin = fnames[idx]
        tree = ET.parse(fin)
        root = tree.getroot()
        xml_data = root[0][0]
        niters, T = len(xml_data), len(xml_data[0])

        residuals[method] = np.zeros((niters, T), dtype = np.float64)
        for itr in range(niters):
            for t in range(T):
                residuals[method][itr, t] = float(xml_data[itr][t].text)

    # write to output file
    out_file = f'{directory}/residuals/m{mstr}.h5'
    print('Writing output.')

    f = h5py.File(out_file, 'w')
    for k, v in residuals.items():
        f[k] = v
    f.close()

    print(f'Output written to {out_file}.')