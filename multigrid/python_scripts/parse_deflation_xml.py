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

directory = '/Users/patrickoare/Dropbox (MIT)/research/multigrid/gcr_coeffs/run1'
methods = [
    'invert_50',
    'relax_50',
    # 'invert_5',
    # 'relax_5',
]

# fnames = [f'{directory}/{method}_m{mstr}.xml' for method in methods]
data = {}
# print(f'Reading data from {fnames}.')

########################################################################
########################### PARSE INPUT FILE ###########################
########################################################################

def xml_to_complex(z):
    """ Converts an XML string of the form '(a,b)' to a complex number z = a + 1j*b. """
    tokens = z[1:-1].split(',')
    return float(tokens[0]) + 1j * float(tokens[1])

for idx, method in enumerate(methods):
    fin = f'{directory}/{method}.xml'
    print(f'Reading data from {fin}.')

    # fin = fnames[idx]
    tree = ET.parse(fin)
    root = tree.getroot()
    # xml_data = root[0][0]

    coeffs   = root[0][0]
    betas  = root[0][1]
    alphas = root[0][2]
    assert coeffs.tag == 'data'
    assert betas.tag  == 'betas'
    assert alphas.tag == 'alphas'

    niters, max_len = len(coeffs), len(coeffs[-1])
    print(f'Number of iterations: {niters}. Max iterations saved: {max_len}.')

    data[method] = {
        'coeffs'   : np.zeros((niters, max_len), dtype = np.complex64),
        'alphas' : np.zeros((niters), dtype = np.complex64),
        # 'betas'  : np.zeros((niters, max_len), dtype = np.complex64),
        'betas'  : [],              # storing betas as a triangular list for GCR.py implementation
    }
    for itr in range(niters):
        data[method]['alphas'][itr] = xml_to_complex(alphas[itr].text)
        for ii, z in enumerate(coeffs[itr]):
            data[method]['coeffs'][itr, ii] = xml_to_complex(z.text)
        # for ii, z in enumerate(betas[itr]):
        #     data[method]['betas'][itr, ii] = xml_to_complex(z.text)
        # data[method]['betas'][itr] = []
        # data[method]['betas'][itr] = []
        data[method]['betas'].append([])
        for ii in range(itr + 1):
            data[method]['betas'][itr].append(
                xml_to_complex(betas[itr][ii].text)
            )
            

# write to output file
out_file = f'{directory}/parsed_data.h5'
print('Writing output.')

f = h5py.File(out_file, 'w')
# for k1, v1 in data.items():
#     for k2, v2 in v1.items():
#         f[f'{k1}/{k2}'] = v2
for k, v in data.items():
    f[f'{k}/coeffs'] = v['coeffs']
    f[f'{k}/alphas'] = v['alphas']
    for ii, b in enumerate(v['betas']):
        f[f'{k}/betas/{ii}'] = b
f.close()

print(f'Output written to {out_file}.')