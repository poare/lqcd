"""
Parses Grid output to determine the residuals at each iteration for CGNE and CGNR. 
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

# mstr = '0p03'
mstr = '0p1'
m = fmt.str_to_float(mstr)

directory = '/Users/patrickoare/Dropbox (MIT)/research/multigrid/grid_out'
fname = f'{directory}/out_m{mstr}.txt'
print(f'Reading data from {fname}.')

########################################################################
########################### PARSE INPUT FILE ###########################
########################################################################

solve_type = 'cgnr'
residuals = {'cgnr' : [], 'cgne' : []}
with open(fname) as file:
    for line in file:
        if 'CGNR' in line:
            solve_type = 'cgnr'
        elif 'CGNE' in line:
            solve_type = 'cgne'
        if 'residual' in line and 'target' in line and not 'k=0' in line:
            token = line.split('residual ')[1].split(' target')[0]
            resid = float(token)
            residuals[solve_type].append(resid)

########################################################################
######################### WRITE TO OUTPUT FILE #########################
########################################################################

out_file = f'{directory}/summed_residuals/m{mstr}.h5'
print('Writing output.')

f = h5py.File(out_file, 'w')
for k, v in residuals.items():
    f[k] = v
f.close()

print(f'Output written to {out_file}.')