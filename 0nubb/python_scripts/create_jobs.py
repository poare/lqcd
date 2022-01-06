import numpy as np
from scipy.optimize import root
import h5py
import os
from utils import *

# switch this on if running for nnpp project
nnpp = False

# ensemble = '24I/ml0p01/'
# ensemble = '24I/ml0p005/'
ensemble = '32I/ml0p008/'
# ensemble = '32I/ml0p006/'
# ensemble = '32I/ml0p004/'

if nnpp:
    parent = '/Users/theoares/lqcd/0nubb/nnpp/input/'
else:
    parent = '/Users/theoares/lqcd/0nubb/chroma_scripts/input/' + ensemble

# n_cfgs = 11
n_cfgs = 10
mom_idx_list = range(2, 10)
for cfg_idx in range(n_cfgs):
    cfg = str(cfg_idx)
    template = parent + 'cfg_' + cfg + '/mom_1.ini.xml'
    for mom_idx in mom_idx_list:
        k = str(mom_idx)
        out_filename = parent + 'cfg_' + cfg + '/mom_' + k + '.ini.xml'
        fin = open(template, 'r')
        fout = open(out_filename, 'w+')
        if nnpp:
            changes = [
                ['<mom>-1 0 1 0</mom>', '<mom>-' + k + ' 0 ' + k + ' 0</mom>'],
                ['<mom>0 1 1 0</mom>', '<mom>0 ' + k + ' ' + k + ' 0</mom>'],
                ['<mom>1 1 0 0</mom>', '<mom>' + k + ' ' + k + ' 0 0</mom>'],
                ['<mom_idx>1</mom_idx>', '<mom_idx>' + k + '</mom_idx>'],
                ['<xml_file>/data/d10b/users/poare/0nubb/nnpp/cfg_' + cfg + '/mom_1.dat.xml</xml_file>', '<xml_file>/data/d10b/users/poare/0nubb/nnpp/cfg_' + cfg + '/mom_' + k + '.dat.xml</xml_file>']
            ]
        else:
            changes = [
                ['<mom>-1 0 1 0</mom>', '<mom>-' + k + ' 0 ' + k + ' 0</mom>'],
                ['<mom>0 1 1 0</mom>', '<mom>0 ' + k + ' ' + k + ' 0</mom>'],
                ['<mom>1 1 0 0</mom>', '<mom>' + k + ' ' + k + ' 0 0</mom>'],
                ['<mom_idx>1</mom_idx>', '<mom_idx>' + k + '</mom_idx>'],
                ['<xml_file>/data/d10b/users/poare/0nubb/chroma_dwf_inversions/' + ensemble + 'cfg_' + cfg + '/mom_1.dat.xml</xml_file>', '<xml_file>/data/d10b/users/poare/0nubb/chroma_dwf_inversions/' + ensemble + 'cfg_' + cfg + '/mom_' + k + '.dat.xml</xml_file>']
            ]
        for line in fin:
            for change in changes:
                line = line.replace(change[0], change[1])
            fout.write(line)
        fin.close()
        fout.close()
        print(out_filename + ' created.')
