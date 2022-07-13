### Extract given cfg(s) from ensemble. Index specifier is numpy-format slicing.

import argparse
import math
import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import time
from schwinger import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract cfg(s) from ensemble')
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--ind', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()
    L = [args.Lx, args.Lt]
    Nd = len(L)
    shape = tuple([args.Ncfg, Nd] + L)
    fname = args.tag + '.dat'
    with open(fname, 'rb') as f:
        cfgs = np.fromfile(f, dtype=np.complex128).reshape(shape)
    sliced = eval("cfgs["+args.ind+"]")
    print('Extracted cfgs with shape {}.'.format(sliced.shape))
    with open(args.out, 'wb') as f:
        sliced.tofile(f)
    print('Wrote sliced cfgs to {}.'.format(args.out))
