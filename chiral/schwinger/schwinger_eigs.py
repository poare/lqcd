### Compute eig(M) on given cfgs.

import argparse
import math
import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import time
from schwinger import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute eig(M) for Schwinger')
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--kappa', type=float, required=True)
    parser.add_argument('--stop_after', type=int, help='Truncate after how many measurements.')
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()
    L = [args.Lx, args.Lt]
    V = np.prod(L)
    Nd = len(L)
    shape = tuple([args.Ncfg, Nd] + L)
    # FREE FIELD:
    # cfg = np.ones(shape[1:], dtype=np.complex128)
    # M = dirac_op(cfg, args.kappa)
    # eigM = np.linalg.eigvals(M.todense())
    # with open('free_field.eig_m.dat', 'wb') as f:
    #     eigM.tofile(f)
    # print("Wrote free field eigenvalues!")
    fname = args.tag + '.dat'
    with open(fname, 'rb') as f:
        cfgs = np.fromfile(f, dtype=np.complex128).reshape(shape)
    start = time.time()
    eigs = []
    for i,cfg in enumerate(cfgs):
        if args.stop_after is not None and i >= args.stop_after: break
        if i % 10 == 0: print("Cfg {} ({:.2f}s)".format(i, time.time()-start))
        M = dirac_op(cfg, args.kappa)
        eigM = np.linalg.eigvals(M.todense())
        eigs.append(eigM)
    fname = args.tag + '.eig_m.dat'
    with open(fname, 'wb') as f:
        np.array(eigs).tofile(f)
    print("Wrote eigs to {}".format(fname))
    print("TIME eigs {:.2f}s".format(time.time()-start))
