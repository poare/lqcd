## Compute topological charge for each cfg.

import argparse
import math
import sys
import numpy as np
import time
from schwinger import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute mesons for Schwinger')
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    args = parser.parse_args()

    start = time.time()
    L = [args.Lx, args.Lt]
    V = np.prod(L)
    Nd = len(L)
    shape = tuple([args.Ncfg, Nd] + L)
    fname = args.tag + '.dat'
    with open(fname, 'rb') as f:
        cfgs = np.fromfile(f, dtype=np.complex128).reshape(shape)

    topos = []
    for i,cfg in enumerate(cfgs):
        topo = np.sum(compute_topo(cfg))
        if i % 10 == 0:
            print("Cfg {} topo = {:.2f}".format(i, topo))
        topos.append(topo)
    fname = args.tag + '.topo.dat'
    with open(fname, 'wb') as f:
        np.array(topos).tofile(f)
    print("Wrote topos to {}".format(fname))
    print("TIME ensemble topos {:.2f}s".format(time.time()-start))    
