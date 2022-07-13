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

    plaqs = []
    for i,cfg in enumerate(cfgs):
        plaq = np.sum(np.real(ensemble_plaqs(cfg)))  / V
        if i % 10 == 0:
            print("Cfg {} plaq = {:.16f}".format(i, plaq))
        plaqs.append(plaq)
    fname = args.tag + '.plaq.dat'
    with open(fname, 'wb') as f:
        np.array(plaqs).tofile(f)
    print("Wrote plaqs to {}".format(fname))
    print("TIME ensemble plaqs {:.2f}s".format(time.time()-start))    
