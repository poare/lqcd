### Compute mesons on given cfgs.

import argparse
import math
import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import time
from schwinger import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute mesons for Schwinger')
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--kappa', type=float, required=True)
    parser.add_argument('--skip_cumu', action='store_true')
    parser.add_argument('--xspace', type=int, default=4)
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()
    L = [args.Lx, args.Lt]
    V = np.prod(L)
    Nd = len(L)
    shape = tuple([args.Ncfg, Nd] + L)
    fname = args.tag + '.dat'
    with open(fname, 'rb') as f:
        cfgs = np.fromfile(f, dtype=np.complex128).reshape(shape)
    all_srcs = get_coarse_spatial(L, args.xspace)
    N_src = len(all_srcs)
    src = make_prop_src(all_srcs, L)
    start = time.time()
    iv_mesons = []
    is_mesons = []
    if not args.skip_cumu:
        iv_meson_cumus = []
        is_meson_cumus = []
    for i,cfg in enumerate(cfgs):
        if i % 10 == 0: print("Cfg {} ({:.2f}s)".format(i, time.time()-start))
        M = dirac_op(cfg, args.kappa)
        prop = sp.sparse.linalg.spsolve(M, src)
        resid = np.linalg.norm(M @ prop - src)
        print("Resid = {}".format(resid))
        prop = prop.reshape(V, NS, N_src, NS)
        prop = np.transpose(prop, axes=[2, 0, 1, 3])
        all_conn_meson = conn_meson_all(
            L, prop, all_srcs, g_plus, g_minus, xspace=args.xspace)
        all_disc_meson = disc_meson_all(
            L, prop, all_srcs, g_plus, g_minus, xspace=args.xspace)
        all_conn_meson = all_conn_meson.reshape(tuple([N_src] + L))
        all_disc_meson = all_disc_meson.reshape(tuple([N_src] + L))
        if not args.skip_cumu:
            iv_meson_cumu = cumu_estimate(all_conn_meson)
            iv_meson_cumus.append(coarse_mom_proj(args.xspace, iv_meson_cumu))
            is_meson_cumu = cumu_estimate(all_conn_meson - 2*all_disc_meson)
            is_meson_cumus.append(coarse_mom_proj(args.xspace, is_meson_cumu))
        iv_mom_proj = np.array([
            coarse_mom_proj(args.xspace, x) for x in all_conn_meson])
        is_mom_proj = np.array([
            coarse_mom_proj(args.xspace, x) for x in all_conn_meson - 2*all_disc_meson])
        iv_mesons.append(iv_mom_proj)
        is_mesons.append(is_mom_proj)
    fname = args.tag + '.iv_meson.dat'
    with open(fname, 'wb') as f:
        np.array(iv_mesons).tofile(f)
    print("Wrote IV mesons to {}".format(fname))
    fname = args.tag + '.is_meson.dat'
    with open(fname, 'wb') as f:
        np.array(is_mesons).tofile(f)
    print("Wrote IS mesons to {}".format(fname))
    if not args.skip_cumu:
        assert iv_meson_cumus is not None
        fname = args.tag + '.iv_meson_cumu.dat'
        with open(fname, 'wb') as f:
            np.array(iv_meson_cumus).tofile(f)
        print("Wrote IV meson cumus to {}".format(fname))
        assert is_meson_cumus is not None
        fname = args.tag + '.is_meson_cumu.dat'
        with open(fname, 'wb') as f:
            np.array(is_meson_cumus).tofile(f)
        print("Wrote IS meson cumus to {}".format(fname))
    print("TIME mesons {:.2f}s".format(time.time()-start))
