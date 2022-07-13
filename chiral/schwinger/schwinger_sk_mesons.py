### Compute mesons on given cfgs.

import argparse
import math
import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import time
from schwinger_sk import *
from analysis import *
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute mesons for Schwinger')
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--TE', type=int, required=True)
    parser.add_argument('--TM', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--Ncfg', type=int, required=True)
    #parser.add_argument('--kappa', type=float, required=True)
    parser.add_argument('--m', type=float, required=True)
    parser.add_argument('--r4E', type=int, default=1)
    parser.add_argument('--r4M', type=int, default=1)
    parser.add_argument('--skip_cumu', action='store_false')
    parser.add_argument('--skip_disco', action='store_false')
    parser.add_argument('--xspace', type=int, default=4)
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()
    L = [args.Lx, args.TE + 2*args.TM]
    TE = args.TE
    TM = args.TM
    m = args.m
    r4E = args.r4E
    r4M = args.r4M
    kappa = m_to_kappa(m, 2 - 1 + r4E)
    V = np.prod(L)
    Nd = len(L)
    shape = tuple([args.Ncfg, Nd] + L)
    fname = args.tag + '.dat'
    #with open(fname, 'rb') as f:
    #    cfgs = np.fromfile(f, dtype=np.complex128).reshape(shape)
    cfgs = np.array(np.ones(shape, dtype=np.complex128))
    t_src = int((2*TE - ((2*TE) % 3))/3)
    print("TE = {} \nTM = {} \ntsrc = {} \n".format(TE,TM,t_src))
    all_srcs = get_coarse_spatial(L, t_src, args.xspace)
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
        M = dirac_op(cfg, TE, TM, kappa, r4E, r4M)
        anti_M = anti_dirac_op(cfg, TE, TM, kappa, r4E, r4M)
        #Minv = np.linalg.inv(M.todense())
        #anti_Minv = np.linalg.inv(anti_M.todense())
        #prop = np.array(Minv @ src)
        #anti_prop = np.array(anti_Minv @ src)
        prop = sp.sparse.linalg.spsolve(M, src)
        anti_prop = sp.sparse.linalg.spsolve(anti_M, src)
        resid = np.linalg.norm(M @ prop - src)
        anti_resid = np.linalg.norm(anti_M @ anti_prop - src)
        print("Resid = {}".format(resid))
        print("antiResid = {}".format(anti_resid))
        prop = prop.reshape(V, NS, N_src, NS)
        prop = np.transpose(prop, axes=[2, 0, 1, 3])
        anti_prop = anti_prop.reshape(V, NS, N_src, NS)
        anti_prop = np.transpose(anti_prop, axes=[2, 0, 1, 3])
        all_conn_meson = conn_meson_all(
            L, prop, anti_prop, all_srcs, g_plus, g_minus, xspace=args.xspace)
        all_conn_meson = all_conn_meson.reshape(tuple([N_src] + L))
        iv_mom_proj = np.array([
            coarse_mom_proj(args.xspace, x) for x in all_conn_meson])
        iv_mesons.append(iv_mom_proj)
        # discos
        if not args.skip_disco:
            all_disco_srcs = get_coarse_spatial_all_time(L, args.xspace)
            disco_src = make_prop_src(all_disco_srcs, L)
            disco_prop = sp.sparse.linalg.spsolve(M, disco_src)
            disco_resid = np.linalg.norm(M @ disco_prop - disco_src)
            disco_prop = disco_prop.reshape(V, NS, N_src * L[1], NS)
            disco_prop = np.transpose(disco_prop, axes=[2, 0, 1, 3])
            all_disco_meson = disc_meson_all(
                L, disco_prop, t_src, all_srcs, g_plus, g_minus, xspace=args.xspace)
            all_disco_meson = all_disco_meson.reshape(tuple([N_src] + L))
            is_mom_proj = np.array([
                coarse_mom_proj(args.xspace, x) for x in all_conn_meson - 2*all_disc_meson])
            is_mesons.append(is_mom_proj)
        if not args.skip_cumu:
            iv_meson_cumu = cumu_estimate(all_conn_meson)
            iv_meson_cumus.append(coarse_mom_proj(args.xspace, iv_meson_cumu))
            is_meson_cumu = cumu_estimate(all_conn_meson - 2*all_disc_meson)
            is_meson_cumus.append(coarse_mom_proj(args.xspace, is_meson_cumu))
    # plot
    tE1f = np.arange(0, TE - t_src)
    tE1b = np.arange(TE, t_src, -1)
    tM = np.arange(0, TM)
    tE2f = np.arange(TE - t_src, TE)
    tE2b = np.arange(t_src, 0, -1)
    iv_m = 2*(math.pi/4 + np.arcsin((m - 1)/math.sqrt(2)))
    #iv_m = 2*m
    tZ = int((TE - t_src)/2)
    Z = np.real(iv_mesons[0][0][tZ])/(np.exp(-iv_m * tZ) + np.exp(-iv_m * (TE - tZ)))
    print("Analytic lattice meson mass = {:.5f}".format(iv_m))
    print("Continuum meson mass = {:.5f}".format(2*m))
    print("Numerical ground-state overlap = {:.5f}".format(Z))
    fitE1 = Z*(np.exp(-iv_m * tE1f) + np.exp(-iv_m * tE1b))
    fitMp = fitE1[-1]*np.exp(-1j * iv_m * tM)
    fitMm = fitE1[-1]*np.exp(-1j * iv_m * np.flip(tM,0))
    fitE2 = Z*(np.exp(-iv_m * tE2f) + np.exp(-iv_m * tE2b))
    fitline = np.concatenate((fitE1, fitMp, fitMm, fitE2))
    #iv_m = 2*m
    #tZ = TE - t_src
    #Z = np.real(iv_mesons[0][0][tZ])/(np.exp(-iv_m * tZ) + np.exp(-iv_m * (TE - tZ)))
    #print("Analytic lattice meson mass = {:.5f}".format(iv_m))
    #print("Continuum meson mass = {:.5f}".format(2*m))
    #print("Numerical ground-state overlap = {:.5f}".format(Z))
    #fitE1 = Z*(np.exp(-iv_m * tE1f) + np.exp(-iv_m * tE1b))
    #fitMp = fitE1[-1]*np.exp(-1j * iv_m * tM)
    #fitMm = fitE1[-1]*np.exp(-1j * iv_m * np.flip(tM,0))
    #fitE2 = Z*(np.exp(-iv_m * tE2f) + np.exp(-iv_m * tE2b))
    #fitline2 = np.concatenate((fitE1, fitMp, fitMm, fitE2))
    correlator = iv_mesons[0][0]
    t = np.arange(0, len(correlator))
    fig, ax = plt.subplots()
    #ax.plot(np.real(correlator), np.real(fitline), np.real(fitline2)) 
    ax.plot(t, np.real(correlator), np.real(fitline)) 
    ax.set(xlabel='t', ylabel='Re G(t)', title='Free fermion, TE={}, TM={}, r4E={}, r4M={}'.format(TE,TM,r4E,r4M))
    ax.grid
    #fig.savefig('SK_free_iv_prop_re_coarse.pdf')
    fig.savefig('SK_free_iv_prop_re_r4E_{}_r4M_{}_TE_{}_TM_{}.pdf'.format(r4E,r4M,TE,TM))
    plt.show()
    fig, ax = plt.subplots()
    #ax.plot(np.log(np.abs(np.real(correlator))), np.log(np.abs(np.real(fitline))), np.log(np.abs(np.real(fitline2)))) 
    ax.plot(t, np.log(np.abs(np.real(correlator))), np.log(np.abs(np.real(fitline))) ) 
    ax.set(xlabel='t', ylabel='ln|Re G(t)|', title='Free fermion, TE={}, TM={}, r4E={}, r4M={}'.format(TE,TM,r4E,r4M))
    ax.grid
    #fig.savefig('SK_free_iv_prop_log_abs_re_coarse.pdf')
    fig.savefig('SK_free_iv_prop_log_abs_re_r4E_{}_r4M_{}_TE_{}_TM_{}.pdf'.format(r4E,r4M,TE,TM))
    plt.show()
    fig, ax = plt.subplots()
    #ax.plot(np.imag(correlator), np.imag(fitline), np.imag(fitline2)) 
    ax.plot(t, np.imag(correlator), np.imag(fitline)) 
    ax.set(xlabel='t', ylabel='Im G(t)', title='Free fermion, TE={}, TM={}, r4E={}, r4M={}'.format(TE,TM,r4E,r4M))
    ax.grid
    #fig.savefig('SK_free_iv_prop_im_coarse.pdf')
    fig.savefig('SK_free_iv_prop_im_r4E_{}_r4M_{}_TE_{}_TM_{}.pdf'.format(r4E,r4M,TE,TM))
    plt.show()
    # save
    fname = args.tag + '.iv_meson.dat'
    with open(fname, 'wb') as f:
        np.array(iv_mesons).tofile(f)
    print("Wrote IV mesons to {}".format(fname))
    print(iv_mesons)
    fname = args.tag + '.is_meson.dat'
    if not args.skip_disco:
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
