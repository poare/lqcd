### SU(N) gauge theory utils.

import numpy as np
import scipy as sp
import scipy.linalg
import time

import argparse
import sys

def gauge_adj(cfg):
    return np.conj(np.swapaxes(cfg, axis1=-1, axis2=-2))
def gauge_proj_hermitian(cfg):
    cfg *= 0.5
    cfg += gauge_adj(cfg)
    return cfg
def gauge_proj_traceless(cfg):
    Nc = cfg.shape[-1]
    if Nc == 1: # for Nc=1, do not remove overall phase
        return cfg
    diag_shift = np.trace(cfg, axis1=-1, axis2=-2) / Nc
    for c in range(Nc):
        cfg[...,c,c] -= diag_shift
    return cfg

def gauge_expm(A): # A must be anti-hermitian
    Nc = A.shape[-1]
    if Nc == 2: # specialized to 2x2
        a11 = A[...,0,0]
        a12 = A[...,0,1]
        Delta = np.sqrt(a11**2 - np.abs(a12)**2)
        out = np.zeros_like(A)
        out[...,0,0] = np.cosh(Delta) + a11*np.sinh(Delta)/Delta
        out[...,0,1] = a12*np.sinh(Delta)/Delta
        out[...,1,0] = -np.conj(a12)*np.sinh(Delta)/Delta
        out[...,1,1] = np.cosh(Delta) - a11*np.sinh(Delta)/Delta
        return out
    elif Nc == 1:
        return np.exp(A)
    else:
        P = A
        M = np.broadcast_to(np.identity(Nc), A.shape) + P
        for i in range(2,20):
            P = (A @ P) / i
            M += P
        return M
if __name__ == "__main__":
    np.random.seed(1234)
    A = np.random.normal(size=(2,2)) + 1j * np.random.normal(size=(2,2))
    A = 0.5 * (A + np.conj(np.transpose(A)))
    A -= np.identity(2) * np.trace(A) / 2
    U2 = sp.linalg.expm(1j * A).reshape(1,1,1,2,2)
    A = A.reshape(1,1,1,2,2)
    U1 = gauge_expm(1j * A)
    assert(np.allclose(U1, U2))
    print('[PASSED gauge_expm 2x2]')
if __name__ == '__main__':
    np.random.seed(1234)
    A = np.random.normal(size=(3,3)) + 1j * np.random.normal(size=(3,3))
    A = 0.5 * (A + np.conj(np.transpose(A)))
    A -= np.identity(3) * np.trace(A) / 3
    U2 = sp.linalg.expm(1j * A).reshape(1,1,1,3,3)
    A = A.reshape(1,1,1,3,3)
    U1 = gauge_expm(1j * A)
    assert(np.allclose(U1, U2))
    print('[PASSED gauge_expm 3x3]')


def open_plaqs_above(cfg, mu, nu):
    cfg0, cfg1 = cfg[mu], cfg[nu]
    a = cfg0
    b = np.roll(cfg1, -1, axis=mu)
    c = gauge_adj(np.roll(cfg0, -1, axis=nu))
    d = gauge_adj(cfg1)
    return a @ b @ c @ d
def open_plaqs_below(cfg, mu, nu):
    cfg0, cfg1 = cfg[mu], cfg[nu]
    a = cfg0
    b = gauge_adj(np.roll(np.roll(cfg1, -1, axis=mu), 1, axis=nu))
    c = gauge_adj(np.roll(cfg0, 1, axis=nu))
    d = np.roll(cfg1, 1, axis=nu)
    return a @ b @ c @ d
if __name__ == "__main__":
    shape = (2,4,4,2,2) # (Nd,L,L,Nc,Nc)
    init_cfg_A = 0.3*(np.random.normal(size=shape) + 1j*np.random.normal(size=shape))
    gauge_proj_hermitian(init_cfg_A)
    gauge_proj_traceless(init_cfg_A)
    cfg = gauge_expm(1j * init_cfg_A)
    assert(np.allclose(
        np.conj(np.trace(open_plaqs_above(cfg, 0, 1),
                         axis1=-1, axis2=-2)),
        np.trace(np.roll(open_plaqs_below(cfg, 0, 1), -1, axis=1),
                 axis1=-1, axis2=-2)
        ))
    print('[PASSED open_plaqs 2x2]')

def closed_plaqs(cfg):
    out = np.zeros(cfg.shape[1:-2], dtype=np.float64)
    Nd = cfg.shape[0]
    Nc = cfg.shape[-1]
    for mu in range(Nd-1):
        for nu in range(mu+1, Nd):
            out += np.real(np.trace(
                open_plaqs_above(cfg, mu, nu), axis1=-1, axis2=-2)) / Nc
    return out

def gauge_force(cfg):
    F = np.zeros(cfg.shape, dtype=np.complex128)
    # specialized to U(1)
    Nd = cfg.shape[0]
    Nc = cfg.shape[-1]
    # TODO: Remove double computation
    for mu in range(Nd):
        for nu in range(Nd):
            if mu == nu: continue
            if mu < nu:
                F[mu] += open_plaqs_above(cfg, mu, nu) + open_plaqs_below(cfg, mu, nu)
            else:
                F[mu] += gauge_adj(open_plaqs_above(cfg, nu, mu)) + open_plaqs_below(cfg, mu, nu)
    # plaq = open_plaqs_above(cfg, 0, 1)
    # plaq_down = open_plaqs_below(cfg, 0, 1)
    # plaq_left = open_plaqs_below(cfg, 1, 0)
    # F[0] = plaq + plaq_down
    # F[1] = gauge_adj(plaq) + plaq_left
    F = -1j * (F - gauge_adj(F)) / (2 * Nc)
    # print("gauge_force {:.8f}".format(np.mean(np.abs(F))))
    return F
# TEST:
def test_gauge_force():
    print("test_gauge_force")
    L = [4,4,4]
    Nd = len(L)
    shape = tuple([Nd] + list(L) + [2,2])
    beta = 2.0
    init_cfg_A = 0.3*(np.random.normal(size=shape)+1j*np.random.normal(size=shape))
    gauge_proj_hermitian(init_cfg_A)
    gauge_proj_traceless(init_cfg_A)
    cfg = gauge_expm(1j * init_cfg_A)
    old_S = -beta * np.sum(np.real(closed_plaqs(cfg)))

    # Random perturbation
    d = 0.000001
    dA = d*np.random.normal(size=shape)
    gauge_proj_hermitian(dA)
    gauge_proj_traceless(dA)
    F = beta * gauge_force(cfg)
    dS_thy = np.sum(np.trace(dA @ F, axis1=-1, axis2=-2))

    new_cfg = gauge_expm(1j * dA) @ cfg
    new_S = -beta * np.sum(np.real(closed_plaqs(new_cfg)))
    dS_emp = new_S - old_S
    print("dS (thy.) = {:.5g}".format(dS_thy))
    print("dS (emp.) = {:.5g}".format(dS_emp))
    ratio = dS_thy / dS_emp
    print("ratio = {:.8g}".format(ratio))
    assert(np.isclose(ratio, 1.0, 1e-4))
    print("[PASSED gauge_force 2x2]")
if __name__ == "__main__": test_gauge_force()

# Sample momentum
def sample_pi(shape):
    pi = np.random.normal(size=shape) + 1j*np.random.normal(size=shape)
    gauge_proj_hermitian(pi)
    gauge_proj_traceless(pi)
    return pi

class Action(object):
    def compute_action(self, cfg):
        raise NotImplementedError()
    def init_traj(self, cfg):
        raise NotImplementedError()
    def force(self, cfg, t):
        raise NotImplementedError()
    def make_tag(self):
        raise NotImplementedError()

class PureGaugeAction(Action):
    def __init__(self, beta):
        self.beta = beta
    def compute_action(self, cfg):
        return -self.beta * np.sum(np.real(closed_plaqs(cfg)))
    def init_traj(self, cfg):
        return self.compute_action(cfg)
    def force(self, cfg, t):
        return self.beta * gauge_force(cfg)
    def make_tag(self):
        return 'w_b{:.2f}'.format(self.beta)

def update_x_with_p(cfg, pi, action, t, dt):
    np.copyto(cfg, gauge_expm(1j * dt * pi) @ cfg)
def update_p_with_x(cfg, pi, action, t, dt):
    F = action.force(cfg, t)
    pi -= dt * F

# Mutates cfg, pi according to leapfrog update
def leapfrog_update(cfg, pi, action, tau, n_leap, verbose=True):
    if verbose: print("Leapfrog  update")
    start = time.time()
    dt = tau / n_leap
    update_x_with_p(cfg, pi, action, 0, dt / 2)
    for i in range(n_leap-1):
        update_p_with_x(cfg, pi, action, i*dt, dt)
        update_x_with_p(cfg, pi, action, (i+0.5)*dt, dt)
    update_p_with_x(cfg, pi, action, (n_leap-1)*dt, dt)
    update_x_with_p(cfg, pi, action, (n_leap-0.5)*dt, dt / 2)
    if verbose: print("TIME leapfrog {:.2f}s".format(time.time() - start))

def hmc_update(cfg, action, tau, n_leap, verbose=True):
    old_cfg = np.copy(cfg)
    old_S = action.init_traj(old_cfg)
    old_pi = sample_pi(cfg.shape)
    old_K = np.real(np.sum(np.trace(old_pi @ old_pi, axis1=-1, axis2=-2)) / 2)
    old_H = old_S + old_K

    cfg = np.copy(cfg)
    new_pi = np.copy(old_pi)
    leapfrog_update(cfg, new_pi, action, tau, n_leap, verbose=verbose)

    new_S = action.compute_action(cfg)
    new_K = np.real(np.sum(np.trace(new_pi @ new_pi, axis1=-1, axis2=-2)) / 2)
    new_H = new_S + new_K

    delta_H = new_H - old_H
    if verbose:
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))

    # metropolis step
    acc = 0
    if np.random.random() < np.exp(-delta_H):
        acc = 1
        S = new_S
    else:
        cfg = old_cfg
        S = old_S
    if verbose:
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))
    return cfg, S, acc


def run_hmc(L, n_step, n_skip, n_therm, tau, n_leap, action, cfg):
    V = np.prod(L)
    # MC updates
    total_acc = 0
    cfgs = []
    plaqs = []
    topos = []
    for i in tqdm.tqdm(range(-n_therm, n_step)):
        print("MC step {} / {}".format(i+1, n_step))
        cfg, S, acc = hmc_update(cfg, action, tau, n_leap)
        if i >= 0: total_acc += acc

        # avg plaq
        plaq = np.sum(np.real(closed_plaqs(cfg))) / V
        print("Average plaq = {:.6g}".format(plaq))
        # topo Q
        # topo = np.sum(compute_topo(cfg))
        # print("Topo = {:d}".format(int(round(topo))))

        # save cfg
        if i >= 0 and i % n_skip == 0:
            print("Saving cfg!")
            cfgs.append(cfg)
            plaqs.append(plaq)
            # topos.append(topo)

    print("MC finished.")
    print("Total acc {:.4f}".format(total_acc / n_step))
    return cfgs, plaqs, topos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HMC for U(1)/SU(N) gauge theory')
    # general params
    parser.add_argument('--seed', type=int)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--Ncfg', type=int, required=True)
    parser.add_argument('--n_skip', type=int, required=True)
    parser.add_argument('--n_therm', type=int, required=True)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--n_leap', type=int, default=20)
    parser.add_argument('--init_cfg', type=str)
    parser.add_argument('--Nc', type=int)
    # action params
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--beta', type=float)
    # lattice
    parser.add_argument('dims', metavar='d', type=int, nargs='+')
    args = parser.parse_args()
    print("args = {}".format(args))

    start = time.time()

    # handle params
    if len(args.tag) > 0:
        args.tag = "_" + args.tag
    if args.seed is None:
        args.seed = np.random.randint(np.iinfo('uint32').max)
        print("Generated random seed = {}".format(args.seed))
    np.random.seed(args.seed)
    print("Using seed = {}.".format(args.seed))
    L = args.dims
    Nd = len(L)
    shape = tuple([Nd] + list(L) + [args.Nc,args.Nc])
    if args.init_cfg is None:
        print('Generating warm init cfg.')
        init_cfg_A = 0.4*(np.random.normal(size=shape) + 1j*np.random.normal(size=shape))
        gauge_proj_hermitian(init_cfg_A)
        gauge_proj_traceless(init_cfg_A)
        cfg = gauge_expm(1j * init_cfg_A)
    else:
        print('Loading init cfg from {}.'.format(args.init_cfg))
        cfg = np.fromfile(args.init_cfg, dtype=np.complex128)
        cfg = cfg.reshape(shape)
    tot_steps = args.Ncfg * args.n_skip
    if args.type == "pure_gauge":
        assert(args.beta is not None)
        action = PureGaugeAction(args.beta)
    else:
        print("Unknown action type {}".format(args.type))
        sys.exit(1)

    # do the thing!
    cfgs, plaqs, topos = run_hmc(L, tot_steps, args.n_skip, args.n_therm,
                                 args.tau, args.n_leap, action, cfg)

    # write stuff out
    group_tag = 'u1' if args.Nc == 1 else 'su{:d}'.format(args.Nc)
    prefix = '{:s}_{:s}_N{:d}_skip{:d}_therm{:d}_{:s}{:s}'.format(
        group_tag, action.make_tag(), args.Ncfg, args.n_skip, args.n_therm,
        '_'.join(map(str, L)), args.tag)
    fname = prefix + '.dat'
    with open(fname, 'wb') as f:
        for cfg in cfgs:
            cfg.tofile(f)
    print("Wrote ensemble to {}".format(fname))
    fname = prefix + '.plaq.dat'
    with open(fname, 'wb') as f:
        np.array(plaqs).tofile(f)
    print("Wrote plaqs to {}".format(fname))
    fname = prefix + '.topo.dat'
    with open(fname, 'wb') as f:
        np.array(topos).tofile(f)
    print("Wrote topos to {}".format(fname))
    print("TIME ensemble gen {:.2f}s".format(time.time()-start))
