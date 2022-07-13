### Schwinger utils.

import math
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
import time

NS = 2



### Experiments with Schwinger-Keldysh mixed real/imag time contours.

import argparse
from enum import Enum
import numpy as np
import tqdm
from direct_sampling import gradient_flow
from direct_sampling.cftp import lattice_odd_mask
from schwinger import *

# def sample_vonmises_lattice(cfg, mask, mu, beta, a, S_range, cur_S):
#     Nd, Lx, Lt = cfg.shape
#     accept_count = 0
#     reject_count = 0
#     A_global = np.conj(compute_staples(cfg, mu))
#     for x in range(Lx):
#         for t in range(Lt):
#             if not mask[x,t]: continue
#             U = cfg[mu, x, t]
#             A = A_global[x,t]
#             old_local_S = -beta * np.real(U*A)
#             R = np.abs(A)
#             phi = np.angle(A)
#             beta_eff = R*(beta + a)
#             while True:
#                 new_U = np.exp(1j * np.random.vonmises(-phi, beta_eff))
#                 new_local_S = -beta * np.real(new_U*A)
#                 new_S = cur_S - old_local_S + new_local_S
#                 if new_S <= S_range[1] and new_S >= S_range[0]:
#                     accept_count += 1
#                     break
#                 else:
#                     reject_count += 1
#             cur_S = new_S
#             cfg[mu, x, t] = new_U
#     return cfg, accept_count/(accept_count + reject_count), cur_S

def get_block_masks(mask, block_dim):
    Lx,Lt = mask.shape
    assert(Lx % block_dim == 0)
    assert(Lt % block_dim == 0)
    bx,bt = Lx // block_dim, Lt // block_dim
    masks = [np.copy(mask) for i in range(bx*bt)]
    i = 0
    for x in range(bx):
        for t in range(bt):
            masks[i][:x*block_dim] = False
            masks[i][(x+1)*block_dim:] = False
            masks[i][:, :t*block_dim] = False
            masks[i][:, (t+1)*block_dim:] = False
            i += 1
    return masks
# print(get_block_masks(np.ones((10,10)), 5))

class SchwingerHBActionRestricted(PureGaugeAction):
    def __init__(self, beta, S_range):
        super().__init__(beta)
        self.S_range = S_range
        self.accept_rate = None
        self.a = 0
        
    def heatbath_update(self, cfg):
        Nd, Lx, Lt = cfg.shape # specialized for 2D
        cur_S = self.compute_action(cfg)
        reject_count = 0
        accept_count = 0

        ## NOTE: This works, but accept rate is awful because we are updating
        ## too many links at once. Constrained S range means there's a lot of
        ## dependency between link updates.
        # red_mask = lattice_odd_mask((Lx,Lt))
        # black_mask = np.logical_not(red_mask)
        # for mu in range(Nd):
        #     for mask in (red_mask, black_mask):
        #         opp_mask = np.logical_not(mask)
        #         A = np.conj(compute_staples(cfg, mu)) * mask
        #         old_local_S = -self.beta * np.real(cfg[mu]*A)
        #         R = np.abs(A)
        #         phi = np.angle(A)
        #         beta_eff = R*(self.beta + self.a)
        #         while True:
        #             new_U = np.exp(1j * np.random.vonmises(-phi, beta_eff, size=A.shape))
        #             new_U[opp_mask] = cfg[mu, opp_mask]
        #             new_local_S = -self.beta * np.real(new_U*A)
        #             new_S = cur_S - np.sum(old_local_S) + np.sum(new_local_S)
        #             if new_S <= self.S_range[1] and new_S >= self.S_range[0]:
        #                 accept_count += 1
        #                 print('accept')
        #                 break
        #             else:
        #                 reject_count += 1
        #                 print('reject')
        #         cur_S = new_S
        #         cfg[mu, mask] = new_U[mask]

        ## NOTE: Instead, can precompute staples if we have a good update order
        red_mask = lattice_odd_mask((Lx,Lt))
        black_mask = np.logical_not(red_mask)
        for mask in (red_mask, black_mask):
            A_global = np.conj(compute_staples(cfg, 0))
            for x in range(Lx):
                for t in range(Lt):
                    if not mask[x,t]: continue
                    U = cfg[0, x, t]
                    # staple_up_1 = cfg[1, (x+1)%Lx, t]
                    # staple_up_2 = np.conj(cfg[0, x, (t+1)%Lt])
                    # staple_up_3 = np.conj(cfg[1, x, t])
                    # staple_down_1 = np.conj(cfg[1, (x+1)%Lx, (t-1)%Lt])
                    # staple_down_2 = np.conj(cfg[0, x, (t-1)%Lt])
                    # staple_down_3 = cfg[1, x, (t-1)%Lt]
                    # A = (staple_up_1 * staple_up_2 * staple_up_3 +
                    #      staple_down_1 * staple_down_2 * staple_down_3)
                    A = A_global[x,t]
                    old_local_S = -self.beta * np.real(U*A)
                    R = np.abs(A)
                    phi = np.angle(A)
                    beta_eff = R*(self.beta + self.a)
                    while True:
                        new_U = np.exp(1j * np.random.vonmises(-phi, beta_eff))
                        new_local_S = -self.beta * np.real(new_U*A)
                        new_S = cur_S - old_local_S + new_local_S
                        if new_S <= self.S_range[1] and new_S >= self.S_range[0]:
                            accept_count += 1
                            break
                        else:
                            reject_count += 1
                    cur_S = new_S
                    cfg[0, x, t] = new_U

        for mask in (red_mask, black_mask):
            A_global = np.conj(compute_staples(cfg, 1))
            for x in range(Lx):
                for t in range(Lt):
                    if not mask[x,t]: continue
                    U = cfg[1, x, t]
                    # staple_left_1 = np.conj(cfg[0, (x-1)%Lx, (t+1)%Lt])
                    # staple_left_2 = np.conj(cfg[1, (x-1)%Lx, t])
                    # staple_left_3 = cfg[0, (x-1)%Lx, t]
                    # staple_right_1 = cfg[0, x, (t+1)%Lt]
                    # staple_right_2 = np.conj(cfg[1, (x+1)%Lx, t])
                    # staple_right_3 = np.conj(cfg[0, x, t])
                    # A = (staple_left_1 * staple_left_2 * staple_left_3 +
                    #      staple_right_1 * staple_right_2 * staple_right_3)
                    A = A_global[x,t]
                    old_local_S = -self.beta * np.real(U*A)
                    R = np.abs(A)
                    phi = np.angle(A)
                    beta_eff = R*(self.beta + self.a)
                    while True:
                        new_U = np.exp(1j * np.random.vonmises(-phi, beta_eff))
                        new_local_S = -self.beta * np.real(new_U*A)
                        new_S = cur_S - old_local_S + new_local_S
                        if new_S <= self.S_range[1] and new_S >= self.S_range[0]:
                            accept_count += 1
                            break
                        else:
                            reject_count += 1
                    cur_S = new_S
                    cfg[1, x, t] = new_U

        # FORNOW:
        # assert np.isclose(self.compute_action(cfg), cur_S)
        self.accept_rate = float(accept_count) / (accept_count + reject_count)


def split_plaqs(tE, tM, cfg):
    assert(cfg.shape[2] == tE + 2*tM)
    plaqs = np.real(ensemble_plaqs(cfg))
    return plaqs[:,:tE], plaqs[:,tE:tE+tM], plaqs[:,tE+tM:]

class Contour(Enum):
    EUCLIDEAN = 1
    MINKOWSKI = 2

class SchwingerHBOnSKActionRestricted(PureGaugeAction):
    # Contour in time is [0,tE] imaginary, [tE, tE+tM] real fwd,
    # [tE+tM, tE+2tM] real bwd.
    def __init__(self, beta, S_range, tE, tM):
        super().__init__(beta)
        self.S_range = S_range
        self.tE = tE
        self.tM = tM
        self.accept_rate = None
        self.a = 0

    def compute_action(self, cfg):
        plaqs_E, plaqs_Mp, plaqs_Mm = split_plaqs(self.tE, self.tM, cfg)
        return -self.beta * np.sum(plaqs_Mp - plaqs_Mm)

    # Heatbath with weight beta in euclidean, a in real time,
    # and restriction on S_M+ - S_M- into S_range.
    def heatbath_update(self, cfg):
        Nd, Lx, Lt = cfg.shape # specialized for 2D
        assert(Lt == self.tE + 2*self.tM)
        plaqs_E, plaqs_Mp, plaqs_Mm = split_plaqs(self.tE, self.tM, cfg)
        cur_SE = -self.beta * np.sum(plaqs_E)
        cur_SM = -self.beta * np.sum(plaqs_Mp - plaqs_Mm)
        reject_count = 0
        accept_count = 0
        for x in range(Lx):
            for t in range(Lt):
                # weight based on where in contour each staple falls
                if t < tE:
                    weight_left = weight_right = weight_up = self.beta
                    action_up = action_left = action_right = Contour.EUCLIDEAN
                elif t < tE+tM:
                    weight_up = weight_left = weight_right = self.beta
                    action_up = action_left = action_right = Contour.MINKOWSKI
                else:
                    weight_up = weight_left = weight_right = -self.beta
                    action_up = action_left = action_right = Contour.MINKOWSKI
                if (t-1)%Lt < tE:
                    weight_down = self.beta
                    action_down = Contour.EUCLIDEAN
                elif (t-1)%Lt < tE+tM:
                    weight_down = self.beta
                    action_down = Contour.MINKOWSKI
                else:
                    weight_down = -self.beta
                    action_down = Contour.MINKOWSKI
                    
                U = cfg[0, x, t]
                staple_up_1 = cfg[1, (x+1)%Lx, t]
                staple_up_2 = np.conj(cfg[0, x, (t+1)%Lt])
                staple_up_3 = np.conj(cfg[1, x, t])
                A_up = staple_up_1 * staple_up_2 * staple_up_3
                staple_down_1 = np.conj(cfg[1, (x+1)%Lx, (t-1)%Lt])
                staple_down_2 = np.conj(cfg[0, x, (t-1)%Lt])
                staple_down_3 = cfg[1, x, (t-1)%Lt]
                A_down = staple_down_1 * staple_down_2 * staple_down_3
                old_local_SM = 0
                old_local_SE = 0
                A = 0
                if action_up == Contour.MINKOWSKI:
                    old_local_SM -= weight_up * np.real(U*A_up)
                    A += self.a * weight_up * A_up
                else:
                    old_local_SE -= weight_up * np.real(U*A_up)
                    A += weight_up * A_up
                if action_down == Contour.MINKOWSKI:
                    old_local_SM -= weight_down * np.real(U*A_down)
                    A += self.a * weight_down * A_down
                else:
                    old_local_SE -= weight_down * np.real(U*A_down)
                    A += weight_down * A_down
                    
                beta_eff = np.abs(A)
                phi = np.angle(A)
                while True:
                    new_U = np.exp(1j * np.random.vonmises(-phi, beta_eff))
                    new_local_SM = 0
                    if action_up == Contour.MINKOWSKI:
                        new_local_SM -= weight_up * np.real(new_U*A_up)
                    if action_down == Contour.MINKOWSKI:
                        new_local_SM -= weight_down * np.real(new_U*A_down)
                    new_SM = cur_SM - old_local_SM + new_local_SM
                    if new_SM <= self.S_range[1] and new_SM >= self.S_range[0]:
                        accept_count += 1
                        break
                    else:
                        reject_count += 1
                new_local_SE = 0
                if action_up == Contour.EUCLIDEAN:
                    new_local_SE -= weight_up * np.real(new_U*A_up)
                if action_down == Contour.EUCLIDEAN:
                    new_local_SE -= weight_down * np.real(new_U*A_down)
                new_SE = cur_SE - old_local_SE + new_local_SE
                cur_SM = new_SM
                cur_SE = new_SE
                cfg[0, x, t] = new_U
                
                U = cfg[1, x, t]
                staple_left_1 = np.conj(cfg[0, (x-1)%Lx, (t+1)%Lt])
                staple_left_2 = np.conj(cfg[1, (x-1)%Lx, t])
                staple_left_3 = cfg[0, (x-1)%Lx, t]
                A_left = staple_left_1 * staple_left_2 * staple_left_3
                staple_right_1 = cfg[0, x, (t+1)%Lt]
                staple_right_2 = np.conj(cfg[1, (x+1)%Lx, t])
                staple_right_3 = np.conj(cfg[0, x, t])
                A_right = staple_right_1 * staple_right_2 * staple_right_3
                old_local_SM = 0
                old_local_SE = 0
                A = 0
                if action_left == Contour.MINKOWSKI:
                    old_local_SM -= weight_left * np.real(U*A_left)
                    A += self.a * weight_left * A_left
                else:
                    old_local_SE -= weight_left * np.real(U*A_left)
                    A += weight_left * A_left
                if action_right == Contour.MINKOWSKI:
                    old_local_SM -= weight_right * np.real(U*A_right)
                    A += self.a * weight_right * A_right
                else:
                    old_local_SE -= weight_right * np.real(U*A_right)
                    A += weight_right * A_right

                beta_eff = np.abs(A)
                phi = np.angle(A)
                while True:
                    new_U = np.exp(1j * np.random.vonmises(-phi, beta_eff))
                    new_local_SM = 0
                    if action_left == Contour.MINKOWSKI:
                        new_local_SM -= weight_left * np.real(new_U*A_left)
                    if action_right == Contour.MINKOWSKI:
                        new_local_SM -= weight_right * np.real(new_U*A_right)
                    new_SM = cur_SM - old_local_SM + new_local_SM
                    if new_SM <= self.S_range[1] and new_SM >= self.S_range[0]:
                        accept_count += 1
                        break
                    else:
                        reject_count += 1
                new_local_SE = 0
                if action_left == Contour.EUCLIDEAN:
                    new_local_SE -= weight_left * np.real(new_U*A_left)
                if action_right == Contour.EUCLIDEAN:
                    new_local_SE -= weight_right * np.real(new_U*A_right)
                new_SE = cur_SE - old_local_SE + new_local_SE
                cur_SM = new_SM
                cur_SE = new_SE
                cfg[1, x, t] = new_U

        plaqs_E, plaqs_Mp, plaqs_Mm = split_plaqs(self.tE, self.tM, cfg)
        assert np.isclose(cur_SE, -self.beta * np.sum(plaqs_E))
        assert np.isclose(cur_SM, -self.beta * np.sum(plaqs_Mp - plaqs_Mm))
        self.accept_rate = float(accept_count) / (accept_count + reject_count)



# "Binary search" with gradient flow to hit the action bucket.
def draw_cfg_in_range(action, S_range, shape, rho=0.1):
    init_cfg_A = 0.4*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)
    cur_S = action.compute_action(cfg)
    while cur_S < S_range[0] or cur_S > S_range[1]:
        while cur_S < S_range[0]:
            cfg = gradient_flow.wilson_flow_u1(cfg, -rho, n_step=10)
            cur_S = action.compute_action(cfg)
        while cur_S > S_range[1]:
            cfg = gradient_flow.wilson_flow_u1(cfg, rho, n_step=10)
            cur_S = action.compute_action(cfg)
        rho /= 2
    print(cur_S, S_range)
    return cfg

def draw_cfg_in_range_sk(beta, tE, tM, S_range, shape):
    S_M = np.mean(S_range)
    Vp = Vm = tM * shape[1]
    theta_plus = np.arccos(-S_M / (2*beta*Vp)) / 2
    theta_minus = np.pi/2 - theta_plus
    init_cfg_A = 0.4*np.random.normal(size=shape)
    init_cfg_A[:, :, tE:] = 0
    init_cfg_A[0, :, 0] = 0
    init_cfg_A[1, ::2, tE:tE+tM] = theta_plus
    init_cfg_A[1, 1::2, tE:tE+tM] = -theta_plus
    init_cfg_A[1, ::2, tE+tM:] = theta_minus
    init_cfg_A[1, 1::2, tE+tM:] = -theta_minus
    cfg = np.exp(1j * init_cfg_A)
    plaq_E, plaq_Mp, plaq_Mm = split_plaqs(tE, tM, cfg)
    true_S_M = -beta * np.sum(plaq_Mp - plaq_Mm)
    assert np.isclose(S_M, true_S_M)
    return cfg

def robbins_monroe_dos(restrict_action, shape, beta, a,
                       N_step, N_therm_cfg, N_batch_cfg, N_batch_skip):
    S_range = restrict_action.S_range
    print('Running Robbins-Monroe for S_range = {}'.format(S_range))
    d_S = S_range[1] - S_range[0]
    mid_S = np.mean(S_range)
    all_Ss = []
    all_as = []
    for n in tqdm.tqdm(range(N_step)):
        cfg, _ = draw_cfg_in_range(restrict_action, S_range, shape)
        restrict_action.a = a
        for i in range(N_therm_cfg):
            restrict_action.heatbath_update(cfg)
        Ss = []
        for i in range(N_batch_cfg):
            for j in range(N_batch_skip):
                restrict_action.heatbath_update(cfg)
            Ss.append(restrict_action.compute_action(cfg))
        all_Ss.append(np.array(Ss))
        print('mean S(a={:.8f}) = {}'.format(a, np.mean(Ss)))
        print('acc rate = {:.3f}'.format(restrict_action.accept_rate))
        delta_S = np.mean(Ss) - mid_S
        da = delta_S * 12 / (d_S**2 * (n+1))
        a += da
        print('new a = {:.8f}'.format(a))
        all_as.append(a)
    return np.array(all_Ss), np.array(all_as)

def robbins_monroe_dos_sk(restrict_action, shape, a,
                          N_step, N_outer_therm, N_therm,
                          N_batch_cfg, N_batch_skip):
    S_range = restrict_action.S_range
    tE = restrict_action.tE
    tM = restrict_action.tM
    beta = restrict_action.beta
    print('Running Robbins-Monroe for S_range = {}'.format(S_range))
    d_S = S_range[1] - S_range[0]
    mid_S = np.mean(S_range)
    all_Ss = []
    all_as = []
    cfg = draw_cfg_in_range_sk(beta, tE, tM, S_range, shape)
    print('Running initial therm')
    for i in tqdm.tqdm(range(N_outer_therm)):
        restrict_action.heatbath_update(cfg)
    print('Begin main Robbins-Monroe iteration')
    for n in tqdm.tqdm(range(N_step)):
        restrict_action.a = a
        for i in range(N_therm):
            restrict_action.heatbath_update(cfg)
        Ss = []
        for i in range(N_batch_cfg):
            for j in range(N_batch_skip):
                restrict_action.heatbath_update(cfg)
            Ss.append(restrict_action.compute_action(cfg))
        all_Ss.append(np.array(Ss))
        print('mean S(a={:.8f}) = {}'.format(a, np.mean(Ss)), flush=True)
        print('acc rate = {:.3f}'.format(restrict_action.accept_rate), flush=True)
        delta_S = np.mean(Ss) - mid_S
        da = delta_S * 12 / (d_S**2 * (n+1))
        a += da
        print('new a = {:.8f}'.format(a), flush=True)
        all_as.append(a)
    return np.array(all_Ss), np.array(all_as), cfg

def m_to_kappa(m, d):
    return 1/(2*(m + d))

def kappa_to_m(kappa, d):
    return 1/(2*kappa) - d

def pauli(i):
    if i == 0: # ident in spinor space
        return np.array([[1,0], [0,1]], dtype=np.complex128)
    elif i == 1:
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    elif i == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    elif i == 3:
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    else:
        assert False

# Fixed 2-spinor options
g_plus = (pauli(1) + 1j * pauli(2))/2
g_minus = (pauli(1) - 1j * pauli(2))/2

## 4-spinor version for testing!
def gamma(i):
    if i == -1: # ident
        return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]], dtype=np.complex128)
    elif i == 0:
        return np.array([
            [0,0,0,1j],
            [0,0,1j,0],
            [0,-1j,0,0],
            [-1j,0,0,0]], dtype=np.complex128)
    elif i == 1:
        return np.array([
            [0,0,0,-1],
            [0,0,1,0],
            [0,1,0,0],
            [-1,0,0,0]], dtype=np.complex128)
    elif i == 2:
        return np.array([
            [0,0,1j,0],
            [0,0,0,-1j],
            [-1j,0,0,0],
            [0,1j,0,0]], dtype=np.complex128)
    elif i == 3:
        return np.array([
            [0,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
            [0,1,0,0]], dtype=np.complex128)
    elif i == 5:
        return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,-1]], dtype=np.complex128)
    else:
        assert(False) # unknown gamma

def get_coord_index(x, L):
    assert(len(L) == 2)
    return x[0] * L[1] + x[1]
def index_to_coord(i, L):
    assert(len(L) == 2)
    return (i // L[1], i % L[1])
# TEST:
def test_indices():
    print("test_indices")
    L_test = [4,4]
    for i in range(np.prod(L_test)):
        assert(i == get_coord_index(index_to_coord(i, L_test), L_test))
    print("GOOD\n")
if __name__ == "__main__": test_indices()

def dirac_op(cfg, TE, TM, kappa, sign=1, r4E = 1, r4M = 1):
    # print("Making dirac op...")
    start = time.time()
    Nd, L = cfg.shape[0], cfg.shape[1:]
    assert(Nd == 2 and len(L) == 2)
    assert(L[1] == TE + 2*TM)
    V = np.prod(L)
    cfg0, cfg1 = cfg[0], cfg[1]
    m = kappa_to_m(kappa, 1 + r4E) 
    indptr = []
    indices = []
    data = []
    for i in range(V):
        # get index and link for shifted fermion
        x = index_to_coord(i, L)
        fwd = list(x)
        fwd[0] = (fwd[0] + 1) % L[0]
        # fwd_sign = -1 if fwd[0] == 0 else 1
        fwd_sign = 1 # spatial PBC
        bwd = list(x)
        bwd[0] = (bwd[0] - 1) % L[0]
        # bwd_sign = -1 if bwd[0] == L[0]-1 else 1
        bwd_sign = 1 # spatial PBC
        up = list(x)
        up[1] = (up[1] + 1) % L[1]
        up_sign = -1 if up[1] == 0 else 1 # temporal APBC
        down = list(x)
        down[1] = (down[1] - 1) % L[1]
        down_sign = -1 if down[1] == L[1]-1 else 1 # temporal APBC
        link_fwd = fwd_sign*cfg0[x]
        link_bwd = bwd_sign*np.conj(cfg0[tuple(bwd)])
        link_up = up_sign*cfg1[x]
        link_down = down_sign*np.conj(cfg1[tuple(down)])
        j_fwd = get_coord_index(fwd, L)
        j_bwd = get_coord_index(bwd ,L)
        j_up = get_coord_index(up, L)
        j_down = get_coord_index(down, L)
        # SK contour [t, t - iTE, -iTE, t - itE]
        x = list(index_to_coord(i, L))
        xk = x[0]
        xk_f = (xk + 1) % L[0]
        xk_b = (xk - 1) % L[0]
        xt = x[1]
        xt_f = (xt + 1) % L[1]
        xt_b = (xt - 1) % L[1]
        if xt < TE: # Euclidean
            hop_k_sign = -1
            hop_4_sign = -1
            hop_gk_sign = sign
            hop_g4_sign = sign
            mass = 1
            r4 = r4E
        elif xt < TE + TM: # Minkowski + 
            hop_k_sign = complex(0, -1)
            hop_4_sign = complex(0, 1)
            hop_gk_sign = sign
            hop_g4_sign = sign * complex(0, 1)
            mass = complex(0, 1) * (m+2-1-r4M)/(m+2-1+r4E)
            r4 = r4M
        else: # Minkowski -
            hop_k_sign = complex(0, 1)
            hop_4_sign = complex(0, -1)
            hop_gk_sign = sign
            hop_g4_sign = sign * complex(0, -1)
            mass = complex(0, -1) * (m+2-1-r4M)/(m+2-1+r4E)
            r4 = r4M
        # build Dirac operator
        if NS == 2: # paulis for 2-spinors
            j_blocks = [(i, mass * pauli(0)),
                        (j_fwd, kappa * link_fwd * hop_k_sign * (pauli(0) - hop_gk_sign*pauli(1))),
                        (j_bwd, kappa * link_bwd * hop_k_sign * (pauli(0) + hop_gk_sign*pauli(1))),
                        #(j_up, kappa * link_up * hop_4_sign * (pauli(0) - hop_g4_sign*pauli(2))),
                        #(j_down, kappa * link_down * hop_4_sign * (pauli(0) + hop_g4_sign*pauli(2)))]
                        (j_up, kappa * link_up * hop_4_sign * (r4 * pauli(0) - hop_g4_sign*pauli(2))),
                        (j_down, kappa * link_down * hop_4_sign * (r4 * pauli(0) + hop_g4_sign*pauli(2)))]
        elif NS == 4: # gammas for 4-spinors
            j_blocks = [(i, mass * gamma(-1)),
                        (j_fwd, kappa * link_fwd * hop_k_sign * (gamma(-1) - hop_gk_sign*gamma(0))),
                        (j_bwd, kappa * link_bwd * hop_k_sign * (gamma(-1) + hop_gk_sign*gamma(0))),
                        (j_up, kappa * link_up * hop_4_sign * (gamma(-1) - hop_g4_sign*gamma(1))),
                        (j_down, kappa * link_down * hop_4_sign * (gamma(-1) + hop_g4_sign*gamma(1)))]
        else: assert(False) # unknown NS
        j_blocks.sort(key=lambda x: x[0])
        indptr.append(len(indices))
        for j,block in j_blocks:
            indices.append(j)
            data.append(block)
    indptr.append(len(indices))
    data = np.array(data, dtype=np.complex128)
    indptr = np.array(indptr)
    indices = np.array(indices)
    out = sp.sparse.bsr_matrix((data, indices, indptr), shape=(NS*V,NS*V))
    # print("TIME dirac op {:.2f}s".format(time.time() - start))
    rescale = 1/(2*kappa)
    return rescale*out

def anti_dirac_op(cfg, TE, TM, kappa, sign=1, r4E = 1, r4M = 1):
    # print("Making dirac op...")
    start = time.time()
    Nd, L = cfg.shape[0], cfg.shape[1:]
    assert(Nd == 2 and len(L) == 2)
    assert(L[1] == TE + 2*TM )
    V = np.prod(L)
    cfg0, cfg1 = cfg[0], cfg[1]
    m = kappa_to_m(kappa, 2)
    indptr = []
    indices = []
    data = []
    for i in range(V):
        # get index and link for shifted fermion
        x = index_to_coord(i, L)
        fwd = list(x)
        fwd[0] = (fwd[0] + 1) % L[0]
        # fwd_sign = -1 if fwd[0] == 0 else 1
        fwd_sign = 1 # spatial PBC
        bwd = list(x)
        bwd[0] = (bwd[0] - 1) % L[0]
        # bwd_sign = -1 if bwd[0] == L[0]-1 else 1
        bwd_sign = 1 # spatial PBC
        up = list(x)
        up[1] = (up[1] + 1) % L[1]
        up_sign = -1 if up[1] == 0 else 1 # temporal APBC
        down = list(x)
        down[1] = (down[1] - 1) % L[1]
        down_sign = -1 if down[1] == L[1]-1 else 1 # temporal APBC
        link_fwd = fwd_sign*cfg0[x]
        link_bwd = bwd_sign*np.conj(cfg0[tuple(bwd)])
        link_up = up_sign*cfg1[x]
        link_down = down_sign*np.conj(cfg1[tuple(down)])
        j_fwd = get_coord_index(fwd, L)
        j_bwd = get_coord_index(bwd ,L)
        j_up = get_coord_index(up, L)
        j_down = get_coord_index(down, L)
        # SK contour [t, t - iTE, -iTE, t - itE]
        x = list(index_to_coord(i, L))
        xk = x[0]
        xk_f = (xk + 1) % L[0]
        xk_b = (xk - 1) % L[0]
        xt = x[1]
        xt_f = (xt + 1) % L[1]
        xt_b = (xt - 1) % L[1]
        if xt < TE: # Euclidean
            hop_k_sign = -1
            hop_4_sign = -1
            hop_gk_sign = sign
            hop_g4_sign = sign
            mass = 1
            r4 = r4E
        elif xt < TE + TM: # Minkowski + 
            hop_k_sign = complex(0, 1)
            hop_4_sign = complex(0, -1)
            hop_gk_sign = sign
            hop_g4_sign = sign * complex(0, -1)
            mass = complex(0, -1) * (m+2-1-r4M)/(m+2-1+r4E)
            r4 = r4M
        else: # Minkowski -
            hop_k_sign = complex(0, -1)
            hop_4_sign = complex(0, 1)
            hop_gk_sign = sign
            hop_g4_sign = sign * complex(0, 1)
            mass = complex(0, 1) * (m+2-1-r4M)/(m+2-1+r4E)
            r4 = r4M
        # build Dirac operator
        if NS == 2: # paulis for 2-spinors
            j_blocks = [(i, mass * pauli(0)),
                        (j_fwd, kappa * link_fwd * hop_k_sign * (pauli(0) - hop_gk_sign*pauli(1))),
                        (j_bwd, kappa * link_bwd * hop_k_sign * (pauli(0) + hop_gk_sign*pauli(1))),
                        #(j_up, kappa * link_up * hop_4_sign * (pauli(0) - hop_g4_sign*pauli(2))),
                        #(j_down, kappa * link_down * hop_4_sign * (pauli(0) + hop_g4_sign*pauli(2)))]
                        (j_up, kappa * link_up * hop_4_sign * (r4 * pauli(0) - hop_g4_sign*pauli(2))),
                        (j_down, kappa * link_down * hop_4_sign * (r4 * pauli(0) + hop_g4_sign*pauli(2)))]
        elif NS == 4: # gammas for 4-spinors
            j_blocks = [(i, mass * gamma(-1)),
                        (j_fwd, kappa * link_fwd * hop_k_sign * (gamma(-1) - hop_gk_sign*gamma(0))),
                        (j_bwd, kappa * link_bwd * hop_k_sign * (gamma(-1) + hop_gk_sign*gamma(0))),
                        (j_up, kappa * link_up * hop_4_sign * (gamma(-1) - hop_g4_sign*gamma(1))),
                        (j_down, kappa * link_down * hop_4_sign * (gamma(-1) + hop_g4_sign*gamma(1)))]
        else: assert(False) # unknown NS
        j_blocks.sort(key=lambda x: x[0])
        indptr.append(len(indices))
        for j,block in j_blocks:
            indices.append(j)
            data.append(block)
    indptr.append(len(indices))
    data = np.array(data, dtype=np.complex128)
    indptr = np.array(indptr)
    indices = np.array(indices)
    out = sp.sparse.bsr_matrix((data, indices, indptr), shape=(NS*V,NS*V))
    # print("TIME dirac op {:.2f}s".format(time.time() - start))
    rescale = 1/(2*kappa)
    return rescale*out

# Do it the stupid way first.
def det_sparse(M):
    start = time.time()
    out = sp.linalg.det(M.todense())
    # print("det = {:.6g} (TIME {:.2f}s)".format(out, time.time()-start))
    return out

### MESONS
def coarse_mom_proj(xspace, c):
    assert(len(c.shape) == 2) # 2D
    return np.sum(c[::xspace], axis=0)


def get_coarse_spatial(L, t, coarse_factor):
    out = []
    for x in range(0,L[0],coarse_factor):
        #for t in range(L[1]):
        out.append((x,t))
    return out

def get_coarse_spatial_all_time(L, coarse_factor):
    print(coarse_factor)
    out = []
    for x in range(0,L[0],coarse_factor):
        for t in range(L[1]):
            out.append((x,t))
    return out

# Compute isovector meson given N_src props from all_srcs
def conn_meson_all(L, all_prop, all_anti_prop, all_srcs, src_gamma, snk_gamma, **kwargs):
    # FORNOW: specialized for src_gamma = Gamma+, snk_gamma = Gamma-
    N_src = len(all_srcs)
    assert(N_src == all_prop.shape[0])
    #assert(N_src == L[0] * L[1] // kwargs['xspace'])
    V = all_prop.shape[1]
    assert(all_prop.shape[2:] == (NS,NS))
    assert(all_anti_prop.shape[2:] == (NS,NS))
    out = np.zeros((N_src, V), dtype=np.complex128)
    assert(NS == 2) # cannot do IV this way for 4-spinor
    topo = 0
    for j,src in enumerate(all_srcs):
        prop = all_prop[j].reshape((L[0], L[1], NS, NS))
        prop = np.roll(prop, (-src[0], -src[1]), axis=(0,1)).reshape((V,NS,NS))
        anti_prop = all_anti_prop[j].reshape((L[0], L[1], NS, NS))
        anti_prop = np.roll(anti_prop, (-src[0], -src[1]), axis=(0,1)).reshape((V,NS,NS))
        # prop_adj = np.conj(np.einsum('xab->xba', prop))
        # out += np.einsum('ab,xbc,cd,de,xef,fa->x',
        #                  pauli(3), prop_adj, pauli(3), snk_gamma, prop, src_gamma) / N_src
        out[j] = np.conj(anti_prop[:,0,0]) * prop[:,1,1]
        # phases = np.angle(out[j]).reshape(L)
        # phase0 = wrap(np.roll(phases, -1, axis=0) - phases)
        # phase1 = wrap(np.roll(phases, -1, axis=1) - phases)
        # topo += np.sum(compute_topo(np.array([ phase0, phase1 ])))
    # print('Mean topo = {:.4f}'.format(topo / len(all_srcs)))
    return out


def compute_staples(cfg, mu):
    Nd = cfg.shape[0]
    staples = np.zeros(cfg.shape[1:], dtype=cfg.dtype)
    for nu in range(Nd):
        if nu == mu: continue
        staples += cfg[nu] * np.roll(cfg[mu], -1, axis=nu) * np.conj(np.roll(cfg[nu], -1, axis=mu))
        staples += np.conj(np.roll(cfg[nu], 1, axis=nu)) * np.roll(cfg[mu], 1, axis=nu) * np.roll(cfg[nu], (1,-1), axis=(nu,mu))
    return staples
# def random_gauge_transform(cfg):
#     phases = np.exp(1j * np.random.normal(size=cfg.shape[1:]))
#     cfg_out = np.copy(cfg)
#     for mu in range(cfg.shape[0]):
#         cfg_out[mu] *= phases * np.conj(np.roll(phases, -1, axis=mu))
#     return cfg_out
def test_staples():
    shape = (2,10,10)
    cfg = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    plaqs = np.real(ensemble_plaqs(cfg))
    plaqs_2 = np.real(compute_staples(cfg, 0) * np.conj(cfg[0])) / 2
    assert np.isclose(np.mean(plaqs_2), np.mean(plaqs))
    print('GOOD')
# test_staples()

## TEST perf for heatbath:
if __name__ == "__main__":
    # Model
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tM', type=int, default=4)
    parser.add_argument('--tE', type=int, default=4)
    parser.add_argument('--L', type=int, default=10)
    parser.add_argument('--N_outer_therm', type=int, default=500)
    parser.add_argument('--N_therm', type=int, default=10)
    parser.add_argument('--N_mc_iter', type=int, default=100)
    parser.add_argument('--N_rm_iter', type=int, default=1000)
    parser.add_argument('--N_mc_skip', type=int, default=2)
    parser.add_argument('--min_S', type=float, required=True)
    parser.add_argument('--max_S', type=float, required=True)
    parser.add_argument('--out_prefix', type=str, required=True)
    args = parser.parse_args()
    beta, tM, tE, L = args.beta, args.tM, args.tE, args.L
    shape = (2,L,tE+2*tM)
    Vp = Vm = tM * L
    min_SM, max_SM = -2*beta*Vp, 2*beta*Vp
    assert(args.min_S >= min_SM)
    assert(args.max_S <= max_SM)
    action = SchwingerHBOnSKActionRestricted(beta, [None, None], tE, tM)

    # Robbins-Monroe on given bin
    S_range = [args.min_S, args.max_S]
    action.S_range = S_range
    all_Ss, all_as, cfg = robbins_monroe_dos_sk(
        action, shape, 0.0, args.N_rm_iter, args.N_outer_therm, args.N_therm,
        args.N_mc_iter, args.N_mc_skip)
    fname = '{}.S.dat'.format(args.out_prefix)
    np.array(all_Ss).tofile(fname)
    print('all_Ss.shape = {}'.format(all_Ss.shape))
    print('Wrote all actions to {}'.format(fname))
    fname = '{}.a.dat'.format(args.out_prefix)
    np.array(all_as).tofile(fname)
    print('all_as.shape = {}'.format(all_as.shape))
    print('Wrote all as to {}'.format(fname))
    fname = '{}.cfg.dat'.format(args.out_prefix)
    cfg.tofile(fname)
    print('Wrote final cfg to {}'.format(fname))

