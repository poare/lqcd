### Experiments with Schwinger-Keldysh mixed real/imag time contours.

import argparse
from enum import Enum
import numpy as np
import tqdm
from direct_sampling import gradient_flow
from schwinger import *

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
        for x in range(Lx):
            for t in range(Lt):
                U = cfg[0, x, t]
                staple_up_1 = cfg[1, (x+1)%Lx, t]
                staple_up_2 = np.conj(cfg[0, x, (t+1)%Lt])
                staple_up_3 = np.conj(cfg[1, x, t])
                staple_down_1 = np.conj(cfg[1, (x+1)%Lx, (t-1)%Lt])
                staple_down_2 = np.conj(cfg[0, x, (t-1)%Lt])
                staple_down_3 = cfg[1, x, (t-1)%Lt]
                A = (staple_up_1 * staple_up_2 * staple_up_3 +
                     staple_down_1 * staple_down_2 * staple_down_3)
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
                
                U = cfg[1, x, t]
                staple_left_1 = np.conj(cfg[0, (x-1)%Lx, (t+1)%Lt])
                staple_left_2 = np.conj(cfg[1, (x-1)%Lx, t])
                staple_left_3 = cfg[0, (x-1)%Lx, t]
                staple_right_1 = cfg[0, x, (t+1)%Lt]
                staple_right_2 = np.conj(cfg[1, (x+1)%Lx, t])
                staple_right_3 = np.conj(cfg[0, x, t])
                A = (staple_left_1 * staple_left_2 * staple_left_3 +
                     staple_right_1 * staple_right_2 * staple_right_3)
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
    return cfg, cur_S

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

        
if __name__ == "__main__":
    # Model
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tM', type=int, default=4)
    parser.add_argument('--tE', type=int, default=4)
    parser.add_argument('--L', type=int, default=10)
    parser.add_argument('--N_therm', type=int, default=10)
    parser.add_argument('--N_mc_iter', type=int, default=100)
    parser.add_argument('--N_mc_skip', type=int, default=2)
    parser.add_argument('--min_S', type=float, required=True)
    parser.add_argument('--max_S', type=float, required=True)
    parser.add_argument('--a_i', type=float)
    parser.add_argument('--warm_init_cfg', action='store_true')
    parser.add_argument('--out_prefix', type=str, required=True)
    args = parser.parse_args()
    beta, tM, tE, L = args.beta, args.tM, args.tE, args.L
    shape = (2,L,tE+2*tM)
    Vp = Vm = tM * L
    min_SM, max_SM = -2*beta*Vp, 2*beta*Vp
    assert(args.min_S >= min_SM)
    assert(args.max_S <= max_SM)
    S_range = [args.min_S, args.max_S]
    action = SchwingerHBOnSKActionRestricted(beta, S_range, tE, tM)

    if args.a_i is None:
        all_as = np.fromfile('{}.a.dat'.format(args.out_prefix))
        action.a = all_as[-1]
    else:
        action.a = args.a_i
    if args.warm_init_cfg:
        cfg = draw_cfg_in_range_sk(beta, tE, tM, S_range, shape)
    else:
        cfg = np.fromfile('{}.cfg.dat'.format(args.out_prefix), dtype=np.complex128).reshape(2, L, tE+2*tM)

    # MC in given bin
    action.S_range = S_range
    for i in tqdm.tqdm(range(args.N_therm)):
        action.heatbath_update(cfg)
    all_cfgs = []
    all_Ss = []
    for i in tqdm.tqdm(range(args.N_mc_iter)):
        for j in tqdm.tqdm(range(args.N_mc_skip)):
            action.heatbath_update(cfg)
        action.heatbath_update(cfg)
        all_cfgs.append(np.copy(cfg))
        all_Ss.append(action.compute_action(cfg))
        print('Iter {}: S = {}'.format(i, all_Ss[-1]), flush=True)
    fname = '{}_mc.S.dat'.format(args.out_prefix)
    np.array(all_Ss).tofile(fname)
    print('Wrote all actions to {}'.format(fname))
    fname = '{}_mc.dat'.format(args.out_prefix)
    np.array(all_cfgs).tofile(fname)
    print('Wrote cfgs to {}'.format(fname))
