### Experiments with Schwinger-Keldysh mixed real/imag time contours.

import argparse
import numpy as np
import tqdm
from schwinger import *

def split_plaqs(tE, tM, cfg):
    assert(cfg.shape[2] == tE + 2*tM)
    plaqs = np.real(ensemble_plaqs(cfg))
    return plaqs[:,:tE], plaqs[:,tE:tE+tM], plaqs[:,tE+tM:]
        
if __name__ == "__main__":
    # Model
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--tM', type=int, required=True)
    parser.add_argument('--tE', type=int, required=True)
    parser.add_argument('--L', type=int, required=True)
    parser.add_argument('--N_cfg', type=int, required=True)
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
    # action = SchwingerHBOnSKActionRestricted(beta, [None, None], tE, tM)

    fname = '{}_mc.dat'.format(args.out_prefix)
    cfgs = np.fromfile(fname, dtype=np.complex128).reshape(tuple([args.N_cfg] + list(shape)))
    S_Ms = []
    for cfg in tqdm.tqdm(cfgs):
        plaqs_E, plaqs_Mp, plaqs_Mm = split_plaqs(tE, tM, cfg)
        S_Ms.append(-beta * np.sum(plaqs_Mp - plaqs_Mm))
    fname = '{}_mc.SM.dat'.format(args.out_prefix)
    np.array(S_Ms).tofile(fname)
    print('Wrote SMs to {}'.format(fname))
