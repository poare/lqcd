"""Single-configuration consistency check of the Pi / Z analysis chain.

Runs the SAME chain as the production path -- npr_analysis.run_mixing_chain,
which is itself built on utilities/pytools.py -- on a single configuration
instead of an ensemble of bootstrap samples. Only the loader differs. That is
the point: if the chain is wrong, it is wrong in both places.

What this can and cannot show
-----------------------------
It CANNOT validate the Chroma port. The port's correlators already agree with
the 2020 QLUA correlators to 4e-13, so any deterministic function of them
agrees too.

What it DOES check is our implementation of the analysis chain -- the irrep
projection, the mixing matrix, the amputation and the normalisation -- against
the published quantities. The reference is a bootstrap over 67 configurations,
so one configuration is expected to scatter around it by roughly sqrt(67)
times the bootstrap width. This catches factor-level and structural errors,
not percent-level ones.

    python3 single_cfg_test.py [chroma_txt]
"""
import sys, os
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python_scripts'))
import npr_analysis as N
import pytools as pt

DEFAULT_TXT = os.path.join(HERE, '..', 'duplication_test', 'cfg1600_chroma_17mom.txt')
REF = os.path.expanduser(
    '~/Dropbox/research/npr_momfrac/analysis_output/mixing_job22454/Pi_subset.h5')

latt = pt.Lattice(16, 48)


def read_single_cfg(path):
    """One configuration from the Chroma port's text output.

    Returns props [1,3,4,3,4], threepts (list of 4 of the same), and the
    momenta present. RAW correlators -- no hypervolume factor; see the
    normalisation note in npr_analysis.
    """
    raw = {}
    for line in open(path):
        p = line.split()
        if len(p) != 11:
            continue
        k = tuple(int(x) for x in p[1:5])
        s0, s1, c0, c1 = (int(x) for x in p[5:9])
        raw.setdefault((p[0], k), np.zeros((4, 4, 3, 3), dtype=np.complex64))[s0, s1, c0, c1] \
            = float(p[9]) + 1j * float(p[10])

    momenta = sorted({k for (t, k) in raw if t == 'prop'})
    props, threepts = {}, {}
    for k in momenta:
        props[k] = np.einsum('ijab->aibj', raw[('prop', k)])[np.newaxis, ...]
        threepts[k] = [np.einsum('ijab->aibj', raw[(f'O{mu+1}{mu+1}', k)])[np.newaxis, ...]
                       for mu in range(4)]
    return props, threepts, momenta


def main():
    txt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TXT
    props, threepts, momenta = read_single_cfg(txt)

    with h5py.File(REF, 'r') as f:
        ref_mom = f['momenta'][()]
        ref = {tuple(int(x) for x in ref_mom[i]): i for i in range(len(ref_mom))}
        rPi11, rPi12, rZq = f['Pi11'][()], f['Pi12'][()], f['Zq'][()]

    print(f"input     : {txt}")
    print(f"lattice   : {latt.LL}, V = {latt.vol}")
    print(f"reference : Pi_subset.h5, {rPi11.shape[1]} bootstraps over 67 configurations")
    print(f"momenta   : {len(momenta)} in input, "
          f"{sum(1 for k in momenta if k in ref)} also in reference\n")

    print(f"{'k':18s} {'Zq cfg':>9s} {'Zq ens':>9s} {'Pi11 cfg':>10s} {'Pi11 ens':>10s}"
          f" {'Pi12 cfg':>10s} {'Pi12 ens':>10s}")
    rows = []
    for k in momenta:
        if k not in ref:
            continue
        p_lat = N.ptwid(list(k), latt)
        if abs(N.detA(p_lat)) < 1e-10:        # A(p) not invertible; mixing.py drops these
            continue
        Pi11, Pi12, Zq = N.run_mixing_chain(props[k], threepts[k], list(k), latt)
        i = ref[k]
        rows.append((k, Zq[0].real, rZq[i].real.mean(),
                     Pi11[0].real, rPi11[i].real.mean(),
                     Pi12[0].real, rPi12[i].real.mean()))
        print(f"{str(k):18s} {rows[-1][1]:9.5f} {rows[-1][2]:9.5f}"
              f" {rows[-1][3]:10.5f} {rows[-1][4]:10.5f}"
              f" {rows[-1][5]:10.5f} {rows[-1][6]:10.5f}")

    if not rows:
        print("no comparable momenta"); return

    a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
    print(f"\n{'quantity':10s} {'cfg mean':>10s} {'ens mean':>10s} {'ratio':>8s}")
    for j, name in ((0, 'Zq'), (2, 'Pi11'), (4, 'Pi12')):
        print(f"{name:10s} {a[:,j].mean():10.5f} {a[:,j+1].mean():10.5f}"
              f" {a[:,j].mean()/a[:,j+1].mean():8.4f}")


if __name__ == '__main__':
    main()
