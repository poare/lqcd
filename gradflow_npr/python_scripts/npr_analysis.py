"""RI/MOM analysis for the quark EMT: correlators -> Zq, Pi11, Pi12, Z.

Built on utilities/pytools.py, which already provides the bootstrap, the
propagator inversion, the amputation, the quark field renormalisation and the
H(4) irrep projection. Only two things are added here, because pytools does
not have them:

  * the tensor structures Lambda1, Lambda2 and the mixing matrix A_ab, copied
    from npr_momfrac/python_scripts/analysis.py:27-28,309-327 -- these are the
    definitions the 2020 paper used, so they are the authority for reproducing
    its numbers;
  * loaders for the two correlator formats in play.

Normalisation, which is the easy thing to get wrong
---------------------------------------------------
analysis.py multiplies by the hypervolume in THREE places: the point-source
loader (analysis.py:154), amputate (:224) and quark_renorm (:239). With
props = V*S_raw and threepts = V*G_raw those factors cancel exactly:

    Sinv  = S_raw^-1 / V
    Gamma = Sinv G Sinv * V = S_raw^-1 G_raw S_raw^-1
    Zq    = i V sum phase tr[gamma Sinv] / (12 sum phase^2)
          = i   sum phase tr[gamma S_raw^-1] / (12 sum phase^2)

pytools carries no V factors at all, so feeding it the RAW correlators
reproduces analysis.py's Gamma and Zq identically, with no V bookkeeping.
That is what this module does. Do not reintroduce the factor.

pytools.quark_renorm takes the momentum vector directly, so it must be handed
q_mu = sin(2 pi (k_mu + b_mu) / L_mu) -- the FULL angle -- to match
analysis.py:234. Note pytools.Lattice.to_lattice_momentum does NOT add bvec;
the half-angle p~ used by the tensor structures is built here with it.
"""
import os, sys, types
import numpy as np

# pytools imports gvar and lsqfit at module scope but never uses either -- zero
# references to gvar. or lsqfit. anywhere in the file. Neither is installed
# here, so stub them rather than pull two packages in for dead imports. If they
# are ever installed (pip install gvar lsqfit) these lines become inert; if
# pytools ever starts using them for real, this will fail loudly rather than
# silently, which is the behaviour we want.
for _dead in ('gvar', 'lsqfit'):
    if _dead not in sys.modules:
        try:
            __import__(_dead)
        except ImportError:
            sys.modules[_dead] = types.ModuleType(_dead)

sys.path.insert(0, os.path.expanduser('~/lqcd/utilities'))
import pytools as pt

gamma = pt.gamma
delta = pt.delta
bvec  = pt.bvec


# ---------------------------------------------------------------- momenta

def ptwid(k, latt):
    """Half-angle lattice momentum WITH the twist: 2 sin(pi (k+b)/L).

    analysis.py:82. Used by the tensor structures and the Born term.
    """
    return np.array([np.complex64(2 * np.sin(np.pi * (k[mu] + bvec[mu]) / latt.LL[mu]))
                     for mu in range(4)])


def qsin(k, latt):
    """Full-angle momentum WITH the twist: sin(2 pi (k+b)/L).

    analysis.py:234. This is what Zq uses, and it is also the momentum
    function of the operator's own tree-level vertex.
    """
    return np.array([np.sin(2 * np.pi * (k[mu] + bvec[mu]) / latt.LL[mu])
                     for mu in range(4)])


# ------------------------------------------------- tensor structures, mixing

def Lambda1(p):
    """analysis.py:27."""
    return (-2j) * np.array([[(p[mu] * gamma[nu] + p[nu] * gamma[mu]) / 2
                              - delta[mu, nu] * pt.slash(p) / 4
                              for mu in range(4)] for nu in range(4)])


def Lambda2(p):
    """analysis.py:28."""
    return (-2j) * np.array([[p[mu] * p[nu] * pt.slash(p) / pt.square(p)
                              - delta[mu, nu] * pt.slash(p) / 4
                              for mu in range(4)] for nu in range(4)])


def inner(O1, O2):
    """analysis.py:304. Inner product on the tau_1^(3) irrep."""
    t1, _ = pt.form_2d_sym_irreps(O1)
    t2, _ = pt.form_2d_sym_irreps(O2)
    return sum(np.einsum('ij,ji', t1[n], t2[n]) for n in range(3))


def A_ab(p):
    """analysis.py:309."""
    L1, L2 = Lambda1(p), Lambda2(p)
    return np.array([[inner(L1, L1), inner(L1, L2)],
                     [inner(L2, L1), inner(L2, L2)]])


def A_inv_ab(p):
    """analysis.py:319."""
    L1, L2 = Lambda1(p), Lambda2(p)
    A11, A12, A22 = inner(L1, L1), inner(L1, L2), inner(L2, L2)
    det = A11 * A22 - A12 * A12
    return (1 / det) * np.array([[A22, -A12], [-A12, A11]])


def detA(p):
    """analysis.py:316. Momenta with |detA| below tolerance are dropped."""
    L1, L2 = Lambda1(p), Lambda2(p)
    return inner(L1, L1) * inner(L2, L2) - inner(L1, L2) ** 2


# ------------------------------------------------------------- the chain

def run_mixing_chain(props, threepts, k, latt):
    """Pi11, Pi12 and Zq at one momentum, mirroring analysis.py:558-599.

    Parameters
    ----------
    props : np.array [Nb, 3, 4, 3, 4]      raw S(p), no 1/V and no V
    threepts : list of 4 np.array [Nb, 3, 4, 3, 4]   raw G_mumu(p), mu = 0..3
    k : the integer momentum
    latt : pytools.Lattice

    Returns (Pi11, Pi12, Zq), each np.array [Nb].
    """
    props_inv = pt.invert_props(props)
    Zq = pt.quark_renorm(props_inv, qsin(k, latt))

    # The three tau_1^(3) combinations of the diagonal operators. pytools'
    # form_2d_sym_irreps expects a full rank-2 tensor, so build one whose
    # diagonal carries the four measured operators; the off-diagonal entries
    # are never touched by the tau_1^(3) projection.
    T = np.zeros((4, 4) + threepts[0].shape, dtype=np.complex64)
    for mu in range(4):
        T[mu, mu] = threepts[mu]
    G_irreps, _ = pt.form_2d_sym_irreps(T)

    Gamma_Dirac = []
    for n in range(3):
        x = pt.amputate_threepoint(props_inv, props_inv, G_irreps[n])
        Gamma_Dirac.append(np.einsum('baiaj->bij', x) / 3)     # colour trace

    p_lat = ptwid(k, latt)
    L1_irr, _ = pt.form_2d_sym_irreps(Lambda1(p_lat))
    L2_irr, _ = pt.form_2d_sym_irreps(Lambda2(p_lat))
    v = np.array([
        sum(np.einsum('ij,bji->b', L1_irr[n], Gamma_Dirac[n]) for n in range(3)),
        sum(np.einsum('ij,bji->b', L2_irr[n], Gamma_Dirac[n]) for n in range(3)),
    ])
    Pi11, Pi12 = A_inv_ab(p_lat).dot(v)
    return Pi11, Pi12, Zq
