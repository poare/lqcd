"""Pseudospectra from a Krylov-Schur factorization.

The Krylov-Schur decomposition is

    D U = U R + u b^dagger,    U^dagger U = I_k,   ||u|| = 1,   U^dagger u = 0.

For any z in the complex plane define the (k+1) x k matrix

    A(z) = [ z I - R ; b^dagger ]

and s(z) = sigma_min(A(z)). Because [U, u] is an isometry,

    s(z) >= sigma_min(z I - D) = 1 / ||(z I - D)^{-1}||,

so { z : s(z) <= eps } is contained in the eps-pseudospectrum of D.

This is an INNER bound. It can demonstrate that Lambda_eps(D) reaches a given
region (e.g. across Re z = 0); it can never demonstrate that it does not.

Input is written by writeKSFactorization in Grid's Example_spec_kryschur.cc.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def read_ks(dir):
    """Read the Krylov-Schur factorization written by writeKSFactorization.

    Returns (R, b, meta) with R complex128 (k, k), b complex128 (k,), and meta a
    dict carrying k, norm2_u and beta_k from the bvec.txt header.
    """
    d = np.loadtxt(os.path.join(dir, "rayleigh.txt"), comments="#", ndmin=2)
    k = int(d[:, 0].max()) + 1
    R = np.zeros((k, k), dtype=np.complex128)
    R[d[:, 0].astype(int), d[:, 1].astype(int)] = d[:, 2] + 1j * d[:, 3]

    bpath = os.path.join(dir, "bvec.txt")
    d = np.loadtxt(bpath, comments="#", ndmin=2)
    b = (d[:, 1] + 1j * d[:, 2]).astype(np.complex128)

    meta = {"k": k}
    with open(bpath) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "=" in line:
                key, val = line.lstrip("# ").split("=", 1)
                if key.strip() != "k":       # keep the int inferred from R
                    meta[key.strip()] = float(val)

    if b.shape[0] != k:
        raise ValueError(f"b has length {b.shape[0]} but R is {k} x {k}")
    return R, b, meta


def check_ks(R, b, meta, tol=1e-10):
    """Sanity checks on a factorization before it is used for a bound.

    Returns a list of warning strings, empty if all is well. Warns rather than
    raises: a non-triangular R is legitimate for the harmonic restart (which
    forms R + g b^dagger), it just forfeits the triangular speedup. A norm2(u)
    far from 1 is the serious one -- it means the isometry assumption behind the
    bound does not hold as written, and the b column needs rescaling.
    """
    msgs = []
    lower = np.abs(np.tril(R, -1)).max() if R.shape[0] > 1 else 0.0
    if lower > tol * np.abs(R).max():
        msgs.append(f"R is not upper triangular: max |subdiagonal| = {lower:.3e}")
    nu = meta.get("norm2_u")
    if nu is not None and abs(nu - 1.0) > 1e-6:
        msgs.append(f"norm2(u) = {nu:.6e}, not 1: the bound needs a unit u")
    return msgs


def smin_grid(R, b, re, im, chunk=256, nthread=8):
    """s(z) = sigma_min([z I - R ; b^dagger]) on the grid re x im.

    Returns a real array of shape (len(im), len(re)), indexed [row, col] = [im, re]
    to match imshow/contour conventions.

    Brute force: one SVD per grid point. At k ~ 64 this is the right algorithm --
    the triangular / inverse-Lanczos machinery in the pseudospectra literature
    exists to make k ~ 10^3 feasible and buys nothing here. (Measured at k = 50:
    getting sigma_min from lambda_min(A^dag A) by eigvalsh is no faster than the
    SVD, and the O(k^2) triangular iteration needs ~20 sweeps to beat one O(k^3)
    factorization, which is a wash.)

    nthread <= 1 runs the serial loop. Otherwise the chunks are farmed out to a
    thread pool: np.linalg.svd releases the GIL for the LAPACK call, so threads
    give real parallelism with no pickling of R and no copies, and each worker
    writes a disjoint slice of the output. Measured ~9x on 14 cores.

    Keep `chunk` small. It sets both the task granularity and the size of the
    batch array (chunk * (k+1) * k * 16 bytes -- 10 MB at 256, 167 MB at 4096);
    256 measured 1.9x faster than 4096 at 14 threads, and marginally faster
    serially too.
    """
    k = R.shape[0]
    Z = (re[None, :] + 1j * im[:, None]).ravel()
    n = Z.shape[0]
    out = np.empty(n, dtype=np.float64)

    eye = np.eye(k, dtype=np.complex128)
    bdag = b.conj()

    def block(start):
        z = Z[start:start + chunk]
        m = z.shape[0]
        A = np.empty((m, k + 1, k), dtype=np.complex128)
        A[:, :k, :] = z[:, None, None] * eye - R
        A[:, k, :] = bdag
        out[start:start + m] = np.linalg.svd(A, compute_uv=False)[:, -1]

    starts = range(0, n, chunk)
    if nthread <= 1:
        for start in starts:
            block(start)
    else:
        with ThreadPoolExecutor(max_workers=nthread) as ex:
            list(ex.map(block, starts))   # list() forces the lazy map to finish

    return out.reshape(im.shape[0], re.shape[0])


def make_grid(R, pad=0.25, n_re=200, n_im=200):
    """A default z-grid bracketing the Ritz values, padded by `pad` of the spread."""
    ev = np.linalg.eigvals(R)
    dre = np.ptp(ev.real) or 1.0
    dim = np.ptp(ev.imag) or 1.0
    re = np.linspace(ev.real.min() - pad * dre, ev.real.max() + pad * dre, n_re)
    im = np.linspace(ev.imag.min() - pad * dim, ev.imag.max() + pad * dim, n_im)
    return re, im
