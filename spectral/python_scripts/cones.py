"""
Conic projection of noisy Euclidean correlators onto the causal cone, following R. Abbott, N. Christ, S. Fields,
W. I. Jay and M. Saccardi, "Positivity, Noise, and Cones" (draft of 2026-10-16). Equation numbers refer to that
draft.

First pass: scalar (p = 1) moment sequences mu = (mu_0, ..., mu_{m-1}) with m = 2d - 1 odd, in double precision.
The three cones are
    Hamburger : H^{[d]} >= 0                                           (Eq. 37)
    Stieltjes : H^{[d]} >= 0,  H^{(1),[d-1]} >= 0                       (Eq. 38)
    Hausdorff : H^{[d]} >= 0,  H^{(1),[d-1]} - H^{(2),[d-1]} >= 0       (Sec. II B, after Eq. 25)
The projection (Eq. 73 / Problem 82) is solved by a primal log-barrier path-following method in whitened coordinates.
Its central path is the perturbed complementarity condition M H(mu) = eps * 1 of Eq. (88), so the dual variables
M = eps * H(mu)^{-1} come for free and give the polar component -Sigma h(M) of Eq. (92c).
"""

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats

HAMBURGER = 'hamburger'
STIELTJES = 'stieltjes'
HAUSDORFF = 'hausdorff'

CATALAN = 0.915965594177219015054603514932384110774
BECKERMANN_GAMMA = np.exp(4 * CATALAN / np.pi)          # Eq. (29), approx 3.210

################################################################################
############################## HANKEL LINEAR MAPS ##############################
################################################################################

def hankel_basis(m, shift, size):
    """
    Basis tensor B for the shifted Hankel map H^{(shift),[size]} : R^m -> S^size, so that
    H^{(shift),[size]}(mu) = sum_t mu_t B[t], with B[t]_{ij} = 1 if i + j + shift == t and 0 otherwise.

    Parameters
    ----------
    m : int
        Number of moments.
    shift : int
        Shift r of the Hankel matrix, [mu_{i+j+r}]_{i,j}.
    size : int
        Dimension of the Hankel matrix.

    Returns
    -------
    np.ndarray [m, size, size], dtype = np.float64
    """
    assert shift + 2 * (size - 1) <= m - 1, 'Not enough moments for this Hankel matrix.'
    B = np.zeros((m, size, size))
    for i in range(size):
        for j in range(size):
            B[i + j + shift, i, j] = 1.
    return B

def hankel_map(mu, shift = 0, size = None):
    """
    Shifted Hankel matrix H^{(shift),[size]}(mu) = [mu_{i+j+shift}]_{i,j=0}^{size-1} (Eqs. 20-21). By default
    the largest Hankel matrix the moments support.
    """
    m = len(mu)
    if size is None:
        size = (m - shift + 1) // 2
    return np.tensordot(mu, hankel_basis(m, shift, size), axes = 1)

def hankel_adjoint(M, shift, m):
    """
    Dual map h^{(shift)} : S^size -> R^m, defined by Tr[M H^{(shift)}(mu)] = mu . h^{(shift)}(M) (Eq. 36). The
    t-th entry is the sum of the antidiagonal i + j = t - shift of M.
    """
    return np.einsum('tij,ij->t', hankel_basis(m, shift, M.shape[0]), M)

def cone_constraints(m, cone = HAMBURGER):
    """
    Basis tensors of the linear matrix inequalities L_i(mu) = sum_t mu_t B_i[t] >= 0 defining the causal cone.

    Parameters
    ----------
    m : int
        Number of moments, m = 2d - 1.
    cone : str
        One of HAMBURGER, STIELTJES, HAUSDORFF.

    Returns
    -------
    list of np.ndarray [m, n_i, n_i]
    """
    assert m % 2 == 1 and m >= 3, 'The draft takes m = 2d - 1 odd; got m = ' + str(m)
    d = (m + 1) // 2
    constraints = [hankel_basis(m, 0, d)]
    if cone == STIELTJES:
        constraints.append(hankel_basis(m, 1, d - 1))
    elif cone == HAUSDORFF:
        constraints.append(hankel_basis(m, 1, d - 1) - hankel_basis(m, 2, d - 1))
    elif cone != HAMBURGER:
        raise ValueError('Unknown cone: ' + str(cone))
    return constraints

def in_cone(mu, cone = HAMBURGER, rtol = 1e-12):
    """
    Returns True if every LMI defining the cone holds at mu, up to rtol relative to the largest eigenvalue.
    """
    for B in cone_constraints(len(mu), cone):
        evals = np.linalg.eigvalsh(np.tensordot(mu, B, axes = 1))
        if evals[0] < -rtol * max(np.abs(evals).max(), 1e-300):
            return False
    return True

################################################################################
############################### MOREAU PROJECTION ##############################
################################################################################

class ConicProjection:
    """
    Result of the Moreau decomposition mu_hat = P_K(mu_hat) + P_Kpolar(mu_hat) (Eq. 92).

    Fields
    ------
    mu : np.ndarray [m]
        Causal component P_K(mu_hat).
    polar : np.ndarray [m]
        Acausal component P_Kpolar(mu_hat) = mu_hat - mu.
    duals : list of np.ndarray [n_i, n_i]
        Dual variables M_i >= 0, one per LMI, so that polar = -cov sum_i L_i^*(M_i) (Eqs. 79, 92c). Recovered on
        the vanishing eigenspaces of L_i(mu), so complementarity M_i L_i(mu) = 0 (Eq. 93) holds by construction.
    chi2 : float
        chi^2(mu_hat - mu) = chi^2_min, the distance to the cone (Eq. 134).
    gap : float
        Duality gap bound on chi^2 / 2 at termination, sum_i n_i / tau.
    ranks : list of int
        Rank of each LMI at the solution resolvable at the noise level of cov: eigenvalues below rank_tol times
        their sigma_eff count as zero. This is the dynamically determined N_states of Eq. (92b) for the Hankel LMI.
    converged : bool
    """
    mu        : np.ndarray
    polar     : np.ndarray
    duals     : list
    chi2      : float
    gap       : float
    ranks     : list
    converged : bool

    def __init__(self, mu, polar, duals, chi2, gap, ranks, converged):
        self.mu        = mu
        self.polar     = polar
        self.duals     = duals
        self.chi2      = chi2
        self.gap       = gap
        self.ranks     = ranks
        self.converged = converged

def _barrier_value(z, Bw):
    """
    -sum_i log det L_i(z) in whitened coordinates. Raises LinAlgError if some L_i(z) is not positive definite.
    """
    return -sum(2 * np.sum(np.log(np.diag(np.linalg.cholesky(np.tensordot(z, B, axes = 1))))) for B in Bw)

def _barrier(z, Bw):
    """
    Value, gradient and Hessian of -sum_i log det L_i(z) in whitened coordinates. Raises LinAlgError if some
    L_i(z) is not positive definite.
    """
    m = len(z)
    val, grad, hess = 0., np.zeros(m), np.zeros((m, m))
    for B in Bw:
        L = np.linalg.cholesky(np.tensordot(z, B, axes = 1))
        val -= 2 * np.sum(np.log(np.diag(L)))
        Linv = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower = True)
        # Y_s = L^{-1} B_s L^{-T}, so that Tr[X^{-1} B_s] = Tr Y_s and Tr[X^{-1} B_s X^{-1} B_t] = <Y_s, Y_t>_F.
        Y = Linv @ B @ Linv.T
        grad -= np.einsum('sii->s', Y)
        hess += np.einsum('sij,tij->st', Y, Y)
    return val, grad, hess

def _newton_direction(hess, grad, floor):
    """
    Solves hess dz = -grad for the symmetric positive-definite Newton matrix. Near the boundary the barrier part
    grows like 1/x^2 in the vanishing eigenvalues x of L_i and can exceed double precision, in which case Cholesky
    fails; the fallback inverts on the eigenbasis with the spectrum floored at the known lower bound tau.
    """
    try:
        return -scipy.linalg.cho_solve(scipy.linalg.cho_factor(hess), grad)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(hess)
        return -evecs @ ((evecs.T @ grad) / np.maximum(evals, floor))

def project(mu_hat, cov, cone = HAMBURGER, tol = 1e-7, mu0 = None, max_newton = 200, rank_tol = 1e-2):
    """
    Moreau projection of mu_hat onto the causal cone in the inner product <u, v> = u^T cov^{-1} v (Eqs. 70-74),
    i.e. the solution of Problem (82),
        minimize 1/2 chi^2(mu_hat - mu)  subject to  L_i(mu) >= 0 for each LMI of the cone.

    Parameters
    ----------
    mu_hat : np.ndarray [m]
        Noisy moment sequence (sample-mean correlator), m odd.
    cov : np.ndarray [m, m]
        Covariance defining the metric. For statistical statements this is Sigma_N, the covariance of mu_hat.
    cone : str
        HAMBURGER, STIELTJES or HAUSDORFF.
    tol : float
        Target duality gap on chi^2 / 2.
    mu0 : np.ndarray [m] or None
        Strictly feasible starting point. Defaults to mu_hat itself when it is strictly causal, and otherwise to
        mu_hat[0] times the moments of the uniform measure on [0, 1], which is interior to all three cones.
    max_newton : int
        Maximum Newton iterations per centering step.
    rank_tol : float
        An eigenvalue of L_i(mu) counts as zero when it is below rank_tol times its own noise scale sigma_eff
        (Eq. 108). Sets ConicProjection.ranks and the support of the recovered duals.

    Returns
    -------
    ConicProjection
    """
    mu_hat = np.asarray(mu_hat, dtype = np.float64)
    m = len(mu_hat)
    B_list = cone_constraints(m, cone)
    R = np.linalg.cholesky(cov)                             # cov = R R^T; whitened coordinates z = R^{-1} mu
    Bw = [np.einsum('ts,tij->sij', R, B) for B in B_list]
    z_hat = scipy.linalg.solve_triangular(R, mu_hat, lower = True)
    if mu0 is None:
        try:
            _barrier_value(z_hat, Bw)
            mu0 = mu_hat                                     # strictly causal data: start at the answer
        except np.linalg.LinAlgError:
            mu0 = (abs(mu_hat[0]) if mu_hat[0] != 0 else 1.) / np.arange(1, m + 1)
    z = scipy.linalg.solve_triangular(R, mu0, lower = True)
    _barrier_value(z, Bw)                                    # raises if mu0 is not strictly feasible

    nu = sum(B.shape[1] for B in B_list)                    # barrier parameter
    tau = nu / max(0.5 * np.sum((z - z_hat)**2), 1.)
    converged, outer = False, 0
    while True:
        # Centering: Newton on f(z) = tau/2 |z - z_hat|^2 - sum log det L_i(z), backtracking inside the cone.
        for _ in range(max_newton):
            bval, bgrad, bhess = _barrier(z, Bw)
            f = 0.5 * tau * np.sum((z - z_hat)**2) + bval
            grad = tau * (z - z_hat) + bgrad
            dz = _newton_direction(tau * np.eye(m) + bhess, grad, tau)
            dec2 = -grad @ dz                                # squared Newton decrement, affine invariant
            if dec2 / 2 < 1e-9:
                break
            alpha, accepted = 1., False
            while alpha > 1e-9:
                try:
                    fnew = 0.5 * tau * np.sum((z + alpha * dz - z_hat)**2) + _barrier_value(z + alpha * dz, Bw)
                    if fnew <= f - 0.25 * alpha * dec2:
                        accepted = True
                        break
                except np.linalg.LinAlgError:
                    pass
                alpha *= 0.5
            if not accepted:                                 # rounding floor of f reached: centered as well as
                break                                        # double precision allows
            z = z + alpha * dz
        outer += 1
        if nu / tau < tol:
            converged = True
            break
        if outer > 40:
            break
        tau *= 10.

    mu = R @ z
    polar = mu_hat - mu
    nulls = [_vanishing_subspace(mu, cov, B, rank_tol) for B in B_list]
    duals = _recover_duals(polar, cov, B_list, nulls)
    ranks = [B.shape[1] - N.shape[1] for B, N in zip(B_list, nulls)]
    return ConicProjection(mu, polar, duals, float(np.sum((z - z_hat)**2)), nu / tau, ranks, converged)

def _vanishing_subspace(mu, cov, B, rank_tol):
    """
    Orthonormal basis of the eigenvectors v of L(mu) whose eigenvalue x is statistically indistinguishable from zero,
    x < rank_tol * sigma_eff(v) with sigma_eff(v)^2 = L^*(v v^T)^T cov L^*(v v^T), the noise scale of Eq. (108). In
    double precision the barrier stalls with boundary eigenvalues far above rounding but far below their noise, so
    a noise-relative cut separates them where a relative-to-largest cut does not.
    """
    evals, evecs = np.linalg.eigh(np.tensordot(mu, B, axes = 1))
    h = np.einsum('tij,ik,jk->kt', B, evecs, evecs)         # h[k] = L^*(v_k v_k^T)
    sigma = np.sqrt(np.einsum('kt,ts,ks->k', h, cov, h))
    return evecs[:, evals < rank_tol * sigma]

def _recover_duals(polar, cov, B_list, nulls, iters = 5000):
    """
    Dual variables M_i = N_i S_i N_i^T >= 0 supported on the vanishing eigenspaces (complementarity, Eq. 93), with
    S_i >= 0 fitted to stationarity sum_i L_i^*(M_i) = -cov^{-1} polar (Eq. 92c) in the cov metric. M is not unique,
    since h has a kernel, and only some representatives are PSD; accelerated projected gradient (eigenvalue
    clipping) selects one. This avoids M = eps L(mu)^{-1}, which inverts eigenvalues the barrier cannot resolve in
    double precision.
    """
    duals = [np.zeros((B.shape[1], B.shape[1])) for B in B_list]
    if all(N.shape[1] == 0 for N in nulls):
        return duals
    R = np.linalg.cholesky(cov)
    target = -scipy.linalg.solve_triangular(R, polar, lower = True)            # whitened: R^T sum L^*(M) = target
    # A[i] maps S_i (k_i x k_i) to R^T L_i^*(N_i S_i N_i^T).
    A = [R.T @ np.einsum('tij,ia,jb->tab', B, N, N).reshape(len(polar), -1) for B, N in zip(B_list, nulls)]
    Lip = np.linalg.norm(np.hstack(A), 2)**2
    k = [N.shape[1] for N in nulls]
    S = [np.zeros((ki, ki)) for ki in k]
    Yv, theta = [Si.copy() for Si in S], 1.
    for _ in range(iters):
        r = sum(Ai @ Yi.ravel() for Ai, Yi in zip(A, Yv)) - target
        S_new = []
        for Ai, Yi, ki in zip(A, Yv, k):
            G = Yi - (Ai.T @ r).reshape(ki, ki) / Lip
            evals, evecs = np.linalg.eigh(0.5 * (G + G.T))
            S_new.append((evecs * np.maximum(evals, 0.)) @ evecs.T)
        theta_new = 0.5 * (1 + np.sqrt(1 + 4 * theta**2))
        Yv = [Sn + (theta - 1) / theta_new * (Sn - So) for Sn, So in zip(S_new, S)]
        S, theta = S_new, theta_new
    return [N @ Si @ N.T for N, Si in zip(nulls, S)]

def moreau_residuals(mu_hat, cov, res, cone = HAMBURGER):
    """
    Diagnostics for a ConicProjection: how well the computed point satisfies the Moreau / KKT conditions.

    Returns
    -------
    dict with keys
        'orthogonality'   : <P_K, P_Kpolar> / (|P_K| |P_Kpolar|) in the cov^{-1} metric (0 at the solution, Eq. 75).
        'stationarity'    : |polar + cov sum_i L_i^*(M_i)|_{cov^{-1}} (0 at the solution, Eq. 92c).
        'complementarity' : sum_i Tr[M_i L_i(mu)], which equals the gap bound on the central path (Eq. 93).
        'min_dual_eig'    : smallest eigenvalue over the M_i (>= 0 for dual feasibility).
        'min_primal_eig'  : smallest eigenvalue over the L_i(mu) (>= 0 for primal feasibility).
    """
    G = np.linalg.inv(cov)
    B_list = cone_constraints(len(mu_hat), cone)
    hM = sum(np.einsum('tij,ij->t', B, M) for B, M in zip(B_list, res.duals))
    r = res.polar + cov @ hM
    ip = res.mu @ G @ res.polar
    norms = np.sqrt((res.mu @ G @ res.mu) * (res.polar @ G @ res.polar))
    return {
        'orthogonality'   : ip / norms if norms > 0 else 0.,
        'stationarity'    : np.sqrt(r @ G @ r),
        'complementarity' : sum(np.trace(M @ np.tensordot(res.mu, B, axes = 1)) for B, M in zip(B_list, res.duals)),
        'min_dual_eig'    : min(np.linalg.eigvalsh(M)[0] for M in res.duals),
        'min_primal_eig'  : min(np.linalg.eigvalsh(np.tensordot(res.mu, B, axes = 1))[0] for B in B_list),
    }

################################################################################
############################## BOUNDARY STRUCTURE ##############################
################################################################################

def numerical_rank(A, rtol = 1e-10):
    """
    Number of eigenvalues of the Hermitian matrix A larger than rtol times the largest in magnitude.
    """
    evals = np.linalg.eigvalsh(A)
    return int(np.sum(evals > rtol * np.abs(evals).max()))

def rank_sequence(mu, rtol = 1e-10):
    """
    Ranks (r_1, ..., r_d) of the leading Hankel submatrices H^{[k]}(mu) (Eq. 62). By Lemma 1, a representable
    singular scalar sequence has r_1 < ... < r_k = ... = r_d.
    """
    d = (len(mu) + 1) // 2
    return [numerical_rank(hankel_map(mu[:2 * k - 1], 0, k), rtol) for k in range(1, d + 1)]

def endpoint_defect(mu, rtol = 1e-10):
    """
    Splits H^{[d]}(mu) = H^{[d]}_flat + Delta with Delta = diag(0, ..., 0, mu_Delta) (Eqs. 64, 67-68), by building
    the flat extension of H^{[d-1]} (Eqs. 46-49) from all but the last moment.

    Returns
    -------
    (extendable, mu_delta) : (bool, float)
        extendable is False when ker H^{[d-1]} is not contained in ker mu_tilde^T (Eq. 48), in which case no flat
        extension exists and mu_delta is None. Otherwise mu_delta = mu_{m-1} - mu_tilde^T H^{[d-1]+} mu_tilde.
        mu_delta is the Schur complement of H^{[d]}: it is only a "defect" when H^{[d]} is singular.
    """
    m = len(mu)
    d = (m + 1) // 2
    Hk = hankel_map(mu[:m - 2], 0, d - 1)
    mu_tilde = mu[d - 1 : m - 1]
    evals, evecs = np.linalg.eigh(Hk)
    null = evecs[:, evals <= rtol * np.abs(evals).max()]
    if null.size and np.abs(null.T @ mu_tilde).max() > np.sqrt(rtol) * np.abs(mu_tilde).max():
        return False, None
    mu_flat = mu_tilde @ np.linalg.pinv(Hk, rcond = rtol, hermitian = True) @ mu_tilde
    return True, float(mu[m - 1] - mu_flat)

def spectral_decomposition(mu, rank = None, rtol = 1e-10):
    """
    Finitely atomic decomposition mu_t = sum_n w_n lambda_n^t of the flat part of a boundary sequence (Eq. 66).
    The atoms are the generalized eigenvalues of (H^{(1),[r]}, H^{[r]}), computed with QZ rather than a Cholesky
    reduction, and the weights come from a least-squares Vandermonde solve on the first 2r moments.

    Parameters
    ----------
    mu : np.ndarray [m]
    rank : int or None
        Number of atoms r. Defaults to the numerical rank of H^{[d-1]}, which excludes the endpoint moment.

    Returns
    -------
    (lam, w) : np.ndarray [r], np.ndarray [r], sorted by decreasing lambda. Complex values signal that mu is not
    representable to the requested rank.
    """
    m = len(mu)
    d = (m + 1) // 2
    r = numerical_rank(hankel_map(mu[:m - 2], 0, d - 1), rtol) if rank is None else rank
    lam = scipy.linalg.eig(hankel_map(mu, 1, r), hankel_map(mu, 0, r), right = False)
    if np.allclose(lam.imag, 0.):
        lam = lam.real
    lam = lam[np.argsort(-lam.real)]
    V = np.vander(lam, 2 * r, increasing = True).T          # V[t, n] = lambda_n^t
    w = np.linalg.lstsq(V, mu[:2 * r], rcond = None)[0]
    return lam, w

################################################################################
######################### STATISTICS: SECTION IV B #############################
################################################################################

def violation_bound(mu, cov, N):
    """
    Critical sample size and the lower bound on the probability of an acausal sample mean (Eqs. 103-117), for an
    exact causal Hamburger sequence mu with H(mu) > 0.

    Parameters
    ----------
    mu : np.ndarray [m]
        Exact moment sequence.
    cov : np.ndarray [m, m]
        Single-sample covariance Sigma, so that mu_hat_N ~ N(mu, Sigma / N).
    N : int
        Number of samples.

    Returns
    -------
    dict with keys 'lam_min', 'lam_max', 'kappa', 'sigma2_eff' (Eq. 108), 'N_crit' (Eq. 116),
    'prob_violation_lower' (Eq. 114) and 'beckermann_kappa_lower' (Eq. 29).
    """
    m = len(mu)
    d = (m + 1) // 2
    evals, evecs = np.linalg.eigh(hankel_map(mu))
    v = evecs[:, 0]
    h_lam = hankel_adjoint(np.outer(v, v), 0, m)            # h(Lambda_min), Eq. (105)
    sigma2_eff = h_lam @ cov @ h_lam
    return {
        'lam_min'                : evals[0],
        'lam_max'                : evals[-1],
        'kappa'                  : evals[-1] / evals[0],
        'sigma2_eff'             : sigma2_eff,
        'N_crit'                 : sigma2_eff / evals[0]**2,
        'prob_violation_lower'   : scipy.stats.norm.cdf(-np.sqrt(N) * evals[0] / np.sqrt(sigma2_eff)),
        'beckermann_kappa_lower' : BECKERMANN_GAMMA**(d - 1) / (16 * d) if d >= 3 else None,
    }

################################################################################
######################### STATISTICS: SECTION V E ##############################
################################################################################

# Profile chi^2 for a kernel in the measured span, rho[K] = g^T mu (Eq. 138). The constrained projection onto
# K_q = K cap {g^T mu = q} (Eq. 133) never needs an equality constraint: minimizing 1/2 chi^2(mu_hat - mu) - s g^T mu
# over K is projecting the shifted point mu_hat + s cov g, and the minimizer mu(s) solves the K_q problem for
# q = g^T mu(s). Since P_K is monotone, q(s) is nondecreasing in s, and Delta chi^2 grows with |s| on either side
# of s = 0. Without the cone this reproduces Eq. (139).

def profile_point(mu_hat, cov, g, s, cone = HAMBURGER, **kwargs):
    """
    Constrained projection at Lagrange multiplier s. Returns (q, chi2_q, ConicProjection).
    """
    res = project(mu_hat + s * (cov @ g), cov, cone, **kwargs)
    r = mu_hat - res.mu
    return g @ res.mu, float(r @ np.linalg.solve(cov, r)), res

def _s_for_q(mu_hat, cov, g, q, cone, **kwargs):
    """
    Multiplier s with g^T mu(s) = q, by bracketing and Brent's method.
    """
    q0 = profile_point(mu_hat, cov, g, 0., cone, **kwargs)[0]
    step = np.sign(q - q0) / np.sqrt(g @ cov @ g) if q != q0 else 0.
    if step == 0.:
        return 0.
    s_far = step * (abs(q - q0) / np.sqrt(g @ cov @ g) + 1.)
    for _ in range(60):
        if np.sign(profile_point(mu_hat, cov, g, s_far, cone, **kwargs)[0] - q) != np.sign(q0 - q):
            break
        s_far *= 2.
    return scipy.optimize.brentq(lambda s : profile_point(mu_hat, cov, g, s, cone, **kwargs)[0] - q,
                                 0., s_far, xtol = 1e-12 * abs(s_far))

def delta_chi2(mu_hat, cov, g, q, cone = HAMBURGER, **kwargs):
    """
    Delta chi^2_q = chi^2_q - chi^2_0 (Eqs. 134-136).
    """
    chi2_0 = project(mu_hat, cov, cone, **kwargs).chi2
    s = _s_for_q(mu_hat, cov, g, q, cone, **kwargs)
    return profile_point(mu_hat, cov, g, s, cone, **kwargs)[1] - chi2_0

def causal_bounds(mu_hat, cov, g, cone = HAMBURGER, c = None, alpha = 0.05, **kwargs):
    """
    Interval {q : Delta chi^2_q <= c} for rho[K] = g^T mu (Eq. 137). With c = None the cutoff is the asymptotic
    (1 - alpha) quantile of chi^2_1, appropriate for an interior true sequence (Eq. 140 and below).

    Returns
    -------
    (q_lo, q_star, q_hi) : the endpoints and the value g^T P_K(mu_hat) at the projection.
    """
    if c is None:
        c = scipy.stats.chi2.ppf(1 - alpha, 1)
    q_star, chi2_0, _ = profile_point(mu_hat, cov, g, 0., cone, **kwargs)
    f = lambda s : profile_point(mu_hat, cov, g, s, cone, **kwargs)[1] - chi2_0 - c
    s_unit = np.sqrt(c / (g @ cov @ g))                     # the exact answer without the cone
    ends = []
    for sgn in (-1., 1.):
        s_far = sgn * s_unit
        for _ in range(60):
            if f(s_far) > 0:
                break
            s_far *= 2.
        s_end = scipy.optimize.brentq(f, 0., s_far, xtol = 1e-12 * abs(s_far))
        ends.append(profile_point(mu_hat, cov, g, s_end, cone, **kwargs)[0])
    return ends[0], q_star, ends[1]

def calibrate_cutoff(mu_hat, cov, g, q, cone = HAMBURGER, alpha = 0.05, n_rep = 200, rng = None, **kwargs):
    """
    Constrained parametric bootstrap for the cutoff c_{1-alpha}(q) (Sec. V E): draw replicas
    mu_hat^(b) ~ N(mu_q, cov) about mu_q = P_{K_q}(mu_hat) and take the (1 - alpha) quantile of Delta chi^2_q.

    Returns
    -------
    (c_hat, samples) : float, np.ndarray [n_rep]
    """
    rng = np.random.default_rng() if rng is None else rng
    s = _s_for_q(mu_hat, cov, g, q, cone, **kwargs)
    mu_q = profile_point(mu_hat, cov, g, s, cone, **kwargs)[2].mu
    reps = rng.multivariate_normal(mu_q, cov, size = n_rep)
    samples = np.array([delta_chi2(rep, cov, g, q, cone, **kwargs) for rep in reps])
    return np.quantile(samples, 1 - alpha), samples
