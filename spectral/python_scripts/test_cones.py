"""
Tests for cones.py on Will's toy correlators (will_scripts/hamburger_mp.py). Run with `pytest test_cones.py` from
python_scripts/. Scalar (p = 1) data is the (0, 0) component of each model.
"""

import os
import sys
import numpy as np
import scipy.optimize
import scipy.stats
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'will_scripts'))
import hamburger_mp as will
import cones

CONES = [cones.HAMBURGER, cones.STIELTJES, cones.HAUSDORFF]

def scalar_correlator(model):
    """
    (0, 0) component of one of Will's models as a float64 moment sequence.
    """
    return np.array([float(c[0, 0]) for c in model.correlator])

def relative_covariance(mu, eps, rho = 0.5):
    """
    Toy covariance of the sample mean: relative error eps on each timeslice, correlated as rho^|t - t'|.
    """
    t = np.arange(len(mu))
    return eps**2 * np.outer(mu, mu) * rho**np.abs(t[:, None] - t[None, :])

@pytest.fixture
def three_state():
    """
    Exact 3-state data on m = 11 timeslices: rank 3 < d = 6, so a boundary point of every cone.
    """
    return scalar_correlator(will.ToyModel(11, 3))

@pytest.fixture
def interior():
    """
    Exact 6-state data on m = 7 timeslices: rank 4 = d, so an interior point.
    """
    return scalar_correlator(will.ToyModel(7, 6))

def acausal_sample(mu, cov, cone, seed = 1):
    """
    First Gaussian draw about mu that lies outside the cone.
    """
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        mu_hat = rng.multivariate_normal(mu, cov)
        if not cones.in_cone(mu_hat, cone):
            return mu_hat
    raise RuntimeError('No acausal sample drawn.')

################################################################################
################################# LINEAR MAPS ##################################
################################################################################

@pytest.mark.parametrize('shift', [0, 1, 2])
def test_adjoint(shift):
    rng = np.random.default_rng(0)
    m = 11
    size = (m - shift + 1) // 2
    mu = rng.normal(size = m)
    M = rng.normal(size = (size, size))
    M = M + M.T
    lhs = np.trace(M @ cones.hankel_map(mu, shift, size))
    assert np.isclose(lhs, mu @ cones.hankel_adjoint(M, shift, m))           # Eq. (36)

def test_hankel_uses_every_moment():
    mu = np.arange(7.)
    assert cones.hankel_map(mu)[-1, -1] == 6.                                 # m = 2d - 1: H^{[d]} reaches mu_{m-1}

################################################################################
############################### MOREAU PROJECTION ##############################
################################################################################

@pytest.mark.parametrize('cone', CONES)
def test_exact_data_is_fixed_point(three_state, cone):
    cov = relative_covariance(three_state, 1e-3)
    res = cones.project(three_state, cov, cone)
    assert res.converged
    assert res.chi2 < 1e-6                                                     # Eq. (78)
    assert cones.numerical_rank(cones.hankel_map(three_state), 1e-12) == 3
    assert res.ranks[0] <= 3                                                   # third state is below 1e-3 noise

@pytest.mark.parametrize('cone', CONES)
def test_moreau_decomposition(three_state, cone):
    cov = relative_covariance(three_state, 1e-3)
    mu_hat = acausal_sample(three_state, cov, cone)
    res = cones.project(mu_hat, cov, cone)
    diag = cones.moreau_residuals(mu_hat, cov, res, cone)
    assert res.converged and res.chi2 > 0
    assert abs(diag['orthogonality']) < 1e-4                                   # Eq. (75)
    assert diag['stationarity'] < 1e-4 * np.sqrt(res.chi2)                    # Eq. (92c)
    assert abs(diag['complementarity']) < 1e-6 * res.chi2                      # Eq. (93)
    assert diag['min_dual_eig'] >= -1e-8 * max(np.abs(M).max() for M in res.duals)
    assert diag['min_primal_eig'] >= 0
    assert res.ranks[0] < (len(mu_hat) + 1) // 2                              # projected onto the boundary, Eq. (76)
    again = cones.project(res.mu, cov, cone)
    assert again.chi2 < 1e-6 * res.chi2                                        # idempotence

def test_cone_nesting(three_state):
    cov = relative_covariance(three_state, 1e-3)
    mu_hat = acausal_sample(three_state, cov, cones.HAMBURGER)
    chi2 = [cones.project(mu_hat, cov, cone).chi2 for cone in CONES]
    assert chi2[0] <= chi2[1] * (1 + 1e-6) and chi2[1] <= chi2[2] * (1 + 1e-6)

def test_hausdorff_matches_nnls(three_state):
    """
    The Hausdorff cone is the conic hull of (1, lam, ..., lam^{m-1}), lam in [0, 1], so covariance-weighted NNLS
    over a fine grid of lam is an inner approximation that should agree with the interior-point projection.
    """
    cov = relative_covariance(three_state, 1e-3)
    mu_hat = acausal_sample(three_state, cov, cones.HAUSDORFF)
    res = cones.project(mu_hat, cov, cones.HAUSDORFF)
    R = np.linalg.cholesky(cov)
    lam = np.linspace(0., 1., 4001)
    A = np.linalg.solve(R, np.vander(lam, len(mu_hat), increasing = True).T)
    w, rnorm = scipy.optimize.nnls(A, np.linalg.solve(R, mu_hat), maxiter = 10000)
    assert res.chi2 <= rnorm**2 * (1 + 1e-6)
    assert rnorm**2 <= res.chi2 * (1 + 1e-2)

################################################################################
############################## BOUNDARY STRUCTURE ##############################
################################################################################

def test_scalar_endpoint_defect():
    """
    mu = (1, 1, 1, 1, 2): PSD Hankel, no representing measure (the v2 counterexample to Lemma 1). It lies in the
    Hamburger cone with a unit endpoint defect, and outside the Hausdorff cone because mu_3 - mu_4 < 0.
    """
    mu = np.array([1., 1., 1., 1., 2.])
    cov = np.eye(5)
    assert cones.in_cone(mu, cones.HAMBURGER)
    assert cones.rank_sequence(mu) == [1, 1, 2]
    extendable, mu_delta = cones.endpoint_defect(mu)
    assert extendable and np.isclose(mu_delta, 1.)
    assert cones.project(mu, cov, cones.HAMBURGER).chi2 < 1e-8
    assert not cones.in_cone(mu, cones.HAUSDORFF)
    assert cones.project(mu, cov, cones.HAUSDORFF).chi2 > 1e-2

def test_spectral_decomposition(three_state):
    lam, w = cones.spectral_decomposition(three_state)
    assert len(lam) == 3
    assert np.allclose(lam, np.exp(-0.1 * np.arange(1, 4)), rtol = 1e-6)
    assert np.allclose(w @ np.vander(lam, 11, increasing = True), three_state, rtol = 1e-8)
    assert np.isclose(cones.endpoint_defect(three_state)[1], 0., atol = 1e-10 * three_state[0])

def test_projection_of_noisy_data_is_atomic(three_state):
    cov = relative_covariance(three_state, 1e-3)
    mu_hat = acausal_sample(three_state, cov, cones.HAUSDORFF)
    res = cones.project(mu_hat, cov, cones.HAUSDORFF)
    lam, w = cones.spectral_decomposition(res.mu, rank = res.ranks[0])
    assert np.all(np.isreal(lam)) and np.all((lam > -1e-8) & (lam < 1 + 1e-8)) and np.all(w > 0)
    fit = w @ np.vander(lam, len(mu_hat), increasing = True)
    assert np.allclose(fit, res.mu, rtol = 1e-4)

################################################################################
############################ STATISTICS: SECTION IV B ##########################
################################################################################

def test_violation_bound_is_a_lower_bound(interior):
    Sigma = relative_covariance(interior, 1e-1)
    vb = cones.violation_bound(interior, Sigma, 1)
    assert vb['lam_min'] > 0
    N = int(vb['N_crit'])
    bound = cones.violation_bound(interior, Sigma, N)['prob_violation_lower']
    rng = np.random.default_rng(2)
    draws = rng.multivariate_normal(interior, Sigma / N, size = 4000)
    freq = np.mean([np.linalg.eigvalsh(cones.hankel_map(x))[0] < 0 for x in draws])
    assert freq >= bound - 3 * np.sqrt(bound * (1 - bound) / len(draws))    # Eq. (112)

################################################################################
############################ STATISTICS: SECTION V E ###########################
################################################################################

def test_profile_closed_form_in_interior(interior):
    """
    When mu_hat and mu_q both lie in K, Delta chi^2_q = (g^T mu_hat - q)^2 / (g^T Sigma_N g) (Eq. 140).
    """
    cov = relative_covariance(interior, 1e-6)
    g = np.exp(-0.3 * np.arange(len(interior)))
    q_hat = g @ interior
    for q in q_hat * (1 + np.array([-2e-6, 1e-6, 3e-6])):
        exact = (q_hat - q)**2 / (g @ cov @ g)
        assert np.isclose(cones.delta_chi2(interior, cov, g, q), exact, rtol = 1e-4)
    lo, q_star, hi = cones.causal_bounds(interior, cov, g)
    half = np.sqrt(scipy.stats.chi2.ppf(0.95, 1) * (g @ cov @ g))
    assert np.isclose(q_star, q_hat) and np.isclose(lo, q_hat - half, rtol = 1e-8) and np.isclose(hi, q_hat + half, rtol = 1e-8)

def test_bounds_tighten_with_cone(three_state):
    cov = relative_covariance(three_state, 1e-3)
    mu_hat = acausal_sample(three_state, cov, cones.HAMBURGER)
    g = np.zeros(len(mu_hat)); g[0] = 1.
    lo_h, _, hi_h = cones.causal_bounds(mu_hat, cov, g, cones.HAMBURGER)
    lo_d, _, hi_d = cones.causal_bounds(mu_hat, cov, g, cones.HAUSDORFF)
    assert lo_h <= lo_d * (1 + 1e-6) and hi_d <= hi_h * (1 + 1e-6)

def test_calibrated_cutoff_near_chi2_1(interior):
    cov = relative_covariance(interior, 1e-6)
    g = np.exp(-0.3 * np.arange(len(interior)))
    c_hat, samples = cones.calibrate_cutoff(interior, cov, g, g @ interior, alpha = 0.05, n_rep = 40,
                                            rng = np.random.default_rng(3))
    assert np.all(samples >= -1e-6)
    assert 1. < c_hat < 10.                                                    # chi^2_1 quantile is 3.84
