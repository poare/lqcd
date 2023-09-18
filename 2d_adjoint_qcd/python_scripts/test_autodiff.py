################################################################################
# Playground for testing Jax's autodiff and how it works with complex input.   # 
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

import numpy as np
import itertools
import scipy
from scipy.sparse import bsr_matrix, csr_matrix
from jax.scipy.linalg import expm
import time

# for complex autodiff
import jax.numpy as jnp
from jax import grad, jit, vmap

from jax import random
key = random.PRNGKey(0)

import rhmc
import jax_rhmc as jrhmc

############################################################
########################## SETUP ###########################
############################################################

# TODO: run with this, then run with the next set of parameters. Will likely take a while...
Nc = 2
L, T = 4, 4
# L, T = 2, 4

# Nc = 3
# L, T = 6, 10

dNc = Nc**2 - 1
Lat = jrhmc.Lattice(L, T)
Lat_np = rhmc.Lattice(L, T)

# generate \omega_\mu^a(n)
d = 2
beta = 1.0
key = random.PRNGKey(0)
omega = random.uniform(key, shape = (d, dNc, L, T))
gens = jrhmc.get_generators(Nc)
U = jrhmc.get_fund_field(omega, gens)

############################################################
######################## UTILITIES #########################
############################################################

def print_line():
    print('-'*50)

def is_equal_jax(A, B, eps = rhmc.EPS):
    """Returns whether two jax arrays are equal up to precision eps."""
    return jnp.allclose(A, B, eps)

def lie_derivative(f, omega0, generators = gens):
    """
    Regular autodifferentiation of a function f(U) returns the derivative of f with 
    respect to the coordinates of U. This does not transform covariantly under gauge 
    transformations of U. Instead, a Lie derivative of f must be taken, which does 
    transform covariantly. In terms of the coordinates U_\mu(n) = exp(i\omega^a_\mu(n) t^a) 
    \omega parameterizing U, the Lie derivative may be computed as follows:
    $$
        \nabla^a f(U) = \frac{d}{dw} [ f(exp(iwt^a)U) ]_{w = 0}.
    $$
    Note here that U is held fixed. This function evaluates the Lie derivative of f 
    at omega0 using jax's autodiff implementation.

    Parameters
    ----------
    f : gauge field --> \mathbb{R}
        Function of the gauge field U (jax numpy array of shape [d, L, T, Nc, Nc]) to differentiate. 
        Note this must be real-valued, but one can differentiate a vector-valued or complex-valued 
        by indidividually calling lie_derivative on each component. 
    omega0 : jnp.array [d, dNc, L, T]
        Coordinates of gauge field to differentiate with respect to, as a jax numpy array.
    generators : np.array [dNc, Nc, Nc] (default = gens)
        Generators of SU(Nc)
    
    Returns
    -------
    deriv : jnp.array [d, dNc, L, T]
        Lie derivative of f with respect to \omega_\mu^a(n). 
    """
    U = jrhmc.get_fund_field(omega0, generators)
    def f_auxiliary(omega):
        phase = jrhmc.get_fund_field(omega, generators)
        phase_U = jnp.einsum('mxtij,mxtjk->mxtik', phase, U)
        return f(phase_U)
    derivative_f = grad(f_auxiliary)
    return derivative_f(jnp.zeros(omega.shape))

############################################################
######################## TEST TR U #########################
############################################################

print('Testing derivative of Re Tr U.')

def trU(omega):
    """Returns Re Tr U_0(0). """
    U = jrhmc.get_fund_field(omega, gens)
    return jnp.real(jnp.trace(U[0, 0, 0]))
grad_trU = grad(trU)
grad_trU_omega = jnp.asarray(grad_trU(omega)[0, :, 0, 0])

# U
grad_trU_analytic = -np.imag(np.einsum('aij,ji->a', gens, U[0, 0, 0]))

print(grad_trU_omega - grad_trU_analytic)
assert np.allclose(grad_trU_omega, grad_trU_analytic), 'Derivative of Re Tr U_0(0) is incorrect.'

############################################################
####################### TEST TR U0 A #######################
############################################################

print('Testing derivative of Re Tr U0 A.')

np.random.seed(1)
A_list = [[np.random.rand(), np.random.rand()], [np.random.rand(), np.random.rand()]]
A_np = np.array(A_list)
A_jax = jnp.array(A_list)

# THE BELOW CODE IS WRONG; IT IMPLEMENTS A STANDARD DERIVATIVE INSTEAD OF A COVARIANT DERIVATIVE
# def trUA(omega):
#     U = jrhmc.get_fund_field(omega, gens)
#     return jnp.real(jnp.trace(U[0, 0, 0] @ A_jax))
# grad_trUA = grad(trUA)
# grad_trUA_omega = jnp.asarray(grad_trUA(omega)[0, :, 0, 0])
# print(grad_trUA_omega)

# THIS CODE WORKS: TIDY IT UP, THOUGH
# def get_trUA(omega0, a):
#     # USE THE LIE DERIVATIVE
#     U = expm(1j*jnp.einsum('a,aij->ij', omega0[0, :, 0, 0], gens))
#     def trUA(t):
#         phase = expm(1j*t*gens[a])
#         return jnp.real(jnp.trace(phase @ U @ A_jax))
#     return trUA
# for a in range(dNc):
#     trUA = get_trUA(omega, a)
#     grad_trUA = grad(trUA)
#     # grad_trUA_omega = jnp.asarray(grad_trUA(0)[0, :, 0, 0])
#     print(f'grad_{a} = {grad_trUA(0.0)}')

# TIDIED UP CODE
def f(U):
    """Function of U to differentiate. """
    return jnp.real(jnp.trace(U @ A_jax))
def df(omega0):
    """
    To evaluate the Lie derivative \partial_a of a function f(U) for fixed U = exp(i\omega0 t), we 
    need to evaluate the derivative \partial / \partial \omega f(e^{i\omega t_a} U)|_{\omega = 0}. 
    """
    U = expm(1j*jnp.einsum('a,aij->ij', omega0[0, :, 0, 0], gens))
    def f_auxiliary(t):
        phase = expm(1j* jnp.einsum('a,aij->ij', t, gens))
        return f(phase @ U)
    derivative_f = grad(f_auxiliary)
    return derivative_f(jnp.zeros((dNc)))
grad_trUA0 = df(omega)
print(grad_trUA0)

grad_trUA_analytic = -np.imag(np.einsum('aij,jk,ki->a', gens, U[0, 0, 0], A_np))
# grad_trUA_analytic = - np.pi * np.imag(np.einsum('aij,ji->a', gens, U[0, 0, 0]))
print(grad_trUA_analytic)

assert np.allclose(grad_trU_omega, grad_trU_analytic, rtol = 1e-4), 'Derivative of Re Tr [UA] is incorrect.'

############################################################
#################### TEST PLAQUETTE AT 0 ###################
############################################################

print('Testing plaquette at 0')

def p0(U):
    """ Function of the gauge field U to differentiate. """
    return jrhmc.plaquette(U)[0, 0]

def dp0(omega0):
    """
    To evaluate the Lie derivative \partial_a of a function f(U) for fixed U = exp(i\omega0 t), we 
    need to evaluate the derivative \partial / \partial \omega f(e^{i\omega t_a} U)|_{\omega = 0}. 
    """
    # U = expm(1j*jnp.einsum('maxt,aij->mxtij', omega0, gens))
    U = jrhmc.get_fund_field(omega0, gens)
    def f_auxiliary(omega):
        # phase = expm(1j* jnp.einsum('maxt,aij->mxtij', omega, gens))
        phase = jrhmc.get_fund_field(omega, gens)
        phase_U = jnp.einsum('mxtij,mxtjk->mxtik', phase, U)
        return p0(phase_U)
    derivative_f = grad(f_auxiliary)
    return derivative_f(jnp.zeros(omega.shape))
grad_trp0 = dp0(omega)
print(grad_trp0)

# Here we have Tr P = Tr [U0 U1 U2 U3]
U0, U1, U2, U3 = U[0, 0, 0], U[1, 1, 0], rhmc.dagger(U[0, 0, 1]), rhmc.dagger(U[1, 0, 0])
grad_p0_analytic = np.zeros(omega.shape, dtype = np.float64)
grad_p0_analytic[0, :, 0, 0] = -(1 / Nc) * np.imag(np.einsum('aij,jk,kl,lm,mi->a', gens, U0, U1, U2, U3))
grad_p0_analytic[1, :, 1, 0] = -(1 / Nc) * np.imag(np.einsum('aij,jk,kl,lm,mi->a', gens, U1, U2, U3, U0))
grad_p0_analytic[0, :, 0, 1] = (1 / Nc) * np.imag(np.einsum('aij,jk,kl,lm,mi->a', gens, U3, U0, U1, U2))
grad_p0_analytic[1, :, 0, 0] = (1 / Nc) * np.imag(np.einsum('aij,jk,kl,lm,mi->a', gens, U0, U1, U2, U3))
# print('Analytic formula for dp0 / dw:')
# print(grad_p0_analytic)

assert np.allclose(grad_trp0, grad_p0_analytic, rtol = 1e-4), 'Derivative of Re Tr [P(0)] is incorrect.'

############################################################
#################### WILSON GAUGE ACTION ###################
############################################################

print('Differentiating Wilson gauge action')

# Autodiff Wilson action
def S_wilson(U):
    return jrhmc.wilson_gauge_action(U, beta, Nc)
grad_wilson = lie_derivative(S_wilson, omega)    # only takes 2-3 times as long as a function call
print('Autodiff for dS_g / dw')
print(grad_wilson)

# Implemented derivative
gens_np = rhmc.get_generators(Nc)
grad_wilson_analytic = np.real(rhmc.gauge_force_wilson(jnp.asarray(omega), gens_np, beta))
print('Analytic formula for dS_g / dw')
print(grad_wilson_analytic)

assert np.allclose(grad_wilson, grad_wilson_analytic, rtol = 1e-3), 'Derivative of Wilson gauge action is incorrect.'

############################################################
################## ADJOINT FIELD DERIVATIVE ################
############################################################

# random pseudofermion field for psi
Ns = 2
_, key = random.split(key, 2)
psi = random.uniform(key, shape = (dNc, Ns, L, T))

aa, bb = 0, 1
def g(U):
    V = jrhmc.construct_adjoint_links(U, gens, lat = Lat)
    return jnp.real(V[0, 0, 0, aa, bb])
grad_g = lie_derivative(g, omega)
print('Autodiff for dg/dw:')
print(grad_g)

# Test analytic expression
W = rhmc.form_W_tensor(jnp.asarray(U), gens_np, lat = Lat_np)

grad_g_analytic = np.zeros(omega.shape, dtype = np.float64)
grad_g_analytic[0, :, 0, 0] = -2*np.imag(W[0, 0, 0, :, aa, bb])
print('Analytic expression for dg/dw:')
print(grad_g_analytic)

assert np.allclose(grad_g_analytic, jnp.asarray(grad_g), rtol = 1e-3), 'Analytic expression for dg/dw does not match autodiff.'

# Test h(U)
def h(U):
    V = jrhmc.construct_adjoint_links(U, gens, lat = Lat)
    return jnp.real(jnp.einsum('txia,xtab,bixt->', psi.conj().transpose(), V[0] + V[1], psi))
grad_h = lie_derivative(h, omega)
print('Autodiff for dh/dw:')
print(grad_h)

grad_h_analytic = -2*np.imag(np.einsum('txib,mxtabc,cixt->maxt', psi.conj().transpose(), W, psi))
print('Analytic expression for dh/dw:')
print(grad_h_analytic)

assert np.allclose(grad_h_analytic, jnp.asarray(grad_h), rtol = 1e-3), 'Analytic expression for dh/dw does not match autodiff.'

############################################################
################### SUB-DIRAC OP DERIVATIVE ################
############################################################
print('Computing derivative for part of the Dirac operator.')

# ferm_bcs = (1, -1)
ferm_bcs = (1, 1)

def G(U):
    V = jrhmc.construct_adjoint_links(U, gens, lat = Lat)
    act = 0.0
    for mu in range(d):
        psi_shift = np.roll(psi, -1, axis = 2 + mu)
        act += jnp.real(jnp.einsum('txia,xtab,ij,bjxt->', psi.conj().transpose(), V[mu], jrhmc.delta - jrhmc.gamma[mu], psi_shift))
    return act

grad_G = lie_derivative(G, omega)
print('Autodiff for dG/dw:')
print(grad_G)

W = rhmc.form_W_tensor(jnp.asarray(U), gens_np, lat = Lat_np)
grad_G_analytic = np.zeros((omega.shape), dtype = np.float64)
for mu, a, nx, nt in itertools.product(range(d), range(dNc), range(Lat_np.L), range(Lat_np.T)):
    n = np.array([nx, nt])
    npmu = Lat_np.mod(n + rhmc.hat(mu))
    if np.abs(n[mu] - npmu[mu]) > 1:            # then we traversed the 0 boundary
        sign = ferm_bcs[mu]
    else:
        sign = 1
    psi_flat = rhmc.flatten_ferm_field(psi, lat = Lat_np)
    psi_n = rhmc.unflatten_colspin_vec(rhmc.flat_field_evalat(psi_flat, nx, nt, dNc, lat = Lat_np), dNc)
    psi_npmu = sign * rhmc.unflatten_colspin_vec(rhmc.flat_field_evalat(psi_flat, npmu[0], npmu[1], dNc, lat = Lat_np), dNc)
    psi_dagger_n = psi_n.conj().transpose()

    grad_G_analytic[mu, a, nx, nt] = -2*np.imag(np.einsum('ib,bc,ij,cj', psi_dagger_n, W[mu, nx, nt, a], rhmc.delta - rhmc.gamma[mu], psi_npmu))
    # grad_G_analytic[mu, a, x, t] = -2*np.imag(np.einsum('txib,mxtabc,mij,cjxt->maxt', psi.conj().transpose(), W, gamma_mat, psi))
print('Analytic expression for dG/dw:')
print(grad_G_analytic)

assert np.allclose(grad_G_analytic, jnp.asarray(grad_G), rtol = 1e-3), 'Analytic expression for dG/dw does not match autodiff.'

def H(U):
    V = jrhmc.construct_adjoint_links(U, gens, lat = Lat)
    act = 0.0
    for mu in range(d):
        psi_shift = np.roll(psi, -1, axis = 2 + mu)
        act += jnp.real(jnp.einsum('txia,xtba,ij,bjxt->', psi_shift.conj().transpose(), V[mu], jrhmc.delta + jrhmc.gamma[mu], psi))
    return act

grad_H = lie_derivative(H, omega)
print('Autodiff for dH/dw:')
print(grad_H)

W = rhmc.form_W_tensor(jnp.asarray(U), gens_np, lat = Lat_np)
grad_H_analytic = np.zeros((omega.shape), dtype = np.float64)
for mu, a, nx, nt in itertools.product(range(d), range(dNc), range(Lat_np.L), range(Lat_np.T)):
    n = np.array([nx, nt])
    npmu = Lat_np.mod(n + rhmc.hat(mu))
    if np.abs(n[mu] - npmu[mu]) > 1:            # then we traversed the 0 boundary
        sign = ferm_bcs[mu]
    else:
        sign = 1
    psi_flat = rhmc.flatten_ferm_field(psi, lat = Lat_np)
    psi_n = rhmc.unflatten_colspin_vec(rhmc.flat_field_evalat(psi_flat, nx, nt, dNc, lat = Lat_np), dNc)
    psi_npmu = sign * rhmc.unflatten_colspin_vec(rhmc.flat_field_evalat(psi_flat, npmu[0], npmu[1], dNc, lat = Lat_np), dNc)
    psi_dagger_n = psi_n.conj().transpose()
    psi_dagger_npmu = psi_npmu.conj().transpose()

    grad_H_analytic[mu, a, nx, nt] = -2*np.imag(np.einsum('ib,cb,ij,cj', psi_dagger_npmu, W[mu, nx, nt, a], rhmc.delta + rhmc.gamma[mu], psi_n))
print('Analytic expression for dH/dw:')
print(grad_H_analytic)

assert np.allclose(grad_H_analytic, jnp.asarray(grad_H), rtol = 1e-3), 'Analytic expression for dH/dw does not match autodiff.'

############################################################
################## DIRAC OPERATOR DERIVATIVE ###############
############################################################
print('Testing derivative of D')
kappa = 0.2
ferm_bcs = (1, 1)       # TODO use boundary conditions

def f(U):
    """Returns the function \overline{\psi} D[U] \psi."""
    V = jrhmc.construct_adjoint_links(U, gens, lat = Lat)
    D = jrhmc.get_dirac_op_full(kappa, V, lat = Lat, bcs = ferm_bcs)
    return jnp.real(jnp.einsum('txia,aixtbjys,bjys->', psi.conj().transpose(), D, psi))
start = time.time()
grad_pf_action = lie_derivative(f, omega)
print(f'Elapsed time for autodiff call: {time.time() - start}')
print('Autodiff for \overline{\psi} D \psi:')
print(grad_pf_action)

# analytic derivative
start = time.time()
psi_flat = rhmc.flatten_ferm_field(jnp.asarray(psi), lat = Lat_np)
grad_pf_action_analytic = rhmc.test_dDdw_bilinear(
    rhmc.get_fund_field(jnp.asarray(omega), gens_np), psi_flat, kappa, gens_np, lat = Lat_np, bcs = ferm_bcs
)
print(f'Elapsed time for analytic derivative call: {time.time() - start}')
print('Analytic expression for derivative:')
print(grad_pf_action_analytic)

assert np.allclose(grad_pf_action_analytic, jnp.asarray(grad_pf_action), rtol = 1e-3), 'Analytic expression for dD/dw does not match autodiff.'

############################################################
################# DIRAC OPERATOR SQ DERIVATIVE #############
############################################################

print('Testing derivative of K')

def F(U):
    """Returns the function \overline{\psi} D[U] \psi."""
    V = jrhmc.construct_adjoint_links(U, gens, lat = Lat)
    D = jrhmc.get_dirac_op_full(kappa, V, lat = Lat, bcs = ferm_bcs)
    K = jrhmc.construct_K(D)
    return jnp.real(jnp.einsum('txia,aixtbjys,bjys->', psi.conj().transpose(), K, psi))
start = time.time()
grad_F = lie_derivative(F, omega)    # uncomment this once I get the function call running faster
print(f'Elapsed time for autodiff call: {time.time() - start}')
print('Autodiff for \overline{\psi} K \psi:')
print(grad_F)

# analytic derivative
start = time.time()
psi_flat = rhmc.flatten_ferm_field(jnp.asarray(psi), lat = Lat_np)
U_np = rhmc.get_fund_field(jnp.asarray(omega), gens_np)
dirac_sparse = rhmc.dirac_op_sparse(kappa, rhmc.construct_adjoint_links(U_np, gens_np, lat = Lat_np), bcs = ferm_bcs, lat = Lat_np)
grad_F_analytic = rhmc.test_dKdw_bilinear(
    U_np, dirac_sparse, psi_flat, kappa, gens_np, lat = Lat_np, bcs = ferm_bcs
)
print(f'Elapsed time for analytic derivative call: {time.time() - start}')
print('Analytic expression for derivative of K:')
print(grad_F_analytic)

assert np.allclose(grad_F_analytic, jnp.asarray(grad_F), rtol = 1e-3), 'Analytic expression for dK/dw does not match autodiff.'

print_line()
print('ALL TESTS PASSED')
print_line()

############################################################
####################### SCRATCH WORK #######################
############################################################

# A1 = rhmc.one_side_staple(jnp.asarray(U))
# plaq_A1 = np.einsum('...ab,...bc->...ac', U, A1)
# print(plaq_A1)

# # print()

# # differentiate \sum_x P(x) with explicit formula with plaquettes
# P0 = rhmc.plaquette_gauge_field(jnp.asarray(U))

# # A2 = np.trace(U[0, 0, 0] @ U[1, 1, 0] @ rhmc.dagger(U[0, 0, 1]) @ rhmc.dagger(U[1, 0, 0])) / Nc
# A2 = U[0, 0, 0] @ U[1, 1, 0] @ rhmc.dagger(U[0, 0, 1]) @ rhmc.dagger(U[1, 0, 0])

# A_key = random.split(key, 1)
# A = random.normal(A_key, (3, 3))

# # Example: derivative of trace with respect to A. Note that tr(A) = A00 + A11 + A22, so 
# # grad(tr)(A) = id.
# grad_tr = grad(jnp.trace)
# print(is_equal_jax(grad_tr(A), jnp.eye(3)))

# # gradient of Tr[A @ A] = Tr[A00^2 + A01*A10 + ...] is 2 A^transpose.
# grad_square = grad(lambda W : jnp.trace(W @ W))
# print(is_equal_jax(grad_square(A), 2*A.transpose()))