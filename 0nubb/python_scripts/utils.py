import numpy as np
import h5py
import os
from scipy.special import zeta
import time
import re
import itertools
import io
import random
from scipy import optimize

# STANDARD BOOTSTRAPPED PROPAGATOR ARRAY FORM: [b, cfg, c, s, c, s] where:
  # b is the boostrap index
  # cfg is the configuration index
  # c is a color index
  # s is a spinor index

Nc = 3
Nd = 4
d = 4

g = np.diag([1, 1, 1, 1])

delta = np.identity(Nd, dtype = np.complex64)
gamma = np.zeros((d, Nd, Nd),dtype=complex)
gamma[0] = gamma[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gamma[1] = gamma[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gamma[2] = gamma[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gamma[3] = gamma[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
gamma5 = np.dot(np.dot(np.dot(gamma[0], gamma[1]), gamma[2]), gamma[3])

# gammaListRev = [np.identity(4,dtype=complex), gamma[0], gamma[1], np.matmul(gamma[1], gamma[0]), gamma[2], \
#     np.matmul(gamma[2], gamma[0]), np.matmul(gamma[2], gamma[1]), np.matmul(gamma[2], np.matmul(gamma[1],gamma[0])), \
#     gamma[3], np.matmul(gamma[3], gamma[0]), np.matmul(gamma[3], gamma[1]), np.matmul(gamma[3], np.matmul(gamma[1], gamma[0])), \
#     np.matmul(gamma[3], gamma[2]), np.matmul(gamma[3], np.matmul(gamma[2], gamma[0])), np.matmul(gamma[3], np.matmul(gamma[2], gamma[1])), \
#     np.matmul(np.matmul(gamma[3], gamma[2]), np.matmul(gamma[1], gamma[0]))]
gammaList = [np.identity(4,dtype=complex), gamma[0], gamma[1], np.matmul(gamma[0], gamma[1]), gamma[2], \
    np.matmul(gamma[0], gamma[2]), np.matmul(gamma[1], gamma[2]), np.matmul(gamma[0], np.matmul(gamma[1],gamma[2])), \
    gamma[3], np.matmul(gamma[0], gamma[3]), np.matmul(gamma[1], gamma[3]), np.matmul(gamma[0], np.matmul(gamma[1], gamma[3])), \
    np.matmul(gamma[2], gamma[3]), np.matmul(gamma[0], np.matmul(gamma[2], gamma[3])), np.matmul(gamma[1], np.matmul(gamma[2], gamma[3])), \
    np.matmul(np.matmul(gamma[0], gamma[1]), np.matmul(gamma[2], gamma[3]))]

bvec = [0, 0, 0, .5]

# initialize Dirac matrices
gammaMu5 = np.array([gamma[mu] @ gamma5 for mu in range(d)])
sigmaD = np.zeros((Nd, Nd, Nd, Nd), dtype = np.complex64)               # sigma_{mu nu}
gammaGamma = np.zeros((Nd, Nd, Nd, Nd), dtype = np.complex64)           # gamma_mu gamma_nu
for mu in range(d):
    for nu in range(mu + 1, d):
        sigmaD[mu, nu, :, :] = (1 / 2) * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu])
        sigmaD[nu, mu, :, :] = -sigmaD[mu, nu, :, :]
        gammaGamma[mu, nu, :, :] = gamma[mu] @ gamma[nu]
        gammaGamma[nu, mu, :, :] = - gammaGamma[mu, nu, :, :]

# tree level structures for the operators
tree = np.zeros((5, Nc, Nd, Nc, Nd, Nc, Nd, Nc, Nd), dtype = np.complex64)
for a, b in itertools.product(range(Nc), repeat = 2):
    for alpha, beta, gam, sigma in itertools.product(range(Nd), repeat = 4):
        tree[2, a, alpha, a, beta, b, gam, b, sigma] += 2 * (delta[alpha, beta] * delta[gam, sigma] - gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[2, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (delta[alpha, sigma] * delta[gam, beta] - gamma5[alpha, sigma] * gamma5[gam, beta])
        tree[3, a, alpha, a, beta, b, gam, b, sigma] += 2 * (delta[alpha, beta] * delta[gam, sigma] + gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[3, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (delta[alpha, sigma] * delta[gam, beta] + gamma5[alpha, sigma] * gamma5[gam, beta])
        for mu in range(d):
            tree[0, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gamma[mu, alpha, beta] * gamma[mu, gam, sigma] + gammaMu5[mu, alpha, beta] * gammaMu5[mu, gam, sigma])
            tree[0, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gamma[mu, alpha, sigma] * gamma[mu, gam, beta] + gammaMu5[mu, alpha, sigma] * gammaMu5[mu, gam, beta])
            tree[1, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gamma[mu, alpha, beta] * gamma[mu, gam, sigma] - gammaMu5[mu, alpha, beta] * gammaMu5[mu, gam, sigma])
            tree[1, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gamma[mu, alpha, sigma] * gamma[mu, gam, beta] - gammaMu5[mu, alpha, sigma] * gammaMu5[mu, gam, beta])
            for nu in range(mu + 1, d):
                tree[4, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gammaGamma[mu, nu, alpha, beta] * gammaGamma[mu, nu, gam, sigma])
                tree[4, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gammaGamma[mu, nu, alpha, sigma] * gammaGamma[mu, nu, gam, beta])

# L = 16
# T = 48
# LL = [L, L, L, T]
# vol = (L ** 3) * T
#
# def set_dimensions(l, t):
#     global L
#     global T
#     global LL
#     L, T, vol = l, t, (l ** 3) * t
#     LL = [l, l, l, t]
#     return L, T, vol, LL
#
# def to_linear_momentum(k):
#     return np.array([np.complex64(2 * np.pi * k[mu] / LL[mu]) for mu in range(4)])
#
# def to_lattice_momentum(k):
#     return np.array([np.complex64(2 * np.sin(np.pi * k[mu] / LL[mu])) for mu in range(4)])
# # Converts a wavevector to an energy scale using ptwid. Lattice parameter is a = A femtometers.
# # Shouldn't use this-- should use k_to_mu_p instead and convert at p^2 = mu^2
# def k_to_mu_ptwid(k, A = .1167):
#     aGeV = fm_to_GeV(A)
#     return 2 / aGeV * np.sqrt(sum([np.sin(np.pi * k[mu] / LL[mu]) ** 2 for mu in range(4)]))
#
# def k_to_mu_p(k, A = .1167):
#     aGeV = fm_to_GeV(A)
#     return (2 * np.pi / aGeV) * np.sqrt(sum([(k[mu] / LL[mu]) ** 2 for mu in range(4)]))

# Saves the dimensions of a lattice.
class Lattice:
    def __init__(self, l, t):
        self.L = l
        self.T = t
        self.LL = [l, l, l, t]
        self.vol = (l ** 3) * t

    def to_linear_momentum(self, k):
        return np.array([np.complex64(2 * np.pi * k[mu] / self.LL[mu]) for mu in range(Nd)])

    def to_lattice_momentum(self, k):
        return np.array([np.complex64(2 * np.sin(np.pi * k[mu] / self.LL[mu])) for mu in range(Nd)])
    # Converts a wavevector to an energy scale using ptwid. Lattice parameter is a = A femtometers.
    # Shouldn't use this-- should use k_to_mu_p instead and convert at p^2 = mu^2
    def k_to_mu_ptwid(self, k, A = .1167):
        aGeV = fm_to_GeV(A)
        return 2 / aGeV * np.sqrt(sum([np.sin(np.pi * k[mu] / self.LL[mu]) ** 2 for mu in range(4)]))

    def k_to_mu_p(self, k, A = .1167):
        aGeV = fm_to_GeV(A)
        return (2 * np.pi / aGeV) * np.sqrt(sum([(k[mu] / self.LL[mu]) ** 2 for mu in range(4)]))

n_boot = 50
# n_boot = 100

def kstring_to_list(pstring, str):
    def get_momenta(x):
        lst = []
        mult = 1
        for digit in x:
            if digit == '-':
                mult *= -1
            else:
                lst.append(mult * int(digit))
                mult = 1
        return lst
    return get_momenta(pstring.split(str)[1])

# str is the beginning of the string, ex klist_to_string([1, 2, 3, 4], 'k1') gives 'k1_1234'
def klist_to_string(k, prefix):
    return prefix + str(k[0]) + str(k[1]) + str(k[2]) + str(k[3])

# squares a 4 vector.
def square(k):
    return np.dot(k, np.dot(g, k.T))

def slash(k):
    return sum([k[mu] * gamma[mu] for mu in range(4)])

def norm(p):
    return np.sqrt(np.abs(square(p)))

# cfgs should contain a list of paths to hdf5 files. If op_renorm is True, then will also
# read in 3 point functions for operator renormalization; if it is False, then
# will only read in 3 point functions for quark / vector / axial renormalization,
# and will return an empty list for G0
def readfiles(cfgs, q, op_renorm = True):
    props_k1 = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    props_k2 = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    props_q = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GV = np.zeros((4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GA = np.zeros((4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GO = np.zeros((16, len(cfgs), 3, 4, 3, 4, 3, 4, 3, 4), dtype = np.complex64)

    for idx, file in enumerate(cfgs):
        f = h5py.File(file, 'r')
        qstr = klist_to_string(q, 'q')
        if idx == 0:            # just choose a specific config to get these on, since they should be the same
            k1 = f['moms/' + qstr + '/k1'][()]
            k2 = f['moms/' + qstr + '/k2'][()]
        props_k1[idx] = np.einsum('ijab->aibj', f['prop_k1/' + qstr][()])
        props_k2[idx] = np.einsum('ijab->aibj', f['prop_k2/' + qstr][()])
        props_q[idx] = np.einsum('ijab->aibj', f['prop_q/' + qstr][()])
        for mu in range(4):
            GV[mu, idx] = np.einsum('ijab->aibj', f['GV' + str(mu + 1) + '/' + qstr][()])
            GA[mu, idx] = np.einsum('ijab->aibj', f['GA' + str(mu + 1) + '/' + qstr][()])
        if op_renorm:
            for n in range(16):
                # GO[n, idx] = np.einsum('ijklabcd->aibjckdl', f['Gn' + str(n) + '/' + qstr][()])
                GO[n, idx] = np.einsum('ijklabcd->dlckbjai', f['Gn' + str(n) + '/' + qstr][()])
    return k1, k2, props_k1, props_k2, props_q, GV, GA, GO

# Read an Npt function in David's format
def read_Npt(folder, stem, fnums, N, n_t):
    files = [folder + '/' + stem + '.' + str(fnum) for fnum in fnums]
    if N == 2:
        Cnpt = np.zeros((len(fnums), n_t, n_t), dtype = np.complex64)
    else:
        Cnpt = np.zeros((len(fnums), 24, n_t, n_t, n_t), dtype = np.complex64)   # for raw readout without operator structures
        # Cnpt = np.zeros((len(fnums), 5, n_t, n_t, n_t), dtype = np.complex64)   # for non time-averaged data
        # Cnpt = np.zeros((len(fnums), 5, n_t, n_t), dtype = np.complex64)     # for time averaged data
    for f_idx, file in enumerate(files):
        print(file)
        with io.open(file, 'r', encoding = 'windows-1252') as f:    # encoding catches ASCII characters
            lines = [line.rstrip() for line in f]
        for l in lines:
            # print(l)
            x = l.split()
            if N == 2:
                tsrc, tsnk = int(x[0]), int(x[1])
                entry = np.complex(float(x[2]), float(x[3]))
                Cnpt[f_idx, tsrc, tsnk] = entry
            else:           # read 3-pt contraction structure
                # read out non time-averaged data, without constructing the appropriate operator structures in this function.
                try:
                    tminus, tx, tplus = int(x[0]), int(x[1]), int(x[2])
                    start = 3               # first index with an actual entry
                    for op_idx in range(24):
                        Cnpt[f_idx, op_idx, tminus, tx, tplus] = np.complex(float(x[start + 2 * op_idx]), float(x[start + 2 * op_idx + 1]))
                except (ValueError, UnicodeDecodeError) as e:
                    print('Error: File ' + str(file) + ' at index (' + str(tminus) + ','  + str(tx) + ', ' + str(tplus) + '), op_idx = ' + str(op_idx))
                    print(e)
    return Cnpt

# C should be an array of size [num_files, T] where T is the lattice temporal extent.
# Folder is a function of two arguments which returns how to add them, (A + B)
# for sym and (A - B) for antisym
def fold(C, T, folder = np.sum):
    folded = np.zeros((C.shape[0], T // 2 + 1), dtype = np.complex64)
    folded[:, 0] = C[:, 0]
    for t in range(T // 2):
        # print((t + 1, T - (t + 1)))
        folded[:, t + 1] = folder(C[:, t + 1], C[:, T - (t + 1)]) / 2
    return folded

# data should be an array of size (n_fits, T) and fit_region gives the times to fit at
def fit_constant(fit_region, data):
    if type(fit_region) != np.ndarray:
        fit_region = np.array([x for x in fit_region])
    if len(data.shape) == 1:        # if data only has one dimension, add an axis
        data = np.expand_dims(data, axis = 0)
    sigma_fit = np.std(data[:, fit_region], axis = 0)
    # sigma_fit = np.std(data[:, fit_region], axis = 0, ddof = 1)
    c_fit = np.zeros((n_boot), dtype = np.float64)
    # cov = np.zeros((n_boot, n_boot), dtype = np.float64)
    for i in range(n_boot):
        data_fit = data[i, fit_region]
        x0 = [1]          # guess to start at
        # leastsq = lambda x, data : np.sum((data - x[0]) ** 2)        # if we aren't allowed to use the error, do least squares
        # out = optimize.minimize(leastsq, x0, args=(data_fit), method = 'Powell')
        chi2 = lambda x, data, sigma : np.sum((data - x[0]) ** 2 / (sigma ** 2))     # x[0] = constant to fit to
        out = optimize.minimize(chi2, x0, args=(data_fit, sigma_fit), method = 'Powell')
        c_fit[i] = out['x']
        # cov_{ij} = 1/2 * D_i D_j chi^2

    # # try with least squares regression in scipy.optimize
    # fitfunc = lambda c, x: c[0]
    # errfunc = lambda c, x, y, err: (y - fitfunc(c, x)) / err
    # c_init = [0.0]
    # out = optimize.leastsq(errfunc, c_init, args=(fit_region, data_fit, sigma_fit), full_output=1)
    # c, cov = out[0], out[1]
    # sigma = np.sqrt(cov[0, 0])

    # # try with np.polyfit-- gives about the same as scipy.curve_fit, neither of them propagate uncertainty on the data points
    # c, cov = np.polyfit(fit_region, data_fit, 0, w = 1 / sigma_fit, cov = True)
    # sigma = np.sqrt(cov[0, 0])

    # c, sigma = np.mean(c_fit), np.std(c_fit)
    # return c, sigma

    return c_fit#, cov

# Generate fake ensemble of data with mean mu and std sigma
def gen_fake_ensemble(val, n_samples, n_ens = 5, s = 20):
    random.seed(s)
    fake_data = np.zeros((n_samples), dtype = np.float64)
    # generate fake data in each ensemble with total std sigma: since we're treating each ensemble
    # independently and add their variances in quadrature, each ensemble should have std = sigma / sqrt(N_ens)
    mu, sigma = val[0], val[1] / np.sqrt(n_ens)
    for i in range(n_samples):
        fake_data[i] = random.gauss(mu, sigma)
    return fake_data

# BOOTSTRAP OBJECT: should be an array of size (n_ens, n_boot)
class Superboot:

    def __init__(self, n_ens, nb = n_boot):
        self.n_ens = n_ens
        self.boots = np.zeros((n_ens, n_boot), dtype = np.float64)
        self.avg = 0
        self.n_boot = nb
        self.card = nb * n_ens    # total cardinality

    def gen_ensemble(self, data, axis):
        self.avg = np.mean(data)
        for e in range(self.n_ens):
            if e == axis:
                self.boots[e] = data
            else:
                self.boots[e] = self.avg
        return self

    def gen_fake_ensemble(self, mean, sigma, s = 20):
        random.seed(s)
        self.avg = mean
        ens_sig = sigma / np.sqrt(self.n_ens)
        for e in range(self.n_ens):
            self.boots[e, b] = random.gauss(mean, ens_sig)
        return self

    def compute_mean(self):
        self.mean = np.sum(self.boots) / self.card
        return self.mean

    def compute_std(self):
        vars = np.std(self.boots, axis = 1, ddof = 1) ** 2    # TODO may need to change ddof
        self.std = np.sqrt(np.sum(vars))
        return self.std

    def __add__(self, other):
        sum = Superboot(self.n_ens, self.n_boot)
        sum.boots = self.boots + other.boots
        sum.compute_mean()
        sum.compute_std()
        return sum

    def __mul__(self, other):
        prod = Superboot(self.n_ens, self.n_boot)
        prod.boots = self.boots * other.boots
        prod.compute_mean()
        prod.compute_std()
        return prod

# X should be a list of length n_ens, containing arrays of shape (n_ops, n_samples[ens_idx]). Averages over ensemble and boot indices (superboot indices)
def superboot_mean(X):
    N = len(X)
    return sum([np.mean(X[i], axis = 1) for i in range(N)]) / N

# Compute sigma^2 for each ensemble, then add in quadrature
def superboot_std(X):
    N = len(X)
    var = sum([np.std(X[i], axis = 1)**2 for i in range(N)])
    return np.sqrt(var)

# Bootstraps an input tensor. Pass in a tensor with the shape (ncfgs, tensor_shape)
def bootstrap(S, seed = 10, weights = None, data_type = np.complex64):
    num_configs, tensor_shape = S.shape[0], S.shape[1:]
    bootshape = [n_boot]
    bootshape.extend(tensor_shape)    # want bootshape = (n_boot, tensor_shape)
    samples = np.zeros(bootshape, dtype = data_type)
    if weights == None:
        weights = np.ones((num_configs))
    weights2 = weights / float(np.sum(weights))
    np.random.seed(seed)
    for boot_id in range(n_boot):
        cfg_ids = np.random.choice(num_configs, p = weights2, size = num_configs, replace = True)
        samples[boot_id] = np.mean(S[cfg_ids], axis = 0)
    return samples

# Invert propagators to get S^{-1} required for amputation.
def invert_props(props):
    Sinv = np.zeros(props.shape, dtype = np.complex64)
    for b in range(n_boot):
        Sinv[b] = np.linalg.tensorinv(props[b])
    return Sinv

# Amputate legs to get vertex function \Gamma(p). Uses first argument to amputate left-hand side and
# second argument to amputate right-hand side.
def amputate_threepoint(props_inv_L, props_inv_R, threepts):
    Gamma = np.zeros(threepts.shape, dtype = np.complex64)
    for b in range(n_boot):
        Sinv_L, Sinv_R, G = props_inv_L[b], props_inv_R[b], threepts[b]
        Gamma[b] = np.einsum('aibj,bjck,ckdl->aidl', Sinv_L, G, Sinv_R)
    return Gamma

# amputates the four point function. Assumes the left leg has momentum p1 and right legs have
# momentum p2, so amputates with p1 on the left and p2 on the right
def amputate_fourpoint(props_inv_L, props_inv_R, fourpoints):
    Gamma = np.zeros(fourpoints.shape, dtype = np.complex64)
    for b in range(n_boot):
        Sinv_L, Sinv_R, G = props_inv_L[b], props_inv_R[b], fourpoints[b]
        Gamma[b] = np.einsum('aiem,ckgp,emfngphq,fnbj,hqdl->aibjckdl', Sinv_L, Sinv_L, G, Sinv_R, Sinv_R)
    return Gamma

# q is the lattice momentum that should be passed in
def quark_renorm(props_inv_q, q):
    Zq = np.zeros((n_boot), dtype = np.complex64)
    for b in range(n_boot):
        Sinv = props_inv_q[b]
        Zq[b] = (1j) * sum([q[mu] * np.einsum('ij,ajai', gamma[mu], Sinv) for mu in range(d)]) / (12 * square(q))
    return Zq

# Returns the adjoint of a bootstrapped propagator object
def adjoint(S):
    return np.conjugate(np.einsum('...aibj->...bjai', S))

def antiprop(S):
    Sdagger = adjoint(S)
    return np.einsum('ij,zajbk,kl->zaibl', gamma5, Sdagger, gamma5)

# key == 'gamma' or key == 'qslash' for the different schemes. If qslash, must enter q, p_1, and p_2
# Along with getF(), seems to return the correct projectors which agree with the paper
def projectors(scheme = 'gamma', q = 0, p1 = 0, p2 = 0):
    P = np.zeros((5, Nc, Nd, Nc, Nd, Nc, Nd, Nc, Nd), dtype = np.complex64)
    if scheme == 'gamma':
        for a, b in itertools.product(range(Nc), repeat = 2):
            for beta, alpha, sigma, gam in itertools.product(range(Nd), repeat = 4):
                P[2, a, beta, a, alpha, b, sigma, b, gam] += delta[beta, alpha] * delta[sigma, gam] - gamma5[beta, alpha] * gamma5[sigma, gam]
                P[3, a, beta, a, alpha, b, sigma, b, gam] += delta[beta, alpha] * delta[sigma, gam] + gamma5[beta, alpha] * gamma5[sigma, gam]
                for mu in range(d):
                    P[0, a, beta, a, alpha, b, sigma, b, gam] += gamma[mu, beta, alpha] * gamma[mu, sigma, gam] + gammaMu5[mu, beta, alpha] * gammaMu5[mu, sigma, gam]
                    P[1, a, beta, a, alpha, b, sigma, b, gam] += gamma[mu, beta, alpha] * gamma[mu, sigma, gam] - gammaMu5[mu, beta, alpha] * gammaMu5[mu, sigma, gam]
                    for nu in range(d):
                        P[4, a, beta, a, alpha, b, sigma, b, gam] += (1 / 2) * sigmaD[mu, nu, beta, alpha] * sigmaD[mu, nu, sigma, gam]
    elif scheme == 'qslash':
        qsq, p1sq, p2sq, qslash, p1p2 = square(q), square(p1), square(p2), slash(q), np.dot(p1, p2)
        qslash5 = qslash @ gamma5
        Pl = (delta - gamma5) / 2
        psigmap = np.zeros((Nd, Nd), dtype = np.complex64)
        for mu, nu in itertools.product(range(Nd), repeat = 2):
            psigmap += p1[mu] * (sigmaD[mu, nu] @ Pl) * p2[nu]
        for a, b in itertools.product(range(Nc), repeat = 2):
            for beta, alpha, sigma, gam in itertools.product(range(Nd), repeat = 4):
                P[0, a, beta, a, alpha, b, sigma, b, gam] += (1 / qsq) * (qslash[beta, alpha] * qslash[sigma, gam] + qslash5[beta, alpha] * qslash5[sigma, gam])
                P[1, a, beta, a, alpha, b, sigma, b, gam] += (1 / qsq) * (qslash[beta, alpha] * qslash[sigma, gam] - qslash5[beta, alpha] * qslash5[sigma, gam])
                P[2, a, beta, b, alpha, b, sigma, a, gam] += (1 / qsq) * (qslash[beta, alpha] * qslash[sigma, gam] - qslash5[beta, alpha] * qslash5[sigma, gam])
                P[3, a, beta, b, alpha, b, sigma, a, gam] += (1 / (p1sq * p2sq - p1p2 * p1p2)) * (psigmap[beta, alpha] * psigmap[sigma, gam])
                P[4, a, beta, a, alpha, b, sigma, b, gam] += (1 / (p1sq * p2sq - p1p2 * p1p2)) * (psigmap[beta, alpha] * psigmap[sigma, gam])
    return P

# gets tree level projections in scheme == 'gamma' or scheme == 'qslash'. Pass in a Lattice instance L so that
# the function knows the lattice dimensions.
def getF(L, scheme = 'gamma'):
    q, p1, p2 = L.to_linear_momentum([1, 1, 0, 0]), L.to_linear_momentum([-1, 0, 1, 0]), L.to_linear_momentum([0, 1, 1, 0])
    P = projectors(scheme, q, p1, p2)
    F = np.einsum('nbjaidlck,maibjckdl->mn', P, tree)
    return F

# Returns a in units of GeV^{-1}
def fm_to_GeV(a):
    return a / .197327

def load_data_h5(file):
    print('Loading ' + str(file) + '.')
    f = h5py.File(file, 'r')
    k_list = f['momenta'][()]
    p_list = np.array([to_lattice_momentum(k) for k in k_list])
    Z = f['Z'][()]
    Zq = f['Zq'][()]
    cfgnum = f['cfgnum'][()]
    f.close()
    return k_list, p_list, Z, Zq, cfgnum

def load_Zq(file):
    f = h5py.File(file, 'r')
    k_list = f['momenta'][()]
    p_list = np.array([to_lattice_momentum(k) for k in k_list])
    Zq = f['Zq'][()]
    f.close()
    return k_list, p_list, Zq

# partition k_list into orbits by O(3) norm and p[3]
def get_O3_orbits(k_list):
    orbits, psquared_p3_list = [], []    # these should always be the same size
    for p in k_list:
        psquared = p[0] ** 2 + p[1] ** 2 + p[2] ** 2
        if (psquared, p[3]) in psquared_p3_list:    # then add to existing orbit
            idx = psquared_p3_list.index((psquared, p[3]))
            orbits[idx].append(p)
        else:
            psquared_p3_list.append((psquared, p[3]))
            orbits.append([p])
    return orbits, psquared_p3_list

# Z is a (n_momentum x n_boot) matrix
def average_O3_orbits(Z, k_list):
    orbits, psquared_p3_list = get_O3_orbits(k_list)
    k_rep_list, Zavg, sigma = [], [], []
    for coset in orbits:
        # idx_list = [k_list.index(p) for p in coset]
        idx_list = [np.where(k_list == p) for p in coset]
        k_rep_list.append(coset[0])    # take the first representative
        Zavg.append(np.mean(Z[idx_list, :]))
        sigma.append(np.std(Z[idx_list, :]))
    return k_rep_list, Zavg, sigma
