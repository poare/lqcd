# Tree level analysis for 0nubb mixing in the gamma^mu scheme

import numpy as np
from scipy.optimize import root
import h5py
import os
from analysis import *

gammaGamma = np.zeros((4, 4, 4, 4), dtype = np.complex64)    # sigma_{mu nu}
for mu in range(4):
    for nu in range(mu + 1, 4):
        gammaGamma[mu, nu, :, :] = gamma[mu] @ gamma[nu]
        gammaGamma[nu, mu, :, :] = - gammaGamma[mu, nu, :, :]

# initialize tree level operators. tree[0] --> Gamma_1, tree[1] --> Gamma_2, tree[2] --> Gamma_3, tree[3] --> Gamma_1', tree[4] --> Gamma2'
tree = np.zeros((5, 3, 4, 3, 4, 3, 4, 3, 4), dtype = np.complex64)
for a, b in itertools.product(range(3), repeat = 2):
    for alpha, beta, gam, sigma in itertools.product(range(4), repeat = 4):
        tree[1, a, alpha, a, beta, b, gam, b, sigma] += 2 * (deltaD[alpha, beta] * deltaD[gam, sigma] + gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[1, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (deltaD[alpha, sigma] * deltaD[gam, beta] + gamma5[alpha, sigma] * gamma5[gam, beta])
        tree[3, a, alpha, a, beta, b, gam, b, sigma] -= 2 * (deltaD[alpha, beta] * deltaD[gam, sigma] - gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[3, a, alpha, b, beta, b, gam, a, sigma] += 2 * (deltaD[alpha, sigma] * deltaD[gam, beta] - gamma5[alpha, sigma] * gamma5[gam, beta])
        tree[4, a, alpha, a, beta, b, gam, b, sigma] -= (deltaD[alpha, beta] * deltaD[gam, sigma] + gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[4, a, alpha, b, beta, b, gam, a, sigma] += (deltaD[alpha, sigma] * deltaD[gam, beta] + gamma5[alpha, sigma] * gamma5[gam, beta])
        for mu in range(4):
            tree[0, a, alpha, a, beta, b, gam, b, sigma] += (gamma[mu, alpha, beta] * gamma[mu, gam, sigma] - gammaMu5[mu, alpha, beta] * gammaMu5[mu, gam, sigma])
            tree[0, a, alpha, b, beta, b, gam, a, sigma] -= (gamma[mu, alpha, sigma] * gamma[mu, gam, beta] - gammaMu5[mu, alpha, sigma] * gammaMu5[mu, gam, beta])
            tree[2, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gamma[mu, alpha, beta] * gamma[mu, gam, sigma] + gammaMu5[mu, alpha, beta] * gammaMu5[mu, gam, sigma])
            tree[2, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gamma[mu, alpha, sigma] * gamma[mu, gam, beta] + gammaMu5[mu, alpha, sigma] * gammaMu5[mu, gam, beta])
            for nu in range(mu + 1, 4):
                tree[4, a, alpha, a, beta, b, gam, b, sigma] += (gammaGamma[mu, nu, alpha, beta] * gammaGamma[mu, nu, gam, sigma])
                tree[4, a, alpha, b, beta, b, gam, a, sigma] -= (gammaGamma[mu, nu, alpha, sigma] * gammaGamma[mu, nu, gam, beta])
P = projectors()
F = np.einsum('nbjaidlck,maibjckdl->mn', P, tree)    # F_{ij} = P_j Gamma_i
print(F)
