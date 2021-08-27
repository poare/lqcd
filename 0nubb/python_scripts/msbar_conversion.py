import numpy as np
from scipy.optimize import root
import h5py
import os
from utils import *

# parameters for 24I
# ensembles = ['24I/ml0p005/', '24I/ml0p01/']
# l, t = 24, 64
# ainv = 1.784                              # GeV
# mpi_list = [0.3396, 0.4322]               # GeV
# amq_list = [0.005, 0.01]

# parameters for 32I
ensembles = ['32I/ml0p004/', '32I/ml0p006/']
l, t = 32, 64
ainv = 2.382                                # GeV
mpi_list = [0.3020, 0.3597]                 # GeV
amq_list = [0.004, 0.006]

L = Lattice(l, t)
a_fm = hbarc / ainv
n_ens = len(ensembles)
ampi_list = [mpi_list[i] / ainv for i in range(n_ens)]

file_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ens + 'Z_gamma.h5' for ens in ensembles]
chi_extrap_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ensembles[0][:3] + '/chiral_extrap/Z_extrap_linear.h5'
Fs = [h5py.File(fpath, 'r') for fpath in file_paths]
# k_list = [f['momenta'][()] for f in Fs]
# mom_list = [[L.to_linear_momentum(k, datatype=np.float64) for k in k_list[i]] for i in range(n_ens)]
# mu_list = [np.array([get_energy_scale(q, a_fm, L) for q in k_list[i]]) for i in range(n_ens)]
k_list_ens = [f['momenta'][()] for f in Fs]
print(k_list_ens)
assert np.array_equal(k_list_ens[0], k_list_ens[1])         # make sure each ensemble has same momentum modes
k_list = k_list_ens[0]
mom_list = np.array([L.to_linear_momentum(k, datatype=np.float64) for k in k_list])
mu_list = np.array([get_energy_scale(q, a_fm, L) for q in k_list])
n_mom = len(mom_list)

# Get renormalization coefficients (not chirally extrapolated)
Zq_list = [np.real(f['Zq'][()]) for f in Fs]
ZV_list = [np.real(f['ZV'][()]) for f in Fs]
ZA_list = [np.real(f['ZA'][()]) for f in Fs]
Z_list = []
for idx in range(n_ens):
    Z = np.zeros((5, 5, n_mom, n_boot), dtype = np.float64)
    f = Fs[idx]
    for i, j in itertools.product(range(5), repeat = 2):
        key = 'Z' + str(i + 1) + str(j + 1)
        Z[i, j] = np.real(f[key][()])
    Z_list.append(Z)

# Zq_boots = [Superboot(n_ens) for i in range(n_mom)]
# for i in range(n_mom):
#     Zq_boots[i].boots = np.array(Zq_list, dtype = np.float64)
#     Zq_boots[i].compute_mean()
#     Zq_boots[i].compute_std()

# Get Lambda factor. Lambdas are bootstrapped, but the boots are uncorrelated. Shape is (n_ens, 5, 5, n_mom, n_boot)
Lambda_list = [np.real(f['Lambda'][()]) for f in Fs]
multiplets = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [1, 2], [2, 1], [3, 4], [4, 3]], dtype = object)    # nonzero indices for Lambda

# perform and save chiral extrapolation on Lambda
# Lambda_fit = np.zeros((len(multiplets), n_mom, n_ens, n_boot), dtype = np.float64)    # last two indices are Superboot boots shape
Lambda_fit = np.zeros((n_ens, n_boot, n_mom, 5, 5), dtype = np.float64)
# mass_list = np.array(mpi_list)
mass_list = np.array(amq_list)
for ii, mult_idx in enumerate(multiplets):
    print('Fitting Lambda' + str(mult_idx[0]) + str(mult_idx[1]))
    Lambda_mult = []
    for mom_idx in range(n_mom):
        print('Momentum index ' + str(mom_idx))
        fitdata_superboot = []    # create n_ens Superboot objects
        for ii in range(n_ens):
            tmp = Superboot(n_ens)
            tmp.populate_ensemble(Lambda_list[ii][mult_idx[0], mult_idx[1], mom_idx], ii)
            fitdata_superboot.append(tmp)
        fit_data, chi2, y_extrap = uncorr_linear_fit(mass_list, fitdata_superboot, 0)
        # chi2, y_extrap = uncorr_const_fit(mass_list, fitdata_superboot, 0.140)
        Lambda_fit[:, :, mom_idx, mult_idx[0], mult_idx[1]] = y_extrap.boots    # just reput them into new superboot objects when we read them in

# Process as a Z factor
scheme = 'gamma'                    # scheme == 'gamma' or 'qslash'
F = getF(L, scheme)                 # tree level projections
Z_chiral = np.zeros((5, 5, n_mom, n_ens, n_boot))
for mom_idx in range(n_mom):
    for ens_idx in range(n_ens):
        for b in range(n_boot):
            Lambda_inv = np.linalg.inv(Lambda_fit[ens_idx, b, mom_idx])
            Z_chiral[:, :, mom_idx, ens_idx, b] = (Zq_list[ens_idx][mom_idx, b] ** 2) * np.einsum('ik,kj->ij', F, Lambda_inv)
    for mult_idx in multiplets:
        Z_boot = Superboot(n_ens)
        Z_boot.boots = Z_chiral[mult_idx[0], mult_idx[1], mom_idx]
        Z_boot.compute_mean()
        Z_boot.compute_std()
        print('Z' + str(mult_idx) + ' at mom_idx ' + str(mom_idx) + ' = ' + str(Z_boot.mean) + ' \pm ' + str(Z_boot.std))

# save chirally extrapolated Lambda and Z
fchi_out = h5py.File(chi_extrap_path, 'w')
fchi_out['momenta'] = k_list
for mult_idx in multiplets:
    fchi_out['Lambda' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = np.einsum('ebq->qeb', Lambda_fit[:, :, :, mult_idx[0], mult_idx[1]])
    fchi_out['Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Z_chiral[mult_idx[0], mult_idx[1]]
print('Chiral extrapolation saved at: ' + chi_extrap_path)
fchi_out.close()

# Compute anomalous dimension and run Z factors to the matching scale

# interpolate each Z factor to get Z(3 GeV). Fit the functional form in (ap)^2
