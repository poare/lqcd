import numpy as np
from scipy.optimize import root
import h5py
import os
from utils import *

# parameters for 24I
ensembles = ['24I/ml0p005/', '24I/ml0p01/']
l, t = 24, 64
ainv = 1.784                              # GeV
mpi_list = [0.3396, 0.4322]               # GeV
amq_list = [0.005, 0.01]

# parameters for 32I
# ensembles = ['32I/ml0p004/', '32I/ml0p006/']
# l, t = 32, 64
# ainv = 2.382                                # GeV
# mpi_list = [0.3020, 0.3597]                 # GeV
# amq_list = [0.004, 0.006]

L = Lattice(l, t)
a_fm = hbarc / ainv
n_ens = len(ensembles)
ampi_list = [mpi_list[i] / ainv for i in range(n_ens)]

file_paths = ['/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ens + 'Z_gamma.h5' for ens in ensembles]
out_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ensembles[0][:3] + '/chiral_extrap/Z_extrap.h5'
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
print('Energy scales')
print(mu_list)
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

# chirally extrapolate Zq
print('Chiral extrapolation for Zq.')
mass_list = np.array(amq_list)
Zq_fit = np.zeros((n_mom, n_ens, n_boot), dtype = np.float64)
for mom_idx in range(n_mom):
    print('Momentum index ' + str(mom_idx))
    fitq_superboot = []
    for ii in range(n_ens):
        tmpq = Superboot(n_ens)
        tmpq.populate_ensemble(Zq_list[ii][mom_idx], ii)
        fitq_superboot.append(tmpq)
    _, chi2_q, y_extrap_q = uncorr_linear_fit(mass_list, fitq_superboot, 0, label = 'Zq')
    Zq_fit[mom_idx] = y_extrap_q.boots

# chirally extrapolate ZV and ZA
print('Chiral extrapolation for ZV and ZA.')
# mass_list = np.array(mpi_list)
# mass_list = np.array(amq_list)
ZV_fit = np.zeros((n_mom, n_ens, n_boot), dtype = np.float64)
ZA_fit = np.zeros((n_mom, n_ens, n_boot), dtype = np.float64)
for mom_idx in range(n_mom):
    print('Momentum index ' + str(mom_idx))
    fitV_superboot = []
    fitA_superboot = []
    for ii in range(n_ens):
        tmpV = Superboot(n_ens)
        tmpV.populate_ensemble(ZV_list[ii][mom_idx], ii)
        fitV_superboot.append(tmpV)
        tmpA = Superboot(n_ens)
        tmpA.populate_ensemble(ZA_list[ii][mom_idx], ii)
        fitA_superboot.append(tmpA)
    _, chi2_V, y_extrap_V = uncorr_linear_fit(mass_list, fitV_superboot, 0, label = 'ZV')
    _, chi2_A, y_extrap_A = uncorr_linear_fit(mass_list, fitA_superboot, 0, label = 'ZA')
    ZV_fit[mom_idx] = y_extrap_V.boots
    ZA_fit[mom_idx] = y_extrap_A.boots

# Get Lambda factor. Lambdas are bootstrapped, but the boots are uncorrelated. Shape is (n_ens, 5, 5, n_mom, n_boot)
Lambda_list = [np.real(f['Lambda'][()]) for f in Fs]
multiplets = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [1, 2], [2, 1], [3, 4], [4, 3]], dtype = object)    # nonzero indices for Lambda

# perform and save chiral extrapolation on Lambda
print('Chiral extrapolation for Lambda_ij.')
Lambda_fit = np.zeros((n_ens, n_boot, n_mom, 5, 5), dtype = np.float64)
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

# fit Zq and Zq / ZV to extract ZV
print('Fitting Zq and Zq / ZV.')
Zq_by_ZV = Zq_fit / ZV_fit
mom_subset = [3, 4, 5, 6, 7]
pwr = 3
apsq_list = np.array([square(p) for p in mom_list])
print('Fitting Zq. Extrapolating to (ap) -> 0.')
fit_coeffs_Zq, chi2_Zq, y_extrap_Zq = corr_superboot_fit_apsq(apsq_list[mom_subset], Zq_fit[mom_subset], order = pwr, mu1 = 0.0, label = 'Zq')

print('Fitting Zq / ZV. Extrapolating to (ap) -> 0.')
fit_coeffs_ZqZV, chi2_ZqZV, y_extrap_ZqZV = corr_superboot_fit_apsq(apsq_list[mom_subset], Zq_by_ZV[mom_subset], order = pwr, mu1 = 0.0, label = 'Zq / ZV')

y_extrap_ZV = y_extrap_Zq / y_extrap_ZqZV
print('Fit for ZV ~ ' + str(np.mean(y_extrap_ZV)))

# interpolate each Z_{ij} / Z_V^2 factor to get Z(3 GeV). Fit the functional form in (ap)^2
print('Interpolating Z_ij / Z_V^2 to the matching point of 3 GeV.')
Zij_by_ZVsq = np.einsum('ijqeb,qeb->ijqeb', Z_chiral, ZV_fit ** (-2))
fit_coeffs_interp = []    # at some point we may start using different powers for different Zij / ZV^2
chi2_interp = np.zeros((5, 5, n_ens, n_boot), dtype = np.float64)
Zij_by_ZVsq_interp = np.zeros((5, 5, n_ens, n_boot), dtype = np.float64)
for mult_idx in multiplets:
    # mom_subset = [2, 3, 4, 5, 6, 7]
    # pwr = 3
    mom_subset = [2, 3, 4, 5]
    pwr = 2
    print('Fitting Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1) + ' / ZV^2 up to \mu^' + str(2 * pwr))
    fit_coeffs_b, chi2_b, y_extrap_b = corr_superboot_fit_apsq(mu_list[mom_subset], Zij_by_ZVsq[mult_idx[0], mult_idx[1], mom_subset], order = pwr)
    fit_coeffs_interp.append(fit_coeffs_b)
    chi2_interp[mult_idx[0], mult_idx[1]] = chi2_b
    Zij_by_ZVsq_interp[mult_idx[0], mult_idx[1]] = y_extrap_b

# save results of chiral extrapolation and of interpolation
fchi_out = h5py.File(out_path, 'w')
fchi_out['momenta'] = k_list
fchi_out['Zq/values'] = Zq_fit
fchi_out['Zq/interpCoeffs'] = fit_coeffs_Zq
fchi_out['Zq/interpChi2'] = chi2_Zq
fchi_out['Zq/interpZq'] = y_extrap_Zq
fchi_out['ZV/value'] = ZV_fit
fchi_out['ZA/value'] = ZA_fit
fchi_out['ZqbyZV/interpCoeffs'] = fit_coeffs_ZqZV
fchi_out['ZqbyZV/interpChi2'] = chi2_ZqZV
fchi_out['ZqbyZV/interpZq'] = y_extrap_ZqZV
fchi_out['ZV/interpZV'] = y_extrap_ZV
for ii, mult_idx in enumerate(multiplets):
    fchi_out['Lambda' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = np.einsum('ebq->qeb', Lambda_fit[:, :, :, mult_idx[0], mult_idx[1]])
    fchi_out['Z' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)] = Z_chiral[mult_idx[0], mult_idx[1]]
    datapath = 'O' + str(mult_idx[0] + 1) + str(mult_idx[1] + 1)
    fchi_out[datapath + '/ZijZVm2'] = Zij_by_ZVsq[mult_idx[0], mult_idx[1]]    # this is the full dataset with p^2 dependence
    fchi_out[datapath + '/interpCoeffs'] = fit_coeffs_interp[ii]
    fchi_out[datapath + '/interpChi2'] = chi2_interp[mult_idx[0], mult_idx[1]]
    fchi_out[datapath + '/interpZijZVm2'] = Zij_by_ZVsq_interp[mult_idx[0], mult_idx[1]]
print('Chiral extrapolation saved at: ' + out_path)

fchi_out.close()
