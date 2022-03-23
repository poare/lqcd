import numpy as np
from scipy.optimize import root
import h5py
import os
from scipy.special import zeta
import time
import re
import itertools
import sys

from utils import *

Pi_Pi = '/Users/theoares/Pi_Pi/'

"""
Run sd_analysis on a given ensemble from the command line with python3 short_distance_analysis.py ${cfg_idx}.
If you run the script sd_run.sh from the command line, it will run all 5 ensembles.
"""
ens_list = [
    '24I/ml_0p01',
    '24I/ml_0p005',
    '32I/ml0p008',
    '32I/ml0p006',
    '32I/ml0p004'
]
ens_idx = int(sys.argv[1])
ensemble = ens_list[ens_idx]
print('Running ensemble: ' + ensemble)

"""
Alternatively, choose an ensemble to run analysis by uncommenting one of the following ensembles.
"""
# ensemble = '24I/ml_0p01'
# ensemble = '24I/ml_0p005'
# ensemble = '32I/ml0p008'
# ensemble = '32I/ml0p006'
# ensemble = '32I/ml0p004'

resultsA = Pi_Pi + ensemble + '/resultsA'
resultsA_tavg = resultsA + '.tavg'
if ensemble[:3] == '24I':
    dt = 40
    l, t, Ls = 24, 64, 16
    ms = 0.04                           # lattice units
    ainv, sig_ainv = 1.784, 5           # GeV
else:
    dt = 20
    l, t, Ls = 32, 64, 16
    ms = 0.03                           # lattice units
    ainv, sig_ainv = 2.382, 8           # GeV

if ensemble == '24I/ml_0p01':
    tmin, tmax = 1460, 3500         # should have 52 files, matches with the number of lines in mpi_jacks.dat
    ml = 0.01               # lattice units
    m_pi = 432.2            # MeV
    sig_m_pi = 1.4          # MeV
elif ensemble == '24I/ml_0p005':
    tmin, tmax = 900, 2980          # should have 53 files
    ml = 0.005               # lattice units
    m_pi = 339.6            # MeV
    sig_m_pi = 1.2          # MeV
elif ensemble == '32I/ml0p008':
    # tmin, tmax = 520, 1180
    tmin, tmax = 520, 1160      # The pion mass fits only go up to file 1160, not to 1180 (check fits/xml/fit_mres.xml)
    ml = 0.008               # lattice units
    m_pi = 410.8            # MeV
    sig_m_pi = 1.5          # MeV
elif ensemble == '32I/ml0p006':
    # tmin, tmax = 1000, 1840         # should be 43 files
    tmin, tmax = 1000, 1820           # fits only go up to 1820
    ml = 0.006               # lattice units
    m_pi = 359.7            # MeV
    sig_m_pi = 1.2          # MeV
else:
    tmin, tmax = 520, 1460          # should be
    ml = 0.004               # lattice units
    m_pi = 302.0            # MeV
    sig_m_pi = 1.1          # MeV
L = Lattice(l, t)

filenums = [i for i in range(tmin, tmax + 1, dt)]
n_ops = 5

# If you need to remove a specific file because of input problems, edit this code
if ensemble == '32I/ml0p004':
    print('Deleting ensembleuration 840')
    err_idx = int((840 - tmin) / dt)
    del filenums[err_idx]

# print(len(filenums))    # uncomment if we want the number of files
# raise Exception('Stopping')

################################################################################
############################# TWO POINT ANALYSIS ###############################
################################################################################
print('2 point analysis')
# stem_2pt = 'pion-00WW'          # <PP> correlator with wall sources
# C2pt = read_Npt(resultsA, stem_2pt, filenums, 2, L.T)
# C2_tavg = np.mean(C2pt, axis = 1)               # t avg over first (source) time-- Should equal what's in resultsA.tavg directory
# print('2pt shape before bootstrap: ' + str(C2_tavg.shape))
# C2_boot = bootstrap(C2_tavg, data_type = np.complex64)

stems = ['pion-00WW', 'pion-00WP', 'fp-00WW', 'fp-00WP']
C2_stem_boot = {}
for stem in stems:
    print('Reading 2pt from ' + stem)
    C2pt_stem = read_Npt(resultsA, stem, filenums, 2, L.T)
    C2_tavg_stem = np.mean(C2pt_stem, axis = 1)               # t avg over first (source) time-- Should equal what's in resultsA.tavg directory
    C2_stem_boot[stem] = bootstrap(C2_tavg_stem, data_type = np.complex64)
C2_boot = C2_stem_boot['pion-00WW']         # use wall-wall PP correlator for 3pt analysis

# Correlators for A and curly A are in the same file-- looks like data is (Re[A], Im[A], Re[curlyA], Im[curlyA])
dwf_stem = 'za_' + str(ml)
print('Reading 2pt from ' + dwf_stem + ' at first position')
C2pt_A = read_Npt(resultsA, dwf_stem, filenums, 2, L.T)
C2_tavg_A = np.mean(C2pt_A, axis = 1)
C2_A_boot = bootstrap(C2_tavg_A, data_type = np.complex64)

print('Reading 2pt from ' + dwf_stem + ' at second position')
C2pt_curlyA = read_Npt(resultsA, dwf_stem, filenums, 2, L.T, start_idx = 4)
C2_tavg_curlyA = np.mean(C2pt_curlyA, axis = 1)
C2_curlyA_boot = bootstrap(C2_tavg_curlyA, data_type = np.complex64)

# Extract pion mass. Fits should be averaged over all stems
# print('Fitting pion mass at one range of times')
# m_eff = get_cosh_effective_mass(C2_boot)
# meff_fit_range = range(10, 25)
# mpi_boot = fit_constant(meff_fit_range, m_eff)[0]
# mpi_mu = np.mean(mpi_boot)
# mpi_sigma = np.std(mpi_boot, ddof = 1)
# print('Pion mass = ' + str(mpi_mu) + ' \\pm ' + str(mpi_sigma))

# time average and get pion mass
print('Fitting pion mass over all ranges and stems.')
meff_fns = [get_cosh_effective_mass, get_cosh_effective_mass, get_sinh_effective_mass, get_sinh_effective_mass]
meff_ens_list, weight_list = [], []
for idx, stem in enumerate(stems):
    meff_stem = meff_fns[idx](C2_stem_boot[stem])
    meff_stem_folded = fold_meff(meff_stem, L.T)
    fits_stem, stats_stem, meff_stem_ens, weights_stem = fit_constant_allrange(meff_stem_folded)
    ampi_stem_bar, ampi_stem_sigma, ampi_stem_boot = analyze_accepted_fits(meff_stem_ens, weights_stem)
    print('meff on ' + stem + ', avg over all fit ranges with cut ϵ = 0.01 on pvals: ' + str(ampi_stem_bar) \
          + ' \pm ' + str(ampi_stem_sigma))
    meff_ens_list.append(meff_stem_ens)
    weight_list.append(weights_stem)
meff_all = np.concatenate((meff_ens_list[0], meff_ens_list[1], meff_ens_list[2], meff_ens_list[3]), axis = 0)
weights_all = np.concatenate((weight_list[0], weight_list[1], weight_list[2], weight_list[3]), axis = 0)

# TODO this is probably not propagating the stats right
mpi_boot = weighted_sum_bootstrap(meff_all, weights_all)
meff_boots_mu = np.mean(mpi_boot)
meff_boots_sigma = np.std(mpi_boot, ddof = 1)
print('Average pion mass over bootstrapped samples: ' + str(meff_boots_mu) + ' \pm ' + str(meff_boots_sigma))

# C2_neg_mode = np.zeros(C2_tavg.shape, dtype = np.complex64)         # subtract off the positive frequency mode to isolate the exponential decay exp(-mt)
# for t in range(L.T):
#     C2_neg_mode[:, t] = C2_tavg[:, t] - (1/2) * C2_tavg[:, L.T // 2] * np.exp(pi_masses[:] * (t - L.T // 2))
C2_neg_mode = np.zeros(C2_boot.shape, dtype = np.complex64)         # subtract off the positive frequency mode to isolate the exponential decay exp(-mt)
for t in range(L.T):
    C2_neg_mode[:, t] = C2_boot[:, t] - (1/2) * C2_boot[:, L.T // 2] * np.exp(mpi_boot[:] * (t - L.T // 2))

################################################################################
############################ THREE POINT ANALYSIS ##############################
################################################################################

print('3 point')
stem_3pt = 'pion_0vbb_4quark'
C3pt = read_Npt(resultsA, stem_3pt, filenums, 3, L.T)
print('3pt shape before bootstrap: ' + str(C3pt.shape))
C3pt = bootstrap(C3pt, data_type = np.complex64)

# time average
# n_files = C3pt.shape[0]   # Should just be n_boot
n_rows = 24               # if we don't contract immediately
# C3pt_tavg = np.zeros((n_rows, L.T, L.T), dtype = np.complex64)    # (tx, |t+ - t-|)
tavg_shape = C3pt.shape[:-1]             # (n_files, n_rows, L.T, L.T)
C3pt_tavg = np.zeros(tavg_shape, dtype = np.complex64)    # (tx, |t+ - t-|)

cnt = np.zeros((L.T, L.T))
for tminus, tx, tplus in itertools.product(range(L.T), repeat = 3):
    C3pt_tavg[:, :, tminus, tplus] += C3pt[:, :, tminus, tx, tplus]    # need to average about tminus = 0 first
    cnt[tminus, tplus] += 1
for tminus, tplus in itertools.product(range(L.T), repeat = 2):
    C3pt_tavg[:, :, tminus, tplus] = C3pt_tavg[:, :, tminus, tplus] / cnt[tminus, tplus]

# reflect time averaged data about midpoint
C3pt_refl = np.zeros(tavg_shape, dtype = np.complex64)
for t1, t2 in itertools.product(range(L.T), repeat = 2):
    C3pt_refl[:, :, t1, t2] = C3pt_tavg[:, :, (L.T - t1) % L.T, (L.T - t2) % L.T]

# confirm against David's data that C3pt_tavg has indices [cfg_num, op_num, sep, t] where sep is in the name of the file in results.tavg
# for idx, fnum in enumerate(range(tmin, tmax + 1, dt)):
#     print('Checking file ' + str(fnum))
#     for sep in range(L.T):
#         fname = resultsA_tavg + '/pion_0vbb_4quark.' + str(sep) + '.' + str(fnum)
#         data = np.genfromtxt(fname)
#         for op_idx in range(24):
#             for t in range(L.T):
#                 my_entry = np.real(C3pt_tavg[idx, op_idx, sep, t])
#                 david_entry = data[t, 2 * op_idx + 1]
#                 delta = np.abs(my_entry - david_entry)
#                 if delta / np.abs(my_entry) > 1e-4 and not (np.abs(my_entry) < 1e-2 and np.abs(david_entry) < 1e-2):
#                     print((my_entry, david_entry))
#                     print('No match at: ' + str(sep) + '.' + str(fnum) + ' on line t = ' + str(t))
# print('Check complete.')


# match these against david's data in resultsA.tavg to check if it works correctly.
# print("Time averaged C3pt in pion_0vbb_4quark.34.1660 at line t = 25:")
# print(C3pt_tavg[5, :, 34, 25])                   # check pion_0vbb_4quark.18.1580 at label t = 7
# print("Reflected C3pt-ref in pion_0vbb_4quark-ref.9.1500 at line t = 3")
# print(C3pt_refl[1, :, 9, 3])             # check pion_0vbb_4quark-ref.44.1500 at label t = 12

# To get the data at pion_0vbb_4quark.SEP.CFG on line t
# at operator index i, index the corresponding C3pt_tavg or C3pt_refl at [CFG, i, SEP, t]
# So SEP = tminus, t = tplus (I don't know why he named it this)

# Get operator structures from C3pt_tavg and C3pt_refl.
Cbar = (C3pt_tavg + C3pt_refl) / 2          # average time-averaged with time reflected data
Cnpt = np.zeros((5, C3pt.shape[0], L.T, L.T), dtype = np.complex64)
cont = {
    'SS' : [Cbar[:, i, :, :] for i in range(4)],
    'PP' : [Cbar[:, i, :, :] for i in range(4, 8)],
    'VV' : [Cbar[:, i, :, :] for i in range(8, 12)],
    'VA' : [Cbar[:, i, :, :] for i in range(12, 16)],
    'AV' : [Cbar[:, i, :, :] for i in range(16, 20)],
    'AA' : [Cbar[:, i, :, :] for i in range(20, 24)]
}
Cnpt[0] = -(1./4.) * (cont['VV'][0] - cont['VV'][1] - cont['AV'][0] + cont['AV'][1] + cont['VA'][0] - cont['VA'][1] - cont['AA'][0] + cont['AA'][1])
Cnpt[1] = -(1./2.) * (cont['SS'][0] - cont['SS'][1] + cont['PP'][0] - cont['PP'][1])
Cnpt[2] = -(1./2.) * (cont['VV'][0] - cont['VV'][1] + cont['AA'][0] - cont['AA'][1])
Cnpt[3] = -(1./4.) * (cont['VV'][2] - cont['VV'][3] - cont['AV'][2] + cont['AV'][3] + cont['VA'][2] - cont['VA'][3] - cont['AA'][2] + cont['AA'][3])
Cnpt[4] = -(1./2.) * (cont['SS'][2] - cont['SS'][3] + cont['PP'][2] - cont['PP'][3])

Cnpt = np.einsum('ifst->fits', Cnpt)                # reindex to match David's code for reading off later
print("Constructed operator contractions")

# construct R ratio. Cnpt -> (CFG, op_idx, tplus, tminus) where (tminus, tx, tplus) is how the columns in the
# original file were read off. tminus is identified with the separation time sep in david's code, and tplus
# is the time t in his code.
R = np.zeros((n_boot, n_ops, L.T, L.T), dtype = np.complex64)
for bidx in range(n_boot):
    for i in range(n_ops):
        for t, delta in itertools.product(range(L.T), repeat = 2):
            R[bidx, i, t, delta] = 2 * mpi_boot[bidx] * Cnpt[bidx, i, t, delta] / C2_neg_mode[bidx, delta]

# Write three point functions here
# add 'nomult' to the path if we aren't multiplying the operators by -2 or -4
f3pt_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/analysis_output/' + ensemble + '/SD_output.h5'
f3pt = h5py.File(f3pt_path, 'w')
f3pt['L'], f3pt['T'] = L.L, L.T
f3pt['ml'], f3pt['ms'] = ml, ms
# f3pt['C2pt'] = C2_boot
f3pt['C3pt'] = C3pt_tavg    # TODO these will all be bootstrapped now
f3pt['C3pt_ref'] = C3pt_refl
f3pt['Cnpt'] = Cnpt
f3pt['R'] = R
f3pt['mpi'] = mpi_boot

# f3pt[stem_2pt] = C2_boot
for stem in stems:
    f3pt[stem] = C2_stem_boot[stem]

f3pt['za_A'] = C2_A_boot
f3pt['za_curlyA'] = C2_curlyA_boot

f3pt.close()

print('File saved at: ' + f3pt_path)
