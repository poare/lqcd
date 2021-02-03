import numpy as np
from scipy.optimize import root
import h5py
import os
from scipy.special import zeta
import time
import re
import itertools

from utils import *

Pi_Pi = '/Users/theoares/Pi_Pi/'

# Choose lattice to run analysis on and initialize parameters
config = '24I/ml_0p01'
# config = '24I/ml_0p005'
# config = '32I/ml0p008'
# config = '32I/ml0p006'
# config = '32I/ml0p004'

resultsA = Pi_Pi + config + '/resultsA'
resultsA_tavg = resultsA + '.tavg'
if config[:3] == '24I':
    dt = 40
    l, t, Ls = 24, 64, 16
    ms = 0.04                           # lattice units
    ainv, sig_ainv = 1.784, 5           # GeV
else:
    dt = 20
    l, t, Ls = 32, 64, 16
    ms = 0.03                           # lattice units
    ainv, sig_ainv = 2.382, 8           # GeV

if config == '24I/ml_0p01':
    tmin, tmax = 1460, 3500         # should have 52 files, matches with the number of lines in mpi_jacks.dat
    ml = 0.01               # lattice units
    m_pi = 432.2            # MeV
    sig_m_pi = 1.4          # MeV
elif config == '24I/ml_0p005':
    tmin, tmax = 900, 2980          # should have 53 files
    ml = 0.005               # lattice units
    m_pi = 339.6            # MeV
    sig_m_pi = 1.2          # MeV
elif config == '32I/ml0p008':
    # tmin, tmax = 520, 1180
    tmin, tmax = 520, 1160      # The pion mass fits only go up to file 1160, not to 1180 (check fits/xml/fit_mres.xml)
    ml = 0.008               # lattice units
    m_pi = 410.8            # MeV
    sig_m_pi = 1.5          # MeV
elif config == '32I/ml0p006':
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
if config == '32I/ml0p004':
    print('Deleting configuration 840')
    err_idx = int((840 - tmin) / dt)
    del filenums[err_idx]

################################################################################
############################# TWO POINT ANALYSIS ###############################
################################################################################
print('2 point')
stem_2pt = 'pion-00WW'          # <PP> correlator with wall sources
C2pt = read_Npt(resultsA, stem_2pt, filenums, 2, L.T)
C2_tavg = np.mean(C2pt, axis = 1)               # t avg over first (source) time-- Should equal what's in resultsA.tavg directory
fold_fn = np.add if (stem_2pt == 'pion-00WW' or stem_2pt == 'pion-00WP') else np.subtract
C2_fold = fold(C2_tavg, L.T, fold_fn)            # fold about midpoint. Might not need this for the 3pt analysis

pi_masses = np.genfromtxt(Pi_Pi + config + '/fits/results/fit_params/mpi_jacks.dat')
if config == '32I/ml0p004':
    pi_masses = np.delete(pi_masses, err_idx)

C2_neg_mode = np.zeros(C2_tavg.shape, dtype = np.complex64)         # subtract off the positive frequency mode to isolate the exponential decay exp(-mt)
for t in range(L.T):
    C2_neg_mode[:, t] = C2_tavg[:, t] - (1/2) * C2_tavg[:, L.T // 2] * np.exp(pi_masses[:] * (t - L.T // 2))

################################################################################
############################ THREE POINT ANALYSIS ##############################
################################################################################

print('3 point')
stem_3pt = 'pion_0vbb_4quark'
C3pt = read_Npt(resultsA, stem_3pt, filenums, 3, L.T)
# C3pt = read_Npt(resultsA_tavg, stem_3pt, filenums, 3, L.T)        # if we want to use t avged readin

# time average
# n_rows = 5                # if we do contract into operator structures when reading in
n_files = C3pt.shape[0]
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
# 4 multiplies C because these are for SS and PP, they don't have a sum on mu
# cont = {
#     'SS' : [4 * Cbar[:, i, :, :] for i in range(4)],
#     'PP' : [4 * Cbar[:, i, :, :] for i in range(4, 8)],
#     'VV' : [Cbar[:, i, :, :] for i in range(8, 12)],
#     'VA' : [Cbar[:, i, :, :] for i in range(12, 16)],
#     'AV' : [Cbar[:, i, :, :] for i in range(16, 20)],
#     'AA' : [Cbar[:, i, :, :] for i in range(20, 24)]
# }
cont = {
    'SS' : [Cbar[:, i, :, :] for i in range(4)],
    'PP' : [Cbar[:, i, :, :] for i in range(4, 8)],
    'VV' : [Cbar[:, i, :, :] for i in range(8, 12)],
    'VA' : [Cbar[:, i, :, :] for i in range(12, 16)],
    'AV' : [Cbar[:, i, :, :] for i in range(16, 20)],
    'AA' : [Cbar[:, i, :, :] for i in range(20, 24)]
}
# TODO figure out why we multiply these by -2 and -4
Cnpt[0] = -2 * (cont['VV'][0] - cont['VV'][1] - cont['AV'][0] + cont['AV'][1] + cont['VA'][0] - cont['VA'][1] - cont['AA'][0] + cont['AA'][1])
Cnpt[1] = -4 * (cont['SS'][0] - cont['SS'][1] + cont['PP'][0] - cont['PP'][1])
Cnpt[2] = -4 * (cont['VV'][0] - cont['VV'][1] + cont['AA'][0] - cont['AA'][1])
Cnpt[3] = -2 * (cont['VV'][2] - cont['VV'][3] - cont['AV'][2] + cont['AV'][3] + cont['VA'][2] - cont['VA'][3] - cont['AA'][2] + cont['AA'][3])
Cnpt[4] = -4 * (cont['SS'][2] - cont['SS'][3] + cont['PP'][2] - cont['PP'][3])

Cnpt = np.einsum('ifst->fits', Cnpt)            # reindex to match David's code for reading off later
print("Constructed operator contractions")

# construct R ratio. Cnpt -> (CFG, op_idx, tplus, tminus) where (tminus, tx, tplus) is how the columns in the
# original file were read off. tminus is identified with the separation time sep in david's code, and tplus
# is the time t in his code.
# R = np.zeros((n_files, n_ops, L.T // 2), dtype = np.complex64)
R = np.zeros((n_files, n_ops, L.T, L.T), dtype = np.complex64)
for fidx in range(n_files):
    for i in range(n_ops):
        # for t in range(L.T // 2):
        #     R[fidx, i, t] = 2 * pi_masses[fidx] * Cnpt[fidx, i, t, 2 * t] / C2_neg_mode[fidx, 2 * t]
        for t, delta in itertools.product(range(L.T), repeat = 2):
            R[fidx, i, t, delta] = 2 * pi_masses[fidx] * Cnpt[fidx, i, t, delta] / C2_neg_mode[fidx, delta]

# Write three point functions here
# add 'nomult' to the path if we aren't multiplying the operators by -2 or -4
f3pt_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/analysis_output/' + config + '/3pt_output.h5'
f3pt = h5py.File(f3pt_path, 'w')
f3pt['L'], f3pt['T'] = L.L, L.T
f3pt['ml'], f3pt['ms'] = ml, ms
f3pt['C2pt'] = C2_tavg
f3pt['C3pt'] = C3pt_tavg
f3pt['C3pt_ref'] = C3pt_refl
f3pt['Cnpt'] = Cnpt
f3pt['R'] = R
f3pt.close()

print('File saved at: ' + f3pt_path)
