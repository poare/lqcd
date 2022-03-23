import numpy as np
from scipy.optimize import root
import h5py
import os
import itertools
from utils import *

# Read the ensemble index. Can also manually set it to choose the ensemble to run on.
ens_idx = int(sys.argv[1])

ensemble = ['24I/ml_0p01', '24I/ml_0p005', '32I/ml0p008', '32I/ml0p006', '32I/ml0p004'][ens_idx]
f3pt_path = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/analysis_output/' + ensemble + '/SD_output.h5'
n_ops = 5
print('Running on ensemble: ' + str(ensemble))

# load correlator data
f = h5py.File(f3pt_path, 'r')
L, T = f['L'][()], f['T'][()]
C2pt_tavg = f['pion-00WW'][()]
C3pt_tavg = f['C3pt'][()]
Cnpt = f['Cnpt'][()]
R_boot = f['R'][()]
f.close()

R_mu = np.mean(R_boot, axis = 0)
R_sigma = np.std(R_boot, axis = 0, ddof = 1)
data_slice = np.zeros((n_boot, n_ops, T), dtype = np.float64)
plot_domain = range(T)
for i in range(n_ops):
    for sep in range(T):
        if sep % 2 == 0:
            data_slice[:, i, sep] = np.real(R_boot[:, i, sep // 2, sep])
        else:
            data_slice[:, i, sep] = np.real((R_boot[:, i, sep // 2, sep] + R_boot[:, i, sep // 2 + 1, sep]) / 2)
data_plot_mu = np.mean(data_slice, axis = 0)
data_plot_sigma = np.std(data_slice, axis = 0, ddof = 1)

# perform fits
fits = np.zeros((n_boot, n_ops), dtype = np.float64)
f_acc = []
stats_acc = []
R_ens_acc = []
chi2_full = []
weights = []
c = [0, 0, 0, 0, 0]
sigmac = [0, 0, 0, 0, 0]
for i in range(n_ops):
    print('Fitting operator: ' + op_labels[i])
    results = fit_const_allrange(data_slice[:, i, : (T // 2)])
    f_acc.append(results[0])
    stats_acc.append(results[1])
    R_ens_acc.append(results[2])
    chi2_full.append(results[3])
    weights.append(results[4])
    c[i], sigmac[i], fits[:, i] = analyze_accepted_fits(results[2], results[4])

# output fit results
for i in range(n_ops):
    print(latex_labels[i] + ': ' + str(c[i]) + ' \pm ' + str(sigmac[i]))

out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/short_distance/bare_matrix_elements/' + ensemble + '/fit_params.h5'
fout = h5py.File(out_file, 'w')
fout['fits'] = fits
fout['data_slice'] = data_slice
fout['c'] = np.array(c)
fout['sigmac'] = np.array(sigmac)
fout['plot_domain'] = plot_domain
fout.close()

print('Results output to: ' + out_file)
