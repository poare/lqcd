# Measures observables on the given gauge field configurations.

import rhmc
import numpy as np
import h5py
# from pfapack import pfaffian as pf

# Macbook
util_path = '/Users/theoares/lqcd/utilities'

# Desktop
# util_path = ''

import sys
sys.path.append(util_path)
import formattools as ft
import plottools as pt

style = ft.styles['prd_twocol']
pt.set_font()

# cfg_dir = '/Users/theoares/Dropbox (MIT)/research/2d_adjoint_qcd/meas/tests/wilson_Nc2_4_4_b2p0_k0p125_cold'
cfg_dir = '/Users/theoares/Dropbox (MIT)/research/2d_adjoint_qcd/meas/tests/wilson_Nc2_4_4_b2p0_k0p125_hot'
cfg_path = f'{cfg_dir}/cfgs.h5'

f = h5py.File(cfg_path, 'r')
U = f['U'][()]
f.close()
(ncfgs, d, L, T, Nc, _) = U.shape
kappa = 0.125
gens = rhmc.get_generators(Nc)
Lat = rhmc.Lattice(L, T)
bcs = (1, -1)

# Get observables
observables = [
    lambda U : np.sum(rhmc.plaquette(U)),
    rhmc.polyakov_loop,
    rhmc.topological_charge,
    rhmc.get_pf_observable(kappa, gens, lat = Lat, bcs = bcs)
]
obs_names = [
    'plaquette',
    'polyakov',
    'top_charge',
    'pfaffian'
]
obs_labels = [
    r'P',
    r'P_{\mathrm{Polyakov}}',
    r'Q',
    r'\mathrm{Pf}\,D_W[U]'
]

for name, label, observable in zip(obs_names, obs_labels, observables):
    obs = np.zeros((ncfgs), dtype = U.dtype)
    for cfg in range(ncfgs):
        if cfg % 100 == 0:
            print(f'Computing {name} on configuration {cfg}.')
        obs[cfg] = observable(U[cfg])
    obs_dir = f'{cfg_dir}/{name}.pdf'
    obs_avg = np.array([np.mean(obs[i:i+20]) for i in range(0, 950)])
    pt.plot_1d_data(np.arange(950), obs_avg, fn_label = label, legend = True, ax_label = ['Trajectory', r'$'+label+r'$'], saveat_path = obs_dir)

# import matplotlib.pyplot as plt
# import seaborn as sns
# fig, ax = plt.subplots(1)
# sns.histplot(np.real(obs), ax = ax)
# fig.savefig(f'{cfg_dir}/hist.pdf')

# # Polyakov loop
# polyakov = np.zeros((ncfgs), dtype = U.dtype)
# for cfg in range(ncfgs):
#     polyakov[cfg] = rhmc.polyakov_loop(U[cfg])
# poly_dir = f'{cfg_dir}/polyakov.pdf'
# pt.plot_1d_data(np.arange(ncfgs), np.real(polyakov), ax_label = ['Trajectory', r'$\sum_x\, \mathrm{Tr}\,\prod_{t = 0}^T U_t(x, t)$'], saveat_path = poly_dir)

