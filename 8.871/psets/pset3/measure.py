#####################################################################
# Code for 8.871 Problem Set 3. Note that this uses a variety of    #
# code written for my research which is stored in the               #
# '/Users/theoares/lqcd/utilities' folder. I will include relevant  #
# code snippets in my writeup if they are imported from another     #
# script that I wrote.                                              #
#                                                                   #
# Note that I will use GPT to generate configurations and compute   #
# observables on these configurations.                              #
#####################################################################

n_boot = 100
import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *
from formattools import *
import plottools as pt

import gpt

cfg_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/cfgs/'
out_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/meas/'
plot_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/figs/'
np.random.seed(20)                 # seed the RNG for reproducibility
Nc = 3
Nd = 4
TT = 32
LL = [16, 16, 16, TT]
beta = 6.0
grid = gpt.grid(LL, gpt.double)

cfgs = []
for (dirpath, dirnames, file) in os.walk(cfg_dir):
    cfgs.extend(file)
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = cfg_dir + cfgs[idx]

# Select configs to use
cfgs = cfgs[::3]        # take every 3 configs (I generated a lot of them)
print(cfgs)

n_cfgs = len(cfgs)
print('Performing measurements on ' + str(n_cfgs) + ' configs.')

# Scale setting params
tstart = 0.0
eps = 0.1
n_smear = 50
plot_x = np.array([tstart + (ii + 1) * eps for ii in range(n_smear)])
plot_y = np.zeros((n_cfgs, n_smear), dtype = np.float64)

#####################################################################
######################## COMPUTE CORRELATORS ########################
#####################################################################

pi_corrs = np.zeros((n_cfgs, TT), dtype = np.complex64)
rho_corrs = np.zeros((3, n_cfgs, TT), dtype = np.complex64)
mpi = np.zeros((n_cfgs, TT), dtype = np.complex64)
mrho = np.zeros((n_cfgs, TT), dtype = np.complex64)
for ii, cfg in enumerate(cfgs):
    print('Measuring configuration ' + cfg)
    U = gpt.load(cfg)

    print('Setting scale.')
    U_wf = gpt.copy(U)
    cfg_plot_y = []
    t = tstart
    for k in range(n_smear):
        U_wf = gpt.qcd.gauge.smear.wilson_flow(U_wf, epsilon = eps)
        t += eps
        E = gpt.qcd.gauge.energy_density(U_wf)
        gpt.message(f"t^2 E(t={t:g})={t**2. * E}")
        plot_y[ii, k] = t**2. * E

    ferm = gpt.qcd.fermion.wilson_clover(U, {
        'kappa' : 0.11,
        'boundary_phases': [1.0, 1.0, 1.0, -1.0],
        'csw_r' : 0.0,
        'csw_t' : 0.0,
        'use_legacy' : False,
        'isAnisotropic' : False,
        'nu' : 0.0,         # what is this and what is xi_0?
        'xi_0' : 0.0
    })

    print('Inverting propagator:')
    pc = gpt.qcd.fermion.preconditioner
    inv = gpt.algorithms.inverter
    gpt.default.push_verbose("cg_convergence", True) # want to see CG progress
    Q = ferm.propagator(inv.preconditioned(pc.eo2_ne(), inv.cg({"eps": 1e-6, "maxiter": 100}))).grouped(1)

    src = gpt.mspincolor(grid)
    gpt.create.point(src, [0, 0, 0, 0])

    prop_field = gpt.eval( Q * src )

    corr = gpt.slice( gpt.trace(prop_field * gpt.adj(prop_field)), 3)
    pi_corrs[ii] = np.array(corr, dtype = np.complex64)
    mpi[ii] = np.array([np.log(corr[t]/corr[(t + 1) % TT]).real for t in range(TT)])

    for k in range(3):
        print(gpt.gamma[k].tensor())
        rcorr = gpt.slice(gpt.trace(gpt.gamma[5].tensor() * gpt.gamma[k].tensor() * prop_field \
            * gpt.gamma[k].tensor() * gpt.gamma[5].tensor() * gpt.adj(prop_field)), 3)
        rho_corrs[k, ii] = np.array(rcorr)
    rho_corr = np.mean(rho_corrs[:, ii, :], axis = 0)
    mrho[ii] = np.array([np.log(rho_corr[t] / rho_corr[(t + 1) % TT]).real for t in range(TT)])

# Plot scale setting
scale_path = plot_dir + 'scale_setting.pdf'
pt.plot_1d_data(plot_x, np.mean(plot_y, axis = 0), np.std(plot_y, axis = 0, ddof = 1), \
    ax_label = [r'$t$', r'$t^2 E$'], col = 'r', saveat_path = scale_path)
print('Scale setting saved at: ' + scale_path)

out_path = out_dir + 'correlators.h5'
f = h5py.File(out_path, 'w')
f['C2/pi'] = pi_corrs
f['m/pi'] = mpi
f['C2/rho'] = rho_corrs
f['m/rho'] = mrho
f['scaleset'] = plot_y
f.close()
print('Correlators written to: ' + out_path)

pi_plot_path = plot_dir + 'mpi.pdf'
mpi_b = np.real(bootstrap(mpi))
mpi_mu = np.mean(mpi_b, axis = 0)
mpi_std = np.std(mpi_b, axis = 0, ddof = 1)
pt.plot_1d_data(range(LL[3]), mpi_mu, mpi_std, ax_label = [r'$t / a$', r'$am_\pi$'], col = 'r', saveat_path = pi_plot_path)
print('Pion effective mass plotted at: ' + pi_plot_path)

rho_plot_path = plot_dir + 'rho_mass.pdf'
rho_b = np.real(bootstrap(mrho))
mrho_mu = np.mean(mrho, axis = 0)
mrho_std = np.std(mrho, axis = 0, ddof = 1)
pt.plot_1d_data(range(LL[3]), mrho_mu, mrho_std, ax_label = [r'$t / a$', r'$am_\rho$'], col = 'b', saveat_path = rho_plot_path)
print('Rho effective mass plotted at: ' + rho_plot_path)