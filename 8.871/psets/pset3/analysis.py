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
from fittools import *
import plottools as pt

corr_path = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/meas/correlators.h5'
plot_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/figs/'
np.random.seed(20)                 # seed the RNG for reproducibility
Nc = 3
Nd = 4
TT = 32
LL = [16, 16, 16, TT]
beta = 6.0
# grid = gpt.grid(LL, gpt.double)

f = h5py.File(corr_path, 'r')
mrho = np.real(bootstrap(f['m/rho'][()], Nb = n_boot))
scale_set = np.real(bootstrap(f['scaleset'][()], Nb = n_boot))
f.close()

#####################################################################
########################### SET THE SCALE ###########################
#####################################################################

tstart = 0.0
eps = 0.1
n_smear = 50
scale_times = np.array([tstart + (ii + 1) * eps for ii in range(n_smear)])
scale_fit_idxs = np.array(list(range(25, 35)))
scale_fit_times = scale_times[scale_fit_idxs]
scale_data_fit = scale_set[:, scale_fit_idxs]
print('Fitting scale setting data at scales: ' + str(scale_fit_times))
lin_model = Model.power_model(1)
scale_fitter = BootstrapFitter(scale_fit_times, scale_data_fit, lin_model)
scale_fitter.shrink_covar(0.9)
scale_fit_params, _, _, scale_fit_covar = scale_fitter.fit()
print('Fit parameters: ' + str(scale_fit_params))
print('Parameter covariance: ' + str(scale_fit_covar))

c0, c1 = gv.gvar(scale_fit_params, scale_fit_covar)     # c0 + c1 x
c_scale = 0.3
print(c0)
print(c1)
t0_over_asq = (c_scale - c0) / c1
print('t0 / asq = ' + str(t0_over_asq))
ainv = 1.3 * t0_over_asq**(1/2)
print('Scale is: ' + str(ainv) + ' GeV')

#####################################################################
############################ FIT RHO MASS ###########################
#####################################################################

print('Fitting rho mass')
fit_range = np.array(list(range(6, 13)))
const_model = Model.const_model()
rho_fitter = BootstrapFitter(fit_range, mrho[:, fit_range], const_model)
rho_fitter.shrink_covar(0.9)
rho_mass_cv, _, _, rho_mass_cov = rho_fitter.fit()
rho_mass = gv.gvar(rho_mass_cv[0], np.sqrt(rho_mass_cov[0, 0]))
print('Rho mass (lattice units) is: ' + str(rho_mass))

rho_mass_GEV = ainv * rho_mass
print('Rho mass = ' + str(rho_mass_GEV) + ' GeV')

#####################################################################
############################# PLOT FITS #############################
#####################################################################

scale_path = plot_dir + 'scale_setting_fit.pdf'
scale_interp_xx = np.linspace(scale_fit_times[0], scale_fit_times[-1])
scale_interp_yy = c0 + c1 * scale_interp_xx
fig, ax = pt.plot_1d_data(scale_times, np.mean(scale_set, axis = 0), np.std(scale_set, axis = 0, ddof = 1), \
    ax_label = [r'$t$', r'$t^2 E$'], col = 'r', fn_label = 'Data')
ax.fill_between(scale_interp_xx, gv.mean(scale_interp_yy) - gv.sdev(scale_interp_yy), \
    gv.mean(scale_interp_yy) + gv.sdev(scale_interp_yy), color = 'b', alpha = 0.5, linewidth = 0)
pt.save_figure(scale_path)
print('Scale setting saved at: ' + scale_path)

rho_plot_path = plot_dir + 'rho_mass_fit.pdf'
rho_mass_xx = np.linspace(fit_range[0], fit_range[-1])
rho_mass_yy = np.array([rho_mass for ii in rho_mass_xx])
mrho_mu = np.mean(mrho, axis = 0)
mrho_std = np.std(mrho, axis = 0, ddof = 1)
fig, ax = pt.plot_1d_data(range(LL[3]), mrho_mu, mrho_std, ax_label = [r'$t / a$', r'$am_\rho$'], col = 'r', fn_label = 'Data')
ax.fill_between(rho_mass_xx, gv.mean(rho_mass_yy) - gv.sdev(rho_mass_yy), \
    gv.mean(rho_mass_yy) + gv.sdev(rho_mass_yy), color = 'b', alpha = 0.8, linewidth = 0)
pt.save_figure(rho_plot_path)
print('Rho effective mass plotted at: ' + rho_plot_path)
