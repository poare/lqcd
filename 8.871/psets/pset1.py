#####################################################################
# Code for 8.871 Problem Set 1. Note that this uses a variety of    #
# code written for my research which is stored in the               #
# '/Users/theoares/lqcd/utilities' folder. I will include relevant  #
# code snippets in my writeup if they are imported from another     #
# script that I wrote.                                              #
#####################################################################

n_boot = 100
import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *
from formattools import *
import plottools as pt
import lsqfit

out_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps1/figs/'
np.random.seed(20)                 # seed the RNG for reproducibility

probs = [
    '2p2p1',
    '2p2p2',
    # '2p3p1',
    # '3p2p1',
    # '3p2p2',
    '3p2p3',
]

def V_sho(x, a = 0.5, m = 1.0):
    """Simple harmonic oscillator potential."""
    return m * (x**2) / 2.

def V_cubic(x, a = 0.5, g = 1.0):
    return g * (x**3) / 3

def get_site_action(V = V_sho):
    def site_action(j, x, m = 1., a = 0.5):
        """
        Computes the part of the action that depends on the value 
        of x at the site j.

        Parameters
        ----------
        j : int
            Site x has been updated at.
        x : np.array[N]
            Discretized path of length N.
        V : function
            Potential function for the action.
        m : float (default = 1.0)
            Mass of the particle.
        a : float (default = 0.5)
            Lattice spacing to use.
        
        Returns
        -------
        float
            Value of S[x] that depends on x[j].
        """
        N = len(x)
        jp, jm = (j + 1) % N, (j - 1) % N
        return (m/a) * x[j] * (x[j] - x[jp] - x[jm]) + a * V(x[j], a, m)
    return site_action

def metropolis(N, n_cfgs, obs = None, delS = get_site_action(), a = 0.5, m = 1.0, eps = 1.4, \
    n_corr = 20, tau = 10):
    """
    Implements the 1D Metropolis algorithm to sample the distribution 
    exp(-S[x]).

    Parameters
    ----------
    N : int
        Number of sites for discretization.
    n_cfgs : int
        Number of configurations to generate.
    obs : function (default = None)
        Observable to compute on each configuration. If None, 
        doesn't compute an observable.
    n_corr : int (default = 20)
        Correlation length between different generated configurations. 
    tau : int (default = 10)
        Thermalization time, in units of n_corr.
    eps : float (default = 1.4)
        Average fluctuation size for Metropolis algorithm.
    
    Returns
    -------
    np.array[n_cfgs, N]
        n_cfgs generated configurations.
    list
        Observable values on each configuration. If obs is None, 
        returns an empty list.
    float 
        Accept/reject ratio of Metropolis algorithm (after thermalization).
    """
    n_therm = tau * n_corr
    x = np.zeros((N), dtype = np.float64)
    x_lst = np.zeros((n_cfgs, N), dtype = np.float64)
    obs_lst = []
    for ii in range(n_therm):                           # thermalize
        x, _ = update(x, delS, a, m, eps)
    print('Thermalization complete.')
    acc_ratios = []
    for cfg_idx in range(n_cfgs):
        for ii in range(n_corr):
            x, n_acc = update(x, delS, a, m, eps)
            acc_ratios.append(n_acc / N)
        x_lst[cfg_idx] = x                              # save cfg
        if obs is not None:
            obs_lst.append(obs(x))
        # print('Configuration ' + str(cfg_idx) + ' generated.')
    return x_lst, np.array(obs_lst), np.array(acc_ratios)

def update(y, delS = get_site_action(), a = 0.5, m = 1.0, eps = 1.4):
    """Updates x at every site on the lattice."""
    n_acc = 0
    x = np.copy(y)                  # only mutate local variable
    # print('Sweeping lattice')
    for j in range(len(x)):
        old_x = x[j]
        old_Sj = delS(j, x, m, a)
        x[j] = x[j] + np.random.uniform(low = -eps, high = eps)
        dS = delS(j, x, m, a) - old_Sj
        if dS > 0 and np.exp(-dS) < np.random.uniform():
            x[j] = old_x
        else:
            n_acc += 1
    return x, n_acc

def twopt_all(x):
    """Computes all time-averaged two-point functions for a path x."""
    return np.array([twopt_site(x, n) for n in range(len(x))])

def twopt_site(x, n):
    """Computes the time-averaged two-point function for a path x with separation n."""
    return np.average(np.roll(x, -n) * x)

def twopt_cubed_all(x):
    """Computes all time-averaged two-point functions of x**3 for a path x."""
    return np.array([twopt_cubed_site(x, n) for n in range(len(x))])

def twopt_cubed_site(x, n):
    """Computes the time-averaged two-point function of x**3 for a path x with separation n."""
    return np.average(np.roll(x**3, -n) * (x**3))

def run_metropolis(n_cfgs, prob, act_params = ['SHO', r'$V = \frac{x^2}{2}$', get_site_action()], \
    observable = twopt_all, gs_energy = 1.0, xlims = [0, 4], ylims = [0, 4], n_corr = 20, a = 0.5):
    """Main loop for running metropolis algorithm. Takes parameters as input."""

    pname1, pname2, act = act_params
    print(pname1 + ', ' + str(n_cfgs) + ' configs, n_corr = ' + str(n_corr) + '.')
    cfgs, corr, acc_ratio = metropolis(N, n_cfgs, delS = act, obs = observable, n_corr = n_corr, a = a)
    print('Average accept ratio per pass: ' + str(np.mean(acc_ratio)))

    corr_bar = np.mean(corr, axis = 0)
    corr_var = (np.sum(corr**2, axis = 0) / n_cfgs - (corr_bar**2)) / n_cfgs
    corr_std = np.sqrt(corr_var)

    corr_gvar = gv.gvar(corr_bar, corr_std)
    dE = np.log(corr_gvar / np.roll(corr_gvar, -1)) / a
    print(dE)

    n_plot = 10             # number of points to plot
    dE_plot = dE[:n_plot]
    x_plot = np.arange(1, n_plot + 1) * a
    fig, ax = pt.plot_1d_data(
        x_plot, gv.mean(dE_plot), gv.sdev(dE_plot), xlims = xlims, \
        ylims = ylims, ax_label = [r'$t/a$', r'$\Delta E(t)$'], \
        title = pname2 + ', ' + str(n_cfgs) + ' configs.', \
        style = styles['prd_twocol'], zorder = 10
    )
    ax = pt.add_line(ax, gs_energy, orientation = 'h', color = 'b', zorder = 0)
    pt.save_figure(out_dir + prob + '/' + pname1 + '_ncfgs_' + str(n_cfgs) + '.pdf')
    return dE, corr

def fit_const(rng, data):
    """
    Fits a constant to data in a given fit region.

    Parameters
    ----------
    rng : np.array
        Range to fit over. 
    data : np.array(gv.gvar)
        Data to fit.

    Returns
    -------
    gv.gvar
        Fit result.
    """
    def const_model(t, p):
        c0 = p['c0']
        return c0 + 0*t
    fit = lsqfit.nonlinear_fit(data = (rng, data), fcn = const_model, p0 = {'c0' : 1.0})
    print('Chi^2 / dof = ' + str(fit.chi2 / fit.dof))
    return fit.p['c0']

#####################################################################
########################### PROBLEM 2.2.1 ###########################
#####################################################################

N, a = 20, 0.5
if '2p2p1' in probs:
    print('Running problem 2.2.1')
    cfgs_list = [25, 100, 1000, 10000]
    dE_list = np.zeros((len(cfgs_list), N), dtype = object)
    for ii, n_cfgs in enumerate(cfgs_list):
        dE_list[ii], _ = run_metropolis(n_cfgs, '2p2p1')
        run_metropolis(n_cfgs, '2p2p1', act_params = ['cubic', r'$V = \frac{x^3}{3}$', get_site_action(V_cubic)], \
            gs_energy = 0.0, ylims = [-2, 2])

#####################################################################
########################### PROBLEM 2.2.2 ###########################
#####################################################################

if '2p2p2' in probs:
    print('Running problem 2.2.2')
    dE_cube_list = np.zeros((len(cfgs_list), N), dtype = object)
    for ii, n_cfgs in enumerate(cfgs_list):
        dE_cube_list[ii], _ = run_metropolis(n_cfgs, '2p2p2', observable = twopt_cubed_all)
        run_metropolis(n_cfgs, '2p2p2', act_params = ['cubic', r'$V = \frac{x^3}{3}$', get_site_action(V_cubic)], \
            observable = twopt_cubed_all, gs_energy = 0.0, ylims = [-2, 2])

    # fit the data to extract ground state energy
    plateau_x = np.arange(6)
    plateau_x_cubed = np.arange(2, 6)
    dE_fit = fit_const(plateau_x, dE_list[-1, plateau_x])
    dE_fit_cube = fit_const(plateau_x_cubed, dE_cube_list[-1, plateau_x_cubed])
    print('Ground state energy from x(t) source / sink: ' + str(dE_fit))
    print('Ground state energy from x(t)^3 source / sink: ' + str(dE_fit_cube))


#####################################################################
########################### PROBLEM 2.3.1 ###########################
#####################################################################

def bin(corr, binsize):
    """
    Bins a correlation function measurement.

    Parameters
    ----------
    corr : np.array(n_cfgs, T)
        Correlation function to bin.
    binsize : int
        Number of configurations per bin. Should divide n_cfgs.
    
    Returns
    -------
    np.array(n_cfgs / n_bin, T)
        Binned correlator.
    """
    n_cfgs, T = corr.shape
    assert n_cfgs % binsize == 0
    corr_binned = []
    for ii in range(0, n_cfgs, binsize):
        corr_avg = 0
        for jj in range(binsize):
            corr_avg = corr_avg + corr[ii + jj] 
        corr_binned.append(corr_avg / binsize)
    return np.array(corr_binned)

if '2p3p1' in probs:
    print('Running problem 2.3.1')
    N, a = 20, 0.5
    n_corr, n_bin = 1, 20
    cfgs_list = [100, 1000, 10000]
    dE_list = np.zeros((len(cfgs_list), N), dtype = object)
    corrs = [0] * len(cfgs_list)
    for ii, n_cfgs in enumerate(cfgs_list):
        dE_list[ii], corrs[ii] = run_metropolis(n_cfgs, '2p3p1', n_corr = n_corr)

        # bin and rerun calculation
        corr_bin = bin(corrs[ii], n_bin)
        num_bins = corr_bin.shape[0]

        corr_bar = np.mean(corr_bin, axis = 0)
        corr_var = (np.sum(corr_bin**2, axis = 0) / num_bins - (corr_bar**2)) / num_bins
        corr_std = np.sqrt(corr_var)

        corr_gvar = gv.gvar(corr_bar, corr_std)
        dE = np.log(corr_gvar / np.roll(corr_gvar, -1)) / a
        print(dE)

        n_plot = 10             # number of points to plot
        dE_plot = dE[:n_plot]
        x_plot = np.arange(1, n_plot + 1) * a
        fig, ax = pt.plot_1d_data(
            x_plot, gv.mean(dE_plot), gv.sdev(dE_plot), xlims = [0, 4], \
            ylims = [0, 4], ax_label = [r'$t/a$', r'$\Delta E(t)$'], \
            title = r'$V = \frac{x^2}{2}$, ' + str(n_cfgs) + ' configs.', \
            style = styles['prd_twocol'], zorder = 10
        )
        ax = pt.add_line(ax, 1.0, orientation = 'h', color = 'b', zorder = 0)
        pt.save_figure(out_dir + '2p3p1/SHO_binned_ncfgs_' + str(n_cfgs) + '.pdf')


#####################################################################
########################### PROBLEM 3.2.1 ###########################
#####################################################################

def delta(x, a = 0.5):
    """Finite central-value difference operator."""
    return (np.roll(x, 1) + np.roll(x, -1) - 2 * x) / (a**2)

def get_site_action_improved(V = V_sho):
    site_action = get_site_action(V)
    def site_action_improved(j, x, m = 1.0, a = 0.5):
        """Order a**2 improved SHO action for a single site j."""
        N = len(x)
        jp2, jm2 = (j + 2) % N, (j - 2) % N
        jp, jm = (j + 1) % N, (j - 1) % N
        return site_action(j, x, m, a) - m / (24 * a) * x[j] * (
            2 * x[jp2] - 8 * x[jp] + 6 * x[j] - 8 * x[jm] + 2 * x[jm2]
        )
    return site_action_improved

if '3p2p1' in probs:
    print('Running problem 3.2.1')
    n_cfgs = 1000
    run_metropolis(n_cfgs, '3p2p1', act_params = ['improved_action', r'$S^{(\mathrm{imp})}[x(t)]$', get_site_action_improved()])

#####################################################################
########################### PROBLEM 3.2.2 ###########################
#####################################################################

def V_imp(x, a = 0.5, m = 1.0):
    """Improved SHO potential with no ghosts."""
    return (m * (x**2) / 2.) * (1 + (a**2) / 12)

if '3p2p2' in probs:
    print('Running problem 3.2.2')
    n_cfgs = 1000
    run_metropolis(n_cfgs, '3p2p2', act_params = ['improved_action', r'$\tilde{S}^{(\mathrm{imp})}[x(t)]$', get_site_action(V_imp)])


#####################################################################
########################### PROBLEM 3.2.3 ###########################
#####################################################################

def V_imp_aharmonic(x, a = 0.5, m = 1.0):
    """Improved anharmonic oscillator potential with no ghosts."""
    c = 2.0
    dv = c * m * (x**2) / 4
    return (m * (x**2) / 2.) * (1 + c * m * (x**2)) \
        + (a**2) * m * ((x + 2 * c * m * (x**3))**2) / 24 \
        - a * dv + (a**3) * (dv**2) / 2

if '3p2p3' in probs:
    print('Running problem 3.2.3')
    n_cfgs = 10000
    dE0p5, _ = run_metropolis(n_cfgs, '3p2p3', act_params = ['anharmonic_ahalf', r'$\tilde{S}^{(\mathrm{anharm})}[x(t)]$', get_site_action(V_imp_aharmonic)], \
        gs_energy = 1.933)
    dE0p25, _ = run_metropolis(n_cfgs, '3p2p3', act_params = ['anharmonic_aquarter', r'$\tilde{S}^{(\mathrm{anharm})}[x(t)]$', get_site_action(V_imp_aharmonic)], \
        gs_energy = 1.933, a = 0.25, xlims = [0, 2])
    
    fit_rng = np.arange(0, 6)
    dE0p5_fit = fit_const(fit_rng, dE0p5[fit_rng])
    print('\Delta E for a = 0.5 is: ' + str(dE0p5_fit))
    dE0p25_fit = fit_const(fit_rng, dE0p25[fit_rng])
    print('\Delta E for a = 0.25 is: ' + str(dE0p25_fit))
