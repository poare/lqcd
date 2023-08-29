#####################################################################
# Code for 8.871 Problem Set 2. Note that this uses a variety of    #
# code written for my research which is stored in the               #
# '/Users/theoares/lqcd/utilities' folder. I will include relevant  #
# code snippets in my writeup if they are imported from another     #
# script that I wrote.                                              #
#                                                                   #
# Note that the data structure for a gauge field is a numpy array   #
# of shape (4, Lx, Ly, Lz, Lt, Nc, Nc).                             #
#####################################################################

n_boot = 100
import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *
from formattools import *
import plottools as pt

out_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps2/figs/'
np.random.seed(20)                 # seed the RNG for reproducibility
Nc = 3
d = 4
LL = [8, 8, 8, 8]
n_cfgs = 10

unimproved_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps2/cfgs/unimproved/'
improved_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps2/cfgs/improved/'

start_at = None
lbl_idx = 0
# start_at = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps2/cfgs/improved/cfg5.h5'         # configuration to start generation at.
# lbl_idx = 6                         # corresponding label for start cfg

def dagger(U):
    """Hermitian conjugate of a color matrix."""
    return np.einsum('...ab->...ba', U.conjugate())

def mod(x, NN = LL):
    return tuple([x[ii] % NN[ii] for ii in range(d)])

def staple(U, n, mu):
    """
    Computes the staple of U at n in the \hat{\mu} direction.
    """
    A = np.zeros((Nc, Nc), dtype = np.complex64)
    mu_hat = np.array([1 if rho == mu else 0 for rho in range(d)])
    for nu in range(d):
        if nu == mu:
            continue
        nu_hat = np.array([1 if rho == nu else 0 for rho in range(d)])
        A += U[(nu, *mod(n + mu_hat))] @ dagger(U[(mu, *mod(n + nu_hat))]) @ dagger(U[(nu, *n)])
        A += dagger(U[(nu, *mod(n + mu_hat - nu_hat))]) @ dagger(U[(mu, *mod(n - nu_hat))]) @ U[(nu, *mod(n - nu_hat))]
    return A

def improved_staple(U, n, mu):
    """
    The 'staple' analogue for the improved action consists of 6 strings of 5 link variables each.
    """
    A = np.zeros((Nc, Nc), dtype = np.complex64)
    mu_hat = np.array([1 if rho == mu else 0 for rho in range(d)])
    for nu in range(d):
        if nu == mu:
            continue
        nu_hat = np.array([1 if rho == nu else 0 for rho in range(d)])
        A += U[(mu, *mod(n + mu_hat))] @ U[(nu , *mod(n + 2*mu_hat))] @ dagger(U[(mu , *mod(n + mu_hat + nu_hat))]) \
            @ dagger(U[(mu, *mod(n + nu_hat))]) @ dagger(U[(nu, *mod(n))])
        A += U[(mu, *mod(n + mu_hat))] @ dagger(U[(nu, *mod(n + 2*mu_hat - nu_hat))]) @ dagger(U[(mu, *mod(n - nu_hat + mu_hat))]) \
            @ dagger(U[(mu, *mod(n - nu_hat))]) @ U[(nu, *mod(n - nu_hat))]
        A += U[(nu, *mod(n + mu_hat))] @ U[(nu, *mod(n + mu_hat + nu_hat))] @ dagger(U[(mu, *mod(n + 2*nu_hat))]) \
            @ dagger(U[(nu, *mod(n + nu_hat))]) @ dagger(U[(nu, *mod(n))])
        A += dagger(U[(nu, *mod(n + mu_hat - nu_hat))]) @ dagger(U[(nu, *mod(n + mu_hat - 2*nu_hat))]) \
            @ dagger(U[(mu, *mod(n - 2*nu_hat))]) @ U[(nu, *mod(n - 2*nu_hat))] @ U[(nu, *mod(n - nu_hat))]
        A += U[(nu, *mod(n + mu_hat))] @ dagger(U[(mu, *mod(n + nu_hat))]) @ dagger(U[(mu, *mod(n - mu_hat + nu_hat))]) \
            @ dagger(U[(nu, *mod(n - mu_hat))]) @ U[(mu, *mod(n - mu_hat))]
        A += dagger(U[(nu, *mod(n + mu_hat - nu_hat))]) @ dagger(U[(mu, *mod(n - nu_hat))]) \
            @ dagger(U[(mu, *mod(n - mu_hat - nu_hat))]) @ U[(nu, *mod(n - mu_hat - nu_hat))] @ U[(mu, *mod(n - mu_hat))]
    return A

def wilson_loop_a_by_2a(U, mu, nu):
    """
    Computes the a x a Wilson loop of U in the (mu, nu) direction at every site on the lattice.

    Parameters
    ----------
    U : np.array [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge field array.
    mu : {0, 1, 2, 3}
        mu direction.
    nu : {0, 1, 2, 3}
        nu direction.

    Returns
    -------
    np.array [Lx, Ly, Lz, Lt]
        Wilson loop field (1/3) Tr P_{\mu\nu}(n).
    """
    Un_mu = U[mu]
    Unpmu_mu = np.roll(U[mu], -1, axis = mu)
    Unp2mu_nu = np.roll(U[nu], -2, axis = mu)
    Unpmupnu_mu = np.roll(np.roll(U[mu], -1, axis = mu), -1, axis = nu)
    Unpnu_mu = np.roll(U[mu], -1, axis = nu)
    Un_nu = U[nu]
    return np.real(np.einsum('...ab,...bc,...cd,...de,...ef,...fa->...', 
        Un_mu, Unpmu_mu, Unp2mu_nu, dagger(Unpmupnu_mu), dagger(Unpnu_mu), dagger(Un_nu)
    )) / 3.

def wilson_loop_a_by_a(U, mu, nu):
    """
    Computes the a x a Wilson loop of U in the (mu, nu) direction at every site on the lattice.

    Parameters
    ----------
    U : np.array [4, Lx, Ly, Lz, Lt, Nc, Nc]
        Gauge field array.
    mu : {0, 1, 2, 3}
        mu direction.
    nu : {0, 1, 2, 3}
        nu direction.

    Returns
    -------
    np.array [Lx, Ly, Lz, Lt]
        Wilson loop field (1/3) Tr P_{\mu\nu}(n).
    """
    Un_mu = U[mu]
    Unpmu_nu = np.roll(U[nu], -1, axis = mu)
    Unpnu_mu = np.roll(U[mu], -1, axis = nu)
    Un_nu = U[nu]

    return np.real(np.einsum('...ab,...bc,...cd,...da->...', 
        Un_mu, Unpmu_nu, dagger(Unpnu_mu), dagger(Un_nu)
    )) / 3.

def W(U, r, t, t_axis):
    """
    Measurement code for the Wilson loop W(r, t).
    
    Parameters
    ----------
    r : [r1, r2, r3]
        Spatial coordinate. Moves r_i sites in the i_th spatial direction, where the axes are 
        oriented as [r1, r2, r3] = [0, 1, 2, 3] minus t_axis.
    t : int
        Amount to move in time.
    t_axis : int
        Integer between 0 and 3 that dictates the time axis.
    """
    P = id_field(LL)[0]         # color matrix field, initialized to 1.
    r1, r2, r3 = r

    axes = [0, 1, 2, 3]
    del axes[t_axis]
    r1_axis, r2_axis, r3_axis = axes

    Utmp = U.copy()             # push this lad around
    for i1 in range(r1):
        P = P @ Utmp[r1_axis]
        Utmp = np.roll(Utmp, -1, axis = r1_axis + 1)
    for i2 in range(r2):
        P = P @ Utmp[r2_axis]
        Utmp = np.roll(Utmp, -1, axis = r2_axis + 1)
    for i3 in range(r3):
        P = P @ Utmp[r3_axis]
        Utmp = np.roll(Utmp, -1, axis = r3_axis + 1)
    for it in range(t):
        P = P @ Utmp[t_axis]
        Utmp = np.roll(Utmp, -1, axis = t_axis + 1)
    for i1 in range(r1):
        Utmp = np.roll(Utmp, 1, axis = r1_axis + 1)
        P = P @ dagger(Utmp[r1_axis])
    for i2 in range(r2):
        Utmp = np.roll(Utmp, 1, axis = r2_axis + 1)
        P = P @ dagger(Utmp[r2_axis])
    for i3 in range(r3):
        Utmp = np.roll(Utmp, 1, axis = r3_axis + 1)
        P = P @ dagger(Utmp[r3_axis])
    for it in range(t):
        Utmp = np.roll(Utmp, 1, axis = t_axis + 1)
        P = P @ dagger(Utmp[t_axis])
    return np.average(np.real(np.einsum('...aa->...', P)) / 3.)

def smear(U, n, eps = 1/12, u0 = 0.797, a = 0.25):
    """Smears a gauge field n times with parameter eps."""
    if n == 0:
        return U
    smeared_m1 = smear(U, n - 1, eps, u0, a)
    smeared = smeared_m1 + eps * (a**2) * Delta2(smeared_m1, u0, a)
    for mu in range(4):                                 # project to SU(3)
        for x, y, z, t in itertools.product(*[range(LL[mu]) for mu in range(4)]):
            smeared[mu, x, y, z, t] = proj_SU3(smeared[mu, x, y, z, t])
    return smeared

def Delta2(U, u0 = 0.797, a = 0.25):
    DelU = np.zeros(U.shape, dtype = np.complex64)
    for rho in range(d):
        DelU += Delta2_comp(U, rho)
    return DelU

def Delta2_comp(U, rho, u0 = 0.797, a = 0.25):
    DelU = np.zeros(U.shape, dtype = np.complex64)
    for mu in range(d):
        DelU[mu] = (
            np.einsum('...ab,...bc,...cd->...ad', U[rho], np.roll(U[mu], -1, axis = rho), \
                dagger(np.roll(U[rho], -1, axis = mu))) \
            - 2 * (u0**2) * U[mu] \
            + np.einsum('...ab,...bc,...cd->...ad', dagger(np.roll(U[rho], 1, axis = rho)), np.roll(U[mu], 1, axis = rho), \
                dagger(np.roll(np.roll(U[rho], 1, rho), -1, mu)))
        ) / ((u0*a)**2)
    return DelU

def get_wilson_obs(wilson_loop = wilson_loop_a_by_a):
    def obs(U):
        """
        Computes the Wilson loop of a gauge field configuration, averaged over 
        all mu < nu directions and averaged over all sites. wilson_loop should be 
        a function that computes the requisite wilson loop at each site on the lattice.
        """
        loops = np.zeros((6, *U.shape[1:5]), dtype = np.complex64)
        dir_idx = 0
        for mu in range(d):
            for nu in range(mu + 1, d):
                loops[dir_idx] += wilson_loop(U, mu, nu)
                dir_idx += 1
        return np.mean(loops), loops
    return obs

def id_field(LL):
    """
    Returns the identity color gauge field with the given parameters.
    """
    Uid = np.zeros((d, *LL, Nc, Nc), dtype = np.complex64)
    Uid[..., :, :] = np.eye(Nc, dtype = np.complex64)
    return Uid

def vec_dot(u, v):
    return np.dot(u, np.conjugate(v))

def vec_norm(u):
    return np.sqrt(np.abs(vec_dot(u, u)))

def proj_vec(u, v):
    """Projects v onto linear subspace spanned by u."""
    return vec_dot(v, u) / vec_dot(u, u) * u

def proj_SU3(M):
    """
    Projects a matrix M to the group SU(3) by orthonormalizing the first two columns, then 
    taking a cross product.
    """
    N = M.copy()
    [v1, v2, v3] = M.T
    u1 = v1 / vec_norm(v1)              # normalize
    u2 = v2 - proj_vec(u1, v2)
    u2 = u2 / vec_norm(u2)
    u3 = np.cross(u1.conj(), u2.conj())
    return np.array([u1, u2, u3], dtype = np.complex64).T

def rand_su3_matrix(eps):
    """
    Generates a random SU(3) matrix for the metropolis update with parameter eps. 
    Follows Peter Lepage's notes.

    Parameters
    ----------
    eps : float
        Metropolis parameter for update candidate.
    
    Returns
    -------
    np.array [Nc, Nc]
        Metropolis update candidate.
    """
    mat_elems = np.random.uniform(low = -1, high = 1, size = 6)
    H = np.array([
        [mat_elems[0], mat_elems[1], mat_elems[2]], 
        [mat_elems[1], mat_elems[3], mat_elems[4]], 
        [mat_elems[2], mat_elems[4], mat_elems[5]]
    ], dtype = np.complex64)
    return proj_SU3(np.eye(3) + 1j*eps*H)

def get_update_mats(n_mats, eps):
    """Generates 2*n_mats random SU(3) matrices near 1 for use in Metropolis."""
    update_mats = []
    for ii in range(n_mats):
        U = rand_su3_matrix(eps)
        update_mats.append(U)
        update_mats.append(dagger(U))
    return update_mats

def get_dsite_action(beta = 5.5):
    def dsite_action(n, mu, U, U_old, A):
        """
        Computes the part of the action that depends on the value 
        of U_\mu(n). 

        Parameters
        ----------
        n : np.array [4]
            4-position to evaluate U at.
        mu : int
            Direction to evaluate U_mu at.
        U : np.array, [4, Lx, Ly, Lz, Lt, Nc, Nc]
            Gauge field.
        U_old : np.array [Nc, Nc]
            Old color components of gauge field at (n, mu)
        A : np.array [Nc, Nc]
            Staple matrix for the mu direction at n.
        
        Returns
        -------
        float
            Value of S that depends on U_mu(n).
        """
        dU = U - U_old
        return -(beta/3) * np.real(np.einsum('ab,ba->', dU, A))
    return dsite_action

def get_dsite_action_improved(beta_twid = 1.719, u0 = 0.797):
    def dsite_action(n, mu, U, U_old, A):
        """
        Computes the part of the action that depends on the value 
        of U_\mu(n). 

        Parameters
        ----------
        n : np.array [4]
            4-position to evaluate U at.
        mu : int
            Direction to evaluate U_mu at.
        U : np.array, [4, Lx, Ly, Lz, Lt, Nc, Nc]
            Gauge field.
        U_old : np.array [Nc, Nc]
            Old color components of gauge field at (n, mu)
        A : np.array [Nc, Nc]
            (5/3) staple - 1/(12 u_0^2) improved_staple
        
        Returns
        -------
        float
            Value of S that depends on U_mu(n).
        """
        dU = U - U_old
        beta = beta_twid / (u0**4)
        return -(beta/3) * np.real(np.einsum('ab,ba->', dU, A))
    return dsite_action

def metropolis(LL, n_cfgs, delS = get_dsite_action(), eps = 0.24, improved = False, \
    n_corr = 50, tau = 5, n_hit = 10, n_mats = 100, save_cfgs = True, cfg_dir = unimproved_dir):
    """
    Implements the 1D Metropolis algorithm to sample the distribution 
    exp(-S[x]).

    Parameters
    ----------
    LL : [4]
        LL = [Lx, Ly, Lz, Lt] is the lattice geometry.
    n_cfgs : int
        Number of configurations to generate.
    n_corr : int (default = 50)
        Correlation length between different generated configurations. 
    tau : int (default = 10)
        Thermalization time, in units of n_corr.
    eps : float (default = 0.24)
        Average fluctuation size for Metropolis algorithm.
    improved : bool (default = False)
        Whether to use the improved or unimproved action.
    n_hit : int (default = 10)
        Number of times to update each U_mu(x) before moving on.
    n_mats : int (default = 100)
        Number of SU(3) matrices to generate at a given time.
    
    Returns
    -------
    np.array[n_cfgs, 4, LL[0], LL[1], LL[2], LL[3], Nc, Nc]
        n_cfgs generated configurations.
    """
    n_therm = tau * n_corr
    U = id_field(LL)
    U_lst = []
    if start_at is None:
        for ii in range(n_therm):                                   # thermalize
            U = update(U, LL, delS, eps, n_hit, n_mats, improved = improved)
        print('Thermalization complete.')
    else:
        print('Starting with configuration at: ' + start_at)
        f = h5py.File(start_at, 'r')
        U = f['U'][()]
        f.close()
    for cfg_idx in range(n_cfgs):
        for ii in range(n_corr):
            U = update(U, LL, delS, eps, n_hit, n_mats, improved = improved)
        U_lst.append(U)                                         # save cfg
        print('Configuration ' + str(cfg_idx + lbl_idx) + ' generated.')
        if save_cfgs:
            U_path = cfg_dir + 'cfg' + str(cfg_idx + lbl_idx) + '.h5'
            f = h5py.File(U_path, 'w')
            f['U'] = U
            f.close()
            print('Saved at: ' + U_path)
    return np.array(U_lst)

def update(V, LL, delS = get_dsite_action(), eps = 0.24, n_hit = 10, n_mats = 100, \
    observable = get_wilson_obs(wilson_loop_a_by_a), improved = False, u0 = 0.797):
    """Updates U_mu at every site on the lattice."""
    n_acc = 0
    start = time.time()
    update_mats = get_update_mats(n_mats, eps)
    U = np.copy(V)                  # only mutate local variable
    for mu in range(4):
        for x, y, z, t in itertools.product(*[range(LL[mu]) for mu in range(4)]):
            n = np.array([x, y, z, t])
            A = staple(U, n, mu)
            if improved:                        # get rectangle staples
                R = improved_staple(U, n, mu)
                A = (5/3)*A - R / (12 * (u0**2))
            cur_U = U[(mu, *n)]
            for hidx in range(n_hit):
                M = update_mats[np.random.randint(n_mats)]
                new_U = M @ cur_U
                dS = delS(n, mu, new_U, cur_U, A)
                if dS < 0 or np.exp(-dS) >= np.random.uniform():
                    n_acc += 1
                    cur_U = new_U
            U[(mu, *n)] = proj_SU3(cur_U)
    print('Elapsed time for single sweep: ' + str(time.time() - start))

    # assess how the update did
    n_updates = 4 * np.prod(LL) * n_hit                 # updates per sweep
    print('Accept ratio: ' + str(n_acc / n_updates))
    print(repr(U[0, 0, 0, 0, 0]))
    print('U^\dagger U for mu, n = (0, 0, 0, 0, 0): ' + str(U[0, 0, 0, 0, 0] @ dagger(U[0, 0, 0, 0, 0])))

    plaq = observable(U)[1]
    print('Average plaquette value: ' + str(np.mean(plaq)) + ' \pm ' + str(np.std(plaq, ddof = 1)))
    print('\n')
    
    return U

def compute_observables(cfgs, obs):
    """Computes observable obs on configurations cfgs."""
    n_cfgs = cfgs.shape[0]
    obs_lst = []
    for cfg_idx in range(n_cfgs):
        observable = obs(cfgs[cfg_idx])
        obs_lst.append(
            observable[1]
        )
    return np.array(obs_lst, dtype = np.complex64)

def load_cfgs(cfg_dir):
    """Loads configurations from directory dir."""
    fnames = []
    cfgs = []
    for (dirpath, dirnames, file) in os.walk(cfg_dir):
        fnames.extend(file)
    for idx, cfg in enumerate(fnames):
        fnames[idx] = cfg_dir + fnames[idx]
    for fname in fnames:
        print(fname)
        f = h5py.File(fname, 'r')
        cfgs.append(f['U'][()])
        f.close()
    return np.array(cfgs, dtype = np.complex64)

def run_metropolis(LL, n_cfgs, observable = get_wilson_obs(wilson_loop_a_by_a), gen_cfgs = False, \
    n_corr = 50, obs_label = r'$P_{\mu\nu}$', improved = False, gt_value = 0.5, fname = 'filename'):
    """Main loop for running metropolis algorithm. Takes parameters as input."""
    cfg_dir = improved_dir if improved else unimproved_dir
    if gen_cfgs:
        print('Generating configurations.')
        act = get_dsite_action_improved() if improved else get_dsite_action()
        cfgs = metropolis(LL, n_cfgs, delS = act, n_corr = n_corr, improved = improved, cfg_dir = cfg_dir)
    else:
        print('Loading configurations.')
        cfgs = load_cfgs(cfg_dir)
    
    corr = compute_observables(cfgs, observable)

    n_cfgs = corr.shape[0]
    corr_cfgs = np.mean(corr, axis = (1, 2, 3, 4, 5))
    corr_bar = np.mean(corr_cfgs)
    corr_var = (np.sum(corr_cfgs**2) / n_cfgs - (corr_bar**2)) / n_cfgs
    corr_std = np.sqrt(corr_var)
    corr_gvar = gv.gvar(corr_bar, corr_std)
    print('Value of observable: ' + str(corr_gvar))

    fig, ax = pt.add_subplot()
    ax.hist(np.real(corr.flatten()), color = 'r', alpha = 0.5)
    ax.set_xlabel(obs_label)
    ax.set_ylabel('Hits')
    pt.add_line(ax, gt_value, c = 'b')
    pt.save_figure(out_dir + '/' + fname + '.pdf')

    return corr_gvar, cfgs

def run_static_quark(LL, cfgs, t0 = 2, improved = False, fname = 'filename'):
    """
    Runs problem 4.4.1 and computes the static quark potential W(r, t). We will run all 
    values of (r1, r2, r3, t) with each component <= 4 = L/2 to avoid the boundary conditions, 
    and keep only the values of r which have norm <= 5. 
    """
    print('Running problem 4.4.1.')
    n_cfgs = cfgs.shape[0]
    r_eq = {}           # group equivalent [r1, r2, r3] values by their norm, rounded to the 3rd decimal place
    for r1, r2, r3, t in itertools.product(*[range(LL[mu] // 2) for mu in range(4)]):
        r = np.sqrt(r1**2 + r2**2 + r3**2)
        if r > 3.0:
            continue
        r_round = round(r, 3)
        if r_round in r_eq:
            r_eq[r_round].append([r1, r2, r3])
        else:
            r_eq[r_round] = [[r1, r2, r3]]
    r_vals = list(r_eq.keys())
    print(r_vals)

    print('Evaluating static quark potential.')
    V_vals = np.zeros((n_cfgs, len(r_vals)), dtype = np.float64)
    for cfg_idx, cfg in enumerate(cfgs):
        print('Running on cfg ' + str(cfg_idx))
        for ir, r in enumerate(r_eq):
            V_tmp = []
            for rvec in r_eq[r]:
                for t_axis in range(d):
                    ratio = W(cfg, rvec, t0, t_axis) / W(cfg, rvec, t0 + 1, t_axis)
                    if ratio < 0.0:
                        continue
                    V_tmp.append(np.log(ratio))
            V_vals[cfg_idx, ir] = np.mean(V_tmp)
        print(V_vals[cfg_idx])
    
    V_bar = np.mean(V_vals, axis = 0)
    V_var = (np.sum(V_vals**2, axis = 0) / n_cfgs - (V_bar**2)) / n_cfgs
    V_std = np.sqrt(V_var)
    V_gvar = gv.gvar(V_bar, V_std)
    print('Values of r: ' + str(r_vals))
    print('Values of V(r): ' + str(V_gvar))

    fig, ax = pt.plot_1d_data(r_vals, V_bar, V_std)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$aV(r)$')
    pt.save_figure(out_dir + '/' + fname + '.pdf')

#####################################################################
############################ UNIMPROVED #############################
#####################################################################

# Problem 4.3.1
_, cfgs = run_metropolis(LL, n_cfgs, fname = '4p3p1_unimproved_plaq')
run_metropolis(LL, n_cfgs, observable = get_wilson_obs(wilson_loop_a_by_2a), \
    obs_label = r'$R_{\mu\nu}$', fname = '4p3p1_unimproved_rect', gt_value = 0.26)

# Problem 4.4.1
run_static_quark(LL, cfgs, fname = '4p4p1_unimproved')

n_smear = 4
smeared = np.array([smear(cfgs[ii], n_smear) for ii in range(n_cfgs)], dtype = np.complex64)
print('Configurations smeared.')
run_static_quark(LL, smeared, fname = '4p4p1_unimproved_smeared')

#####################################################################
############################# IMPROVED ##############################
#####################################################################

# Problem 4.3.1
_, cfgs = run_metropolis(LL, n_cfgs, fname = '4p3p1_improved_plaq', improved = True, gt_value = 0.54)
run_metropolis(LL, n_cfgs, observable = get_wilson_obs(wilson_loop_a_by_2a), fname = '4p3p1_improved_rect', \
    improved = True, gt_value = 0.28)

# Problem 4.3.1
run_static_quark(LL, cfgs, fname = '4p4p1_improved')

smeared = np.array([smear(cfgs[ii], n_smear) for ii in range(n_cfgs)], dtype = np.complex64)
print('Configurations smeared.')
run_static_quark(LL, smeared, fname = '4p4p1_improved_smeared')
