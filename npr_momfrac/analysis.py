import numpy as np
from scipy.optimize import root
import h5py
import os
from scipy.special import zeta
import time
import re

# STANDARD BOOTSTRAPPED PROPAGATOR ARRAY FORM: [b, cfg, c, s, c, s] where:
  # b is the boostrap index
  # cfg is the configuration index
  # c is a color index
  # s is a spinor index

# Functions which work and match what they're suppose to do (i.e. Phiala's code): readfile,
# bootstrap, invert_prop, quark_renorm.
# See if I can get my code to work with the weird staple objects next to test the rest of the
# analysis.

# g = np.diag([1, -1, -1, -1])
g = np.diag([1, 1, 1, 1])    # in Euclidean space?

gamma = np.zeros((4,4,4),dtype=complex)
gamma[0] = gamma[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gamma[1] = gamma[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gamma[2] = gamma[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gamma[3] = gamma[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
bvec = [0, 0, 0, .5]

mom_list =[[2,2,2,2],[2,2,2,4],[2,2,2,6],[3,3,3,2],[3,3,3,4],[3,3,3,6],[3,3,3,8],[4,4,4,4],[4,4,4,6],[4,4,4,8]]

# propagator mom_list for 16583 test
prop_mom_list = [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]

# mom_list for 16142
# mom_list = []
# for i in range(1, 8 + 1):
#     for j in range(2, 10 + 1, 2):
#         if j + 1 >= i and not (i == 8 and j == 10):
#             mom_list.append([i, i, i, j])

#mom_list for 16165
# mom_list = []
# for i in range(1, 16 + 1):
#     for j in range(i, 16 + 1):
#         if j % 2 == 0:
#             mom_list.append([i, i, i, j])

L = 16
T = 48
LL = [L, L, L, T]
hypervolume = (L ** 3) * T
# LL = [24, 24, 24, 48]    # only for debugging purposes, since Phiala's code is on a 24^3 x 48 lattice.
# hypervolume = (24 ** 3) * T

n_boot = 200
num_cfgs = 1

def get_mom_list():
    return mom_list

def get_prop_mom_list():
    return prop_mom_list

def get_num_cfgs():
    return num_cfgs

def get_hypervolume():
    return hypervolume

def get_metric():
    return g

def pstring_to_list(pstring):
    return [int(pstring[1]), int(pstring[2]), int(pstring[3]), int(pstring[4])]

def plist_to_string(p):
    return 'p' + str(p[0]) + str(p[1]) + str(p[2]) + str(p[3])

# squares a 4 vector.
def square(p):
    if type(p) is str:
        p = pstring_to_list(p)
    p = np.array([p])
    return np.dot(p, np.dot(g, p.T))[0, 0]

def norm(p):
    return np.sqrt(np.abs(square(p)))

mom_str_list = [plist_to_string(p) for p in mom_list]

def set_mom_list(plist):
    global mom_list
    global mom_str_list
    mom_list = plist
    mom_str_list = [plist_to_string(x) for x in plist]
    return True

# directory should contain hdf5 files. Will return props and threepts in form
# of a momentum dictionary with arrays of the form [cfg, c, s, c, s] where s
# is a Dirac index and c is a color index. If sink_momenta is passed in (a
# list of momenta) then only read those momenta in.
def readfile(directory, gauged = False, dpath = '', sink_momenta = None):
    files = []
    for (dirpath, dirnames, file) in os.walk(directory):
        files.extend(file)
    props = {}
    threepts = {}
    global num_cfgs
    num_cfgs = len(files)
    if sink_momenta:
        str_list = sink_momenta
    else:
        str_list = mom_str_list
    for i, p in enumerate(str_list):
        props[p] = np.zeros((num_cfgs, 3, 4, 3, 4), dtype = np.complex64)
        threepts[p] = np.zeros((num_cfgs, 3, 4, 3, 4), dtype = np.complex64)
    idx = 0
    for file in files:
        path_to_file = directory + '/' + file
        f = h5py.File(path_to_file, 'r')
        for pstring in str_list:
            if gauged:
                prop_path = 'propprime/' + dpath + pstring
                threept_path = 'threeptprime/' + dpath + pstring
            else:
                prop_path = 'prop/' + dpath + pstring
                threept_path = 'threept/' + dpath + pstring
                # prop_path = 'ptprop/' + dpath + pstring
                # threept_path = 'threept_pt/' + dpath + pstring

            # delete this block once I push the new code
            config_id = str([x for x in f[prop_path].keys()][0])
            prop_path += '/' + config_id
            threept_path += '/' + config_id

            prop = f[prop_path][()]
            threept = f[threept_path][()]
            props[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', prop)
            threepts[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', threept)
        idx += 1
    return props, threepts

# Bootstraps a set of propagator labelled by momentum. Will return a momentum
# dictionary, and the value of each key will be [boot, cfg, c, s, c, s].
def bootstrap(D, seed = 0):
    samples = {}
    np.random.seed(seed)
    for p in mom_str_list:
        S = D[p]
        num_configs = S.shape[0]
        samples[p] = np.zeros((n_boot, num_configs, 3, 4, 3, 4), dtype = np.complex64)
        for boot_id in range(n_boot):
            cfg_ids = np.random.choice(num_configs, num_configs, replace = True)    #Configuration ids to pick
            for i, cfgidx in enumerate(cfg_ids):
                samples[p][boot_id, i, :, :, :, :] = S[cfgidx, :, :, :, :]
    return samples

# Invert propagator to get S^{-1}. This agrees with Phiala's code.
def invert_prop(props):
    Sinv = {}
    for p in mom_str_list:
        Sinv[p] = np.zeros(props[p].shape, dtype = np.complex64)
        for b in range(n_boot):
            for cfgidx in range(num_cfgs):
                Sinv[p][b, cfgidx, :, :, :, :] = np.linalg.tensorinv(props[p][b, cfgidx])
    return Sinv

# Amputate legs to get vertex function \Gamma(p)
def amputate(props_inv, threepts):
    Gamma = {}
    for p in mom_str_list:
        Gamma[p] = np.zeros(props_inv[p].shape, dtype = np.complex64)
        for b in range(n_boot):
            for cfgidx in range(num_cfgs):
                Sinv = props_inv[p][b, cfgidx]
                G = threepts[p][b, cfgidx]
                Gamma[p][b, cfgidx] = np.einsum('aibj,bjck,ckdl->aidl', Sinv, G, Sinv) * hypervolume
    return Gamma


# Compute quark field renormalization. This agrees with Phiala's code.
def quark_renorm(props_inv):
    Zq = {}
    for p in mom_list:
        pstring = plist_to_string(p)
        Zq[pstring] = np.zeros((n_boot, num_cfgs), dtype = np.complex64)
        phase = [np.sin(2 * np.pi * (p[mu] + bvec[mu]) / LL[mu]) for mu in range(4)]
        for b in range(n_boot):
            for cfgidx in range(num_cfgs):
                Sinv = props_inv[pstring][b, cfgidx]
                num = sum([phase[mu] * np.einsum('ij,ajai', gamma[mu], Sinv) for mu in range(4)])
                denom = 12 * sum([np.sin(2 * np.pi * (p[mu] + bvec[mu]) / LL[mu]) ** 2 for mu in range(4)])
                Zq[pstring][b, cfgidx] = (-1j) * (num / denom) * hypervolume
    return Zq

# Compute \Gamma_{Born}(p). Should be a function of p with Dirac indices. For the mom frac
# paper, the Born term is i(\gamma_\mu p_\nu + \gamma_nu p_\mu) (equation B3). Returns Born
# term with index structure (mu, i, nu, j)
def born_term():
    Gamma_B = {}
    Gamma_B_inv = {}
    for p in mom_list:
        pstring = plist_to_string(p)
        Gamma_B[pstring] = (1j) * np.sqrt(2) * (p[2] * gamma[2] - p[3] * gamma[3])
        Gamma_B_inv[pstring] = np.linalg.inv(Gamma_B[pstring])
    return Gamma_B, Gamma_B_inv


# Compute operator renormalization Z(p)
def get_Z(Zq, Gamma, Gamma_B_inv):
    Z = {}
    for p in mom_str_list:
        Z[p] = np.zeros((n_boot, num_cfgs), dtype = np.complex64)
        for b in range(n_boot):
            for cfgidx in range(num_cfgs):
                trace = np.einsum('aiaj,ji', Gamma[p][b, cfgidx], Gamma_B_inv[p])
                Z[p][b, cfgidx] = 12 * Zq[p][b, cfgidx] / trace
    return Z

# Do the statistics on Z[p][b, cfg] by computing statistics on each boostrapped
# sample Z[p][b] (num_cfgs data points), then using these statistics to compute the
# mean and error on Z_q(p) averaged across all the bootstrap samples.
def get_statistics_Z(Z):
    mu, mu_temp, sigma = {}, {}, {}
    for pstring in mom_str_list:
        mu_temp[pstring] = np.zeros((n_boot), dtype = np.complex64)
        for b in range(n_boot):
            for cfgidx in range(num_cfgs):
                mu_temp[pstring][b] = np.mean(Z[pstring][b])    # average over configurations in boot sample
        mu[pstring] = np.mean(mu_temp[pstring])
        sigma[pstring] = np.std(mu_temp[pstring])
    return mu, sigma

# pass in Z before we do statistics
def to_MSbar(Z):
    Zms = {}
    nf = 3        # 3 flavors of quark
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    c11 = - 124 / 27
    c21 = - 68993 / 729 + (160 / 9) * z3 + (2101 / 243) * nf
    c31 = - 451293899 / 157464 + (1105768 / 2187) * z3 - (8959 / 324) * z4 - (4955 / 81) * z5 \
        + (8636998 / 19683 - (224 / 81) * z3 + (640 / 27) * z4) * nf - (63602 / 6561 + (256 / 243) * z3) * (nf ** 2)
    c12 = - 8 / 9
    c22 = - 2224 / 27 - (40 / 9) * z3 + (40 / 9) * nf
    c32 = - 136281133 / 26244 + (376841 / 243) * z3 - (43700 / 81) * z5 + (15184 / 27 - (1232 / 81) * z3) * nf \
        - (9680 / 729) * (nf ** 2)
    b2 = - 359 / 9 + 12 * z3 + (7 / 3) * nf
    b3 = - 439543 / 162 + (8009 / 6) * z3 + (79 / 4) * z4 - (1165 / 3) * z5 + (24722 / 81 - (440 / 9) * z3) * nf \
        - (1570 / 243) * (nf ** 2)
    g = 1.964    # g_{MS bar}(mu = 2 GeV)
    for p in mom_list:
        pstring = plist_to_string(p)
        # Adjusted R in table X to fit the operator I'm using.
        R = ((p[2] ** 2 - p[3] ** 2) ** 2) / (2 * square(p) * (p[2] ** 2 + p[3] ** 2))
        c1 = c11 + c12 * R
        c2 = c21 + b2 + b2 + c22 * R
        c3 = c31 + b2 * c11 + b3 + (c32 + b2 * c12) * R
        x = (g ** 2) / (16 * (np.pi ** 2))
        Zconv = 1 + c1 * x + c2 * (x ** 2) + c3 * (x ** 3)
        Zms[pstring] = Zconv * Z[pstring]
    return Zms

# Determines how the error at base_time scales as we increase the number of
# samples used in the computation. Z is the set of wavefunction renormalizations.
# n_start and n_step are the configuration numbers to start and end at, and
# n_step is the number of steps to take between different configuration numberes.
# To see pictorally, plot returned cfg_list versus err
def error_analysis(Z, n_start, n_step):
    mom = mom_str_list[0]
    num_configs = Z[mom].shape[1]
    cfg_list = range(n_start, num_configs, n_step)
    err = np.zeros(len(cfg_list))
    means = np.zeros(len(cfg_list))
    for i, n in enumerate(cfg_list):    # sample n configurations from C
        config_ids = np.random.choice(num_configs, n, replace = False)
        # Z_sub = Z[:, config_ids]    #now get error on the subsampled C
        # subensemble = bootstrap(C_sub)
        subensemble = Z[mom][:, config_ids]
        # n_boot x n matrix. Average over n_boot
        subensemble_avg = np.mean(subensemble, axis = 1)
        μ = np.abs(np.mean(subensemble_avg, axis = 0))
        σ = np.abs(np.std(subensemble_avg, axis = 0))
        err[i] = σ
        means[i] = μ
    return cfg_list, err, means

def save_mu_sigma(mu, sigma, directory, clear_path = False):
    mu_file = directory + '/mu.npy'
    sigma_file = directory + '/sigma.npy'
    if clear_path:
        os.remove(mu_file)
        os.remove(sigma_file)
    np.save(mu_file, mu)
    np.save(sigma_file, sigma)
    return True

# Returns the data which was saved after running "python3 perform_analysis.py"
def load_data(directory):
    mu = np.load(directory + '/mu.npy')
    sigma = np.load(directory + '/sigma.npy')
    p_list = np.load(directory + '/mom_list.npy')
    try:
        prop_p_list = np.load(directory + '/prop_mom_list.npy')
    except OSError:
        prop_p_list = ['point source']
    cfgnum = np.load(directory + '/cfgnum.npy')
    return mu, sigma, p_list, prop_p_list, cfgnum

# momenta is subset of mom_list
def subsample(mu, sigma, momenta):
    mu_sub, sigma_sub = [], []
    for mu_i, sigma_i in zip(mu, sigma):
        cur_mu = {}
        cur_sigma = {}
        for p in momenta:
            pstring = plist_to_string(p)
            cur_mu[pstring] = mu_i[pstring]
            cur_sigma[pstring] = sigma_i[pstring]
        mu_sub.append(cur_mu)
        sigma_sub.append(cur_sigma)
    return mu_sub, sigma_sub

# returns a subset of momenta with radius r away from the diagonal [1, 1, 1, 1]
def cylinder(plist, r):
    nhat = np.array([1, 1, 1, 1]) / 2
    psub = []
    for x in plist:
        proj = np.dot(nhat, x) * nhat
        dist = np.linalg.norm(x - proj)
        if dist <= r:
            psub.append(x)
    return np.array(psub)

def run_analysis(directory, s = 0):
    Γ_B, Γ_B_inv = born_term()
    props, threepts = readfile(directory)
    props_boot = bootstrap(props, seed = s)
    threept_boot = bootstrap(threepts, seed = s)
    props_inv = invert_prop(props_boot)
    Γ = amputate(props_inv, threept_boot)
    Zq = quark_renorm(props_inv)
    Z = get_Z(Zq, Γ, Γ_B_inv)
    mu, sigma = get_statistics_Z(Z)
    return mu, sigma

def test_analysis_propagators(directory, s = 0):
    mu, sigma = [], []
    Γ_B, Γ_B_inv = born_term()
    for idx in range(len(prop_mom_list)):
        print('Computing for propagator momentum ' + str(prop_mom_list[idx]))
        mom_prop_path = 'prop' + str(idx + 1) + '/'
        props, threepts = readfile(directory, dpath = mom_prop_path)
        print('Bootstrapping.')
        props_boot = bootstrap(props, seed = s)
        threept_boot = bootstrap(threepts, seed = s)
        print('Inverting propagators.')
        props_inv = invert_prop(props_boot)
        print('Amputating legs.')
        Γ = amputate(props_inv, threept_boot)
        print('Computing quark field renormalization.')
        Zq = quark_renorm(props_inv)
        print('Computing operator renormalization.')
        Z = get_Z(Zq, Γ, Γ_B_inv)
        mu_p, sigma_p = get_statistics_Z(Z)
        mu.append(mu_p)
        sigma.append(sigma_p)
    return mu, sigma

# load in a single momentum at a time so that it doesn't overload python (for large data configs)
# prop_mom_list is the set of momenta wall sources the propagators are computed at.
def run_analysis_single_momenta(directory, s = 0):
    start = time.time()
    mu, sigma = [{}] * len(prop_mom_list), [{}] * len(prop_mom_list)
    # mu, sigma = [0] * len(prop_mom_list), [0] * len(prop_mom_list)
    Γ_B, Γ_B_inv = born_term()
    global mom_list
    global mom_str_list
    mom_str_list_cp = mom_str_list
    for idx in range(len(prop_mom_list)):
        print('Computing for propagator momentum ' + str(prop_mom_list[idx]))
        mom_prop_path = 'prop' + str(idx + 1) + '/'
        for p in mom_str_list_cp:
            print('Computing for sink momentum ' + p)
            mom_list = [pstring_to_list(p)]
            mom_str_list = [p]
            # props, threepts = readfile(directory, dpath = mom_prop_path, sink_momenta = [p])
            props, threepts = readfile(directory, dpath = mom_prop_path)
            print('Bootstrapping.')
            props_boot = bootstrap(props, seed = s)
            threept_boot = bootstrap(threepts, seed = s)
            print('Inverting propagators.')
            props_inv = invert_prop(props_boot)
            print('Amputating legs.')
            Γ = amputate(props_inv, threept_boot)
            print('Computing quark field renormalization.')
            Zq = quark_renorm(props_inv)
            print('Computing operator renormalization.')
            Z = get_Z(Zq, Γ, Γ_B_inv)
            # mu[idx], sigma[idx] = get_statistics_Z(Z)
            mu_p, sigma_p = get_statistics_Z(Z)
            mu[idx][p] = mu_p[p]
            sigma[idx][p] = sigma_p[p]

            # Time per iteration
            print('Elapsed time: ' + str(time.time() - start))
    return mu, sigma

def get_point_list(directory):
    file0 = [file for (dirpath, dirnames, file) in os.walk(directory)][0][0]
    f = h5py.File(directory + '/' + file0, 'r')
    allpts = f['prop']
    pts = []
    for ptstr in allpts.keys():    # should be of the form x#y#z#t#
        parts = re.split('x|y|z|t', ptstr)[1:]
        pt = [int(x) for x in parts]
        pts.append(pt)
    f.close()
    return pts

def average_over_points(S_list, G_list, pt_list):
    N = len(pt_list)
    S_ave, G_ave = {}, {}
    # global mom_str_list
    for pstring in mom_str_list:
        S_ave[pstring], G_ave[pstring] = np.zeros(S_list[0][pstring].shape, dtype = np.complex64), \
                    np.zeros(G_list[0][pstring].shape, dtype = np.complex64)
        for idx, pt in enumerate(pt_list):
            S_ave[pstring] += S_list[idx][pstring]
            G_ave[pstring] += G_list[idx][pstring]
        S_ave[pstring] *= hypervolume / N
        G_ave[pstring] *= hypervolume / N
    return S_ave, G_ave

# Run analysis on a set of point sources by averaging over the S and G's.
def run_analysis_point_sources(directory, s = 0):
    start = time.time()
    # determine points which are run
    pt_list = get_point_list(directory)
    mu, sigma = {}, {}
    Γ_B, Γ_B_inv = born_term()
    global mom_list
    global mom_str_list
    mom_str_list_cp = mom_str_list

    for p in mom_str_list_cp:
        print('Computing for sink momentum ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]
        print('Averaging over propagators.')
        S_list, G_list = [], []
        for idx, pt in enumerate(pt_list):
            # print('Computing for propagator at point ' + str(pt))
            pt_prop_path = 'x' + str(pt[0]) + 'y' + str(pt[1]) + 'z' + str(pt[2]) + 't' + str(pt[3]) + '/'
            props, threepts = readfile(directory, dpath = pt_prop_path, sink_momenta = [p])
            S_list.append(props)
            G_list.append(threepts)
        S_ave, G_ave = average_over_points(S_list, G_list, pt_list)
        print('Bootstrapping.')
        props_boot = bootstrap(S_ave, seed = s)
        threept_boot = bootstrap(G_ave, seed = s)
        print('Inverting propagators.')
        props_inv = invert_prop(props_boot)
        print('Amputating legs.')
        Γ = amputate(props_inv, threept_boot)
        print('Computing quark field renormalization.')
        Zq = quark_renorm(props_inv)
        print('Computing operator renormalization.')
        Z = get_Z(Zq, Γ, Γ_B_inv)
        mu_p, sigma_p = get_statistics_Z(Z)    # mu_p is a dictionary with only one p value
        mu[p], sigma[p] = mu_p[p], sigma_p[p]

        # Time per iteration
        print('Elapsed time: ' + str(time.time() - start))
    return mu, sigma, pt_list
