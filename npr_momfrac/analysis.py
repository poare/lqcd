import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import h5py
import os

# STANDARD BOOTSTRAPPED PROPAGATOR ARRAY FORM: [b, cfg, c, s, c, s] where:
  # b is the boostrap index
  # cfg is the configuration index
  # c is a color index
  # s is a spinor index

gamma = np.zeros((4,4,4),dtype=complex)
gamma[0] = gamma[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gamma[1] = gamma[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gamma[2] = gamma[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gamma[3] = gamma[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
bvec = [0, 0, 0, .5]

mom_list =[[2,2,2,2],[2,2,2,4],[2,2,2,6],[3,3,3,2],[3,3,3,4],[3,3,3,6],[3,3,3,8],[4,4,4,4],[4,4,4,6],[4,4,4,8]]
L = 16
T = 48
LL = [L, L, L, T]
hypervolume = (L ** 3) * T

n_boot = 200
num_cfgs = 27

def get_mom_list():
    return mom_list

def get_num_cfgs():
    return num_cfgs

def pstring_to_list(pstring):
    return [int(pstring[1]), int(pstring[2]), int(pstring[3]), int(pstring[4])]

def plist_to_string(p):
    return 'p' + str(p[0]) + str(p[1]) + str(p[2]) + str(p[3])

def norm(p):
    if type(p) is str:
        p = pstring_to_list(p)
    return np.sqrt(np.sum([p[mu] ** 2 for mu in range(4)]))

mom_str_list = [plist_to_string(p) for p in mom_list]

# directory should contain hdf5 files. Will return props and threepts in form
# of a momentum dictionary with arrays of the form [cfg, c, s, c, s] where s
# is a Dirac index and c is a color index.
def readfile(directory):
    files = []
    for (dirpath, dirnames, file) in os.walk(directory):
        files.extend(file)
    props = {}
    threepts = {}
    global num_cfgs
    num_cfgs = len(files)
    for i, p in enumerate(mom_str_list):
        props[p] = np.zeros((num_cfgs, 3, 4, 3, 4), dtype = np.complex64)
        threepts[p] = np.zeros((num_cfgs, 3, 4, 3, 4), dtype = np.complex64)
    idx = 0
    for file in files:
        path_to_file = directory + '/' + file
        f = h5py.File(path_to_file, 'r')
        for pstring in mom_str_list:
            prop_path = 'prop/' + pstring
            threept_path = 'threept/' + pstring

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
def bootstrap(D):
    samples = {}
    for p in mom_str_list:
        S = D[p]
        num_configs = S.shape[0]
        samples[p] = np.zeros((n_boot, num_configs, 3, 4, 3, 4), dtype = np.complex64)
        for boot_id in range(n_boot):
            cfg_ids = np.random.choice(num_configs, num_configs, replace = True)    #Configuration ids to pick
            for i, cfgidx in enumerate(cfg_ids):
                samples[p][boot_id, i, :, :, :, :] = S[cfgidx, :, :, :, :]
    return samples

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
                # TODO check this is a proper contraction
                Gamma[p][b, cfgidx] = np.einsum('aibj,bjck,ckdl->aidl', Sinv, G, Sinv)
    return Gamma


# Compute quark field renormalization
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
                Zq[pstring][b, cfgidx] = hypervolume * (1j) * num / denom
    return Zq

# Compute \Gamma_{Born}(p). Should be a function of p with Dirac indices. For the mom frac
# paper, the Born term is i(\gamma_\mu p_\nu + \gamma_nu p_\mu) (equation B3). Returns Born
# term with index structure (mu, i, nu, j)
def born_term():
    Gamma_B = {}
    Gamma_B_inv = {}
    for p in mom_list:
        pstring = plist_to_string(p)
        # Gamma_B[pstring] is a (4, 4, 4, 4) matrix. The first two indices are mu and nu, the last 2 are Dirac indices.
        # Gamma_B[pstring] = np.zeros((4, 4, 4, 4), dtype = np.complex64)
        # Gamma_B_inv[pstring] = np.zeros((4, 4, 4, 4), dtype = np.complex64)
        # for mu in range(4):
        #     for nu in range(4):
        #         for i in range(4):
        #             for j in range(4):
        #                 Gamma_B[pstring][mu, i, nu, j] = (1j) * (gamma[mu][i, j] * p[nu] + gamma[nu][i, j] * p[mu])
        #                 # Gamma_B_inv[pstring] = np.linalg.tensorinv(Gamma_B[pstring])
        #                 # TODO Gamma_B is singular (?) if you do an entire inversion. Probably only invert
        #                 # the Dirac structure.
        #         Gamma_B_inv[pstring][mu, :, nu, :] = np.linalg.inv(Gamma_B[pstring][mu, :, nu, :])
        # TDOO the Born term shouldn't have Lorentz indices, should have the same structure as the
        # operator $\mathcal O$. Take the 3, 3 components and subtract off the 4, 4 components to get a
        # Dirac matrix which is a singlet in Lorentz space.
        Gamma_B[pstring] = np.zeros((4, 4), dtype = np.complex64)
        Gamma_B_inv[pstring] = np.zeros((4, 4), dtype = np.complex64)
        for i in range(4):
            for j in range(4):
                # TODO: is mu = 4 the same as mu = 0?
                Gamma_B[pstring][i, j] = (1j) * (gamma[3][i, j] * p[3] - gamma[0][i, j] * p[0]) / np.sqrt(2)
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
                Z[p][b, cfgidx] = 12 * np.dot(Zq[p][b, cfgidx], trace)
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
