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

mom_list =[[2,2,2,2],[2,2,2,4],[2,2,2,6],[3,3,3,2],[3,3,3,4],[3,3,3,6],[3,3,3,8],[4,4,4,4],[4,4,4,6],[4,4,4,8]]
n_boot = 200

def get_mom_list():
    return mom_list

def pstring_to_list(pstring):
    return [int(pstring[1]), int(pstring[2]), int(pstring[3]), int(pstring[4])]

def plist_to_string(p):
    return 'p' + str(p[0]) + str(p[1]) + str(p[2]) + str(p[3])

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
    numcfgs = len(files)
    for i, p in enumerate(mom_str_list):
        props[p] = np.zeros((numcfgs, 3, 4, 3, 4), dtype = np.complex64)
        threepts[p] = np.zeros((numcfgs, 3, 4, 3, 4), dtype = np.complex64)
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

# Need to implement all of these
# invert prop to amputate vertex function legs
def amputate(props, threepts):
    Gamma = {}
    num_cfgs = props[mom_str_list[0]].shape[1]
    for p in mom_str_list:
        Gamma[p] = np.zeros(props[p].shape, dtype = np.complex64)
        for b in range(n_boot):
            for cfgidx in range(num_cfgs):
                Sinv = np.linalg.tensorinv(props[p][b, cfgidx])
                G = threepts[p][b, cfgidx]
                Gamma[p][b, cfgidx] = Sinv * G * Sinv
    return Gamma

# Amputate legs to get vertex function \Gamma(p)


# Compute quark field renormalization


# Compute \Gamma_{Born}(p)


# Compute operator renormalization Z(p)
