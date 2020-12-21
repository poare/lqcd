# Add parent directory to path so that we can import from it
import sys
sys.path.append('/Users/theoares/lqcd/0nubb/python_scripts')

from analysis import *

# read in npr_momfrac format. Used for testing RI/sMOM RCs on RI'-MOM data.
def readfiles_momfrac(cfgs, q):
    props_k1 = np.zeros((len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    props_k2 = np.zeros((len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    props_q = np.zeros((len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    GV = np.zeros((d, len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    GA = np.zeros((d, len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    GO = np.zeros((16, len(cfgs), Nc, Nd, Nc, Nd, Nc, Nd, Nc, Nd), dtype = np.complex64)

    for idx, file in enumerate(cfgs):
        f = h5py.File(file, 'r')
        qstr = klist_to_string(q, 'p')
        props_q[idx] = np.einsum('ijab->aibj', f['prop/' + qstr][()]) / vol
        for mu in range(d):
            GV[mu, idx] = np.einsum('ijab->aibj', f['GV' + str(mu + 1) + '/' + qstr][()]) / vol
            GA[mu, idx] = np.einsum('ijab->aibj', f['GA' + str(mu + 1) + '/' + qstr][()]) / vol
    return props_q, GV, GA

# Reads in propagator output from free_field_props.qlua
def read_propagators(cfgs, q):
    props_k1 = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    props_k2 = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    props_q = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GV = np.zeros((4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GA = np.zeros((4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GO = np.zeros((16, len(cfgs), 3, 4, 3, 4, 3, 4, 3, 4), dtype = np.complex64)

    for idx, file in enumerate(cfgs):
        f = h5py.File(file, 'r')
        qstr = klist_to_string(q, 'q')
        if idx == 0:            # just choose a specific config to get these on, since they should be the same
            k1 = f['moms/' + qstr + '/k1'][()]
            k2 = f['moms/' + qstr + '/k2'][()]
        props_k1[idx] = np.einsum('ijab->aibj', f['prop_k1/' + qstr][()])
        props_k2[idx] = np.einsum('ijab->aibj', f['antiprop_k2/' + qstr][()])
        props_q[idx] = np.einsum('ijab->aibj', f['prop_q/' + qstr][()])
    return k1, k2, props_k1, props_k2, props_q
