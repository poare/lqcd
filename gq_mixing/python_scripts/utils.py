# import all shared utilities
import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *

# n_boot = 50
# set_boots(n_boot)

def readfiles(cfgs, k):
    props = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqg = np.zeros((4, 4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqq3 = np.zeros((3, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqq6 = np.zeros((6, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    for idx, file in enumerate(cfgs):
        f = h5py.File(file, 'r')
        kstr = klist_to_string(k, 'p')
        props[idx] = np.einsum('ijab->aibj', f['prop/' + kstr][()])
        for mu, nu in itertools.product(range(4), repeat = 2):
            Gqg[mu, nu, idx] = np.einsum('ijab->aibj', f['Gqg' + str(mu + 1) + str(nu + 1) + '/' + kstr][()])
        for mu in range(4):
            GV[mu, idx] = np.einsum('ijab->aibj', f['GV' + str(mu + 1) + '/' + kstr][()])
            GA[mu, idx] = np.einsum('ijab->aibj', f['GA' + str(mu + 1) + '/' + kstr][()])
        for a in range(3):
            Gqq3[a, idx] = np.einsum('ijab->aibj', f['Gqq/3' + str(a + 1) + '/' + kstr][()])
        for a in range(6):
            Gqq6[a, idx] = np.einsum('ijab->aibj', f['Gqq/6' + str(a + 1) + '/' + kstr][()])
    return props, Gqg, Gqq3, Gqq6
