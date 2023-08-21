import numpy as np
import h5py
import itertools
import gvar as gv

n_ops = 5
fname = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/results/Z_pert_run_downstream.h5'
f = h5py.File(fname, 'r')
Znpr = f['MSbar'][()]              # shape (nb, 5, 5)

# convert to BSM basis
k = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 2, 0], [2, 0, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, -1, 1]], \
             dtype = np.float64) / 4.0
kinv = np.linalg.inv(k)
Zbsm = np.einsum('ij,...jk,kl->...il', k, Znpr, kinv)

Z = np.zeros((n_ops, n_ops), dtype = object)
for i, j in itertools.product(range(n_ops), repeat = 2):
    Z[i, j] = gv.gvar(
        np.mean(Zbsm[:, i, j]), np.std(Zbsm[:, i, j], ddof = 1)
    )

arrstr = r'\begin{pmatrix} '
for i in range(n_ops):
    for j in range(n_ops):
        arrstr += str(Z[i, j])
        if j < n_ops - 1:
            arrstr += r' & '
    if i < n_ops - 1:
        arrstr += r' \\ '
arrstr += r' \end{pmatrix}'
print(arrstr)