import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
# job_num = 23274
# job_num = 23278
job_num = 23283    # very close (up to numerical error probably) for living in the span of \Lambda^1 and \Lambda^2.
# Note that what we get for Pi is still the same for 23283 as it is for 23278, the approximation is just actually equaling
# what we thought it would. 
k_list = []
for i in range(1, 4):
    for j in range(1, 4):
        for k in range(1, 4):
            for l in range(1, 4):
                k_list.append([i, j, k, l])
print('Number of total sink momenta: ' + str(len(k_list)))
#############################################################################
data_dir = './output/' + cfgbase + '_' + str(job_num)
mom_str_list = [analysis.plist_to_string(p) for p in k_list]
analysis.mom_list = k_list
analysis.mom_str_list = mom_str_list

file = h5py.File(data_dir + '/free_field.h5', 'r')
S = {}
G = [{}, {}, {}, {}]
for k in k_list:
    print(k)
    kstr = analysis.plist_to_string(k)
    S[kstr] = np.zeros((1, 3, 4, 3, 4), dtype = np.complex64)    # one boot
    S[kstr][0] = np.einsum('ijab->aibj', file['prop/' + kstr][()])
    props_inv = analysis.invert_prop(S, B = 1)
    Gamma = np.zeros((4, 4, 4, 4), dtype = np.complex64)
    for mu in range(4):
        G[mu][kstr] = np.zeros((1, 3, 4, 3, 4), dtype = np.complex64)
        G[mu][kstr][0] = np.einsum('ijab->aibj', file['O' + str(mu + 1) + str(mu + 1) + '/' + kstr][()])
        tmp = analysis.amputate(props_inv, G[mu], B = 1)[kstr][0]
        Gamma[mu, mu] = np.einsum('aiaj->ij', tmp) / 3    # color average
    p_lat = analysis.to_lattice_momentum(k)
    x = Gamma
    L1, L2 = analysis.Lambda1(p_lat), analysis.Lambda2(p_lat)
    v1 = np.array([
        analysis.inner(L1, x),
        analysis.inner(L2, x)
    ])
    Pi_i = np.dot(analysis.A_inv_ab(p_lat), v1)
    print('Decomposition:')
    print(Pi_i)

    print('Max difference between x and decomposition (want this to go to 0):')
    δx = x - (Pi_i[0] * L1 + Pi_i[1] * L2)
    δ_max = max([np.max(np.abs(δx[mu, mu])) for mu in range(4)])
    print(δ_max)
