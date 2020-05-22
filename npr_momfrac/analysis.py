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

g = np.diag([1, 1, 1, 1])

delta = np.identity(4, dtype = np.complex64)
gamma = np.zeros((4,4,4),dtype=complex)
gamma[0] = gamma[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gamma[1] = gamma[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gamma[2] = gamma[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gamma[3] = gamma[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
gamma5 = np.dot(np.dot(np.dot(gamma[0], gamma[1]), gamma[2]), gamma[3])
bvec = [0, 0, 0, .5]

#TODO subtract out p slash or not?
Lambda1 = lambda p : (-2j) * np.array([[(p[mu] * gamma[nu] + p[nu] * gamma[mu]) / 2 - delta[mu, nu] * (slash(p)) / 4 for mu in range(4)] for nu in range(4)])
Lambda2 = lambda p : (-2j) * np.array([[p[mu] * p[nu] * slash(p) / square(p)  - delta[mu, nu] * slash(p) / 4 for mu in range(4)] for nu in range(4)])
# Lambda1 = lambda p : (-1j) * np.array([[(p[mu] * gamma[nu] + p[nu] * gamma[mu]) / 2 for mu in range(4)] for nu in range(4)])
# Lambda2 = lambda p : (-1j) * np.array([[p[mu] * p[nu] * slash(p) / square(p) for mu in range(4)] for nu in range(4)])

# mom_list =[[2,2,2,2],[2,2,2,4],[2,2,2,6],[3,3,3,2],[3,3,3,4],[3,3,3,6],[3,3,3,8],[4,4,4,4],[4,4,4,6],[4,4,4,8]]

# propagator mom_list for 16583 test
prop_mom_list = [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]

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

mom_list = []
for i in range(0, 6):
    for j in range(0, 6):
        for k in range(0, 6):
            for l in range(0, 6):
                mom_list.append([i, j, k, l])
mom_list = cylinder(mom_list, 2)

L = 16
T = 48
LL = [L, L, L, T]
hypervolume = (L ** 3) * T

# n_boot = 200
# n_boot = 100
n_boot = 50
num_cfgs = 1

def pstring_to_list(pstring):
    def get_momenta(x):
        lst = []
        mult = 1
        for digit in x:
            if digit == '-':
                mult *= -1
            else:
                lst.append(mult * int(digit))
                mult = 1
        return lst
    return get_momenta(pstring.split('p')[1])

def plist_to_string(p):
    return 'p' + str(p[0]) + str(p[1]) + str(p[2]) + str(p[3])

def to_lattice_momentum(k):
    # TODO should I add bvec to k? It's done like that in quark_renorm
    # return [np.complex64(2 * np.sin(np.pi * k[mu] / LL[mu])) for mu in range(4)]
    return [np.complex64(2 * np.sin(np.pi * (k[mu] + bvec[mu]) / LL[mu])) for mu in range(4)]

# squares a 4 vector.
def square(p):
    if type(p) is str:
        p = pstring_to_list(p)
    p = np.array([p])
    return np.dot(p, np.dot(g, p.T))[0, 0]

def slash(p):
    return sum([p[mu] * gamma[mu] for mu in range(4)])

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
def readfile(directory, dpath = '', pointsrc = False, op = 'O', sink_momenta = None, mu = None, j = 0):
    files = []
    for (dirpath, dirnames, file) in os.walk(directory):
        files.extend(file)
    props = {}
    threepts = {}
    files.sort()
    # print(files)
    # files = ['quarkNPR_1290.h5', 'quarkNPR_400.h5', 'quarkNPR_310.h5']
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
        # print(file)
        path_to_file = directory + '/' + file
        f = h5py.File(path_to_file, 'r')
        if pointsrc:
            for pstring in str_list:
                pt_list = get_point_list(path_to_file)
                # S_list, G_list = [], []
                S_tot = np.zeros((4, 4, 3, 3), dtype = np.complex64)
                G_tot = np.zeros((4, 4, 3, 3), dtype = np.complex64)
                for pt in pt_list:
                    # print('Computing for propagator at point ' + str(pt))
                    pt_prop_path = 'x' + str(pt[0]) + 'y' + str(pt[1]) + 'z' + str(pt[2]) + 't' + str(pt[3]) + '/'
                    prop_path = 'prop/' + pt_prop_path + pstring
                    config_id = str([x for x in f[prop_path].keys()][0])
                    prop_path += '/' + config_id
                    S_tot  += f[prop_path][()]
                    if mu == -1:
                        G_tot += np.zeros((4, 4, 3, 3), dtype = np.complex64)
                    else:
                        threept_path = op + str(mu + 1) + str(mu + 1) + '/' + pt_prop_path + pstring + '/' + config_id
                        G_tot += f[threept_path][()]
                S_ave = (hypervolume / len(pt_list)) * S_tot
                G_ave = (hypervolume / len(pt_list)) * G_tot
                # S_ave, G_ave, sigma_S, sigma_G = average_over_points(S_list, G_list, pt_list)
                props[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', S_ave)
                threepts[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', G_ave)
        else:
            for pstring in str_list:
                # prop_path = 'prop/' + dpath + pstring
                prop_path = 'prop/' + pstring
                # print(prop_path)
                # prop_path = 'prop_sink/' + dpath + pstring
                # prop_path = 'prop_phiala/' + dpath + pstring
                # config_id = str([x for x in f[prop_path].keys()][0])
                # prop_path += '/' + config_id
                prop = f[prop_path][()]
                props[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', prop)

                if mu == -1:
                    threepts = np.zeros(prop.shape, dtype = np.complex64)
                elif mu != None:
                    # threept_path = 'O_operator' + str(mu + 1) + str(mu + 1) + '/' + dpath + pstring + '/' + config_id
                    # op defaults to 'O'
                    threept_path = op + str(mu + 1) + str(mu + 1) + '/' + dpath + pstring + '/' + config_id
                    threept = f[threept_path][()]
                    threepts[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', threept)
                else:
                    # threept_path = op + str(j + 1) + '/' + pstring
                    threept_path = op + str(j + 1) + '/' + pstring + dpath
                    # print(threept_path)
                    threept = f[threept_path][()]
                    threepts[pstring][idx, :, :, :, :] = np.einsum('ijab->aibj', threept)
        idx += 1
        f.close()
    return props, threepts, num_cfgs

# Bootstraps a set of propagator labelled by momentum. Will return a momentum
# dictionary, and the value of each key will be [boot, cfg, c, s, c, s].
def bootstrap(D, seed = 5):
    weights = np.ones((len(D[list(D.keys())[0]])))
    weights2=weights/float(np.sum(weights))
    samples = {}
    np.random.seed(5)
    for p in D.keys():
        S = D[p]
        num_configs = S.shape[0]
        samples[p] = np.zeros((n_boot, 3, 4, 3, 4), dtype = np.complex64)
        for boot_id in range(n_boot):
            cfg_ids = np.random.choice(num_configs, p = weights2, size = num_configs, replace = True)    #Configuration ids to pick
            cfg_ids = np.random.choice(num_configs, size = num_configs, replace = True)
            samples[p][boot_id] = np.mean(S[cfg_ids], axis = 0)
    return samples

# Invert propagator to get S^{-1}. This agrees with Phiala's code.
def invert_prop(props, B = n_boot):
    Sinv = {}
    for p in props.keys():
        Sinv[p] = np.zeros(props[p].shape, dtype = np.complex64)
        for b in range(B):
            Sinv[p][b, :, :, :, :] = np.linalg.tensorinv(props[p][b])
    return Sinv

# Amputate legs to get vertex function \Gamma(p)
def amputate(props_inv, threepts, B = n_boot):
    Gamma = {}
    for p in props_inv.keys():
        # p = plist_to_string(plist)
        Gamma[p] = np.zeros(props_inv[p].shape, dtype = np.complex64)
        for b in range(B):
            Sinv = props_inv[p][b]
            G = threepts[p][b]
            Gamma[p][b] = np.einsum('aibj,bjck,ckdl->aidl', Sinv, G, Sinv) * hypervolume
    return Gamma


# Compute quark field renormalization. This agrees with Phiala's code.
def quark_renorm(props_inv):
    Zq = {}
    # for p in mom_list:
    for pstring in props_inv.keys():
        p = pstring_to_list(pstring)
        Zq[pstring] = np.zeros((n_boot), dtype = np.complex64)
        phase = [np.sin(2 * np.pi * (p[mu] + bvec[mu]) / LL[mu]) for mu in range(4)]
        for b in range(n_boot):
            Sinv = props_inv[pstring][b]
            num = sum([phase[mu] * np.einsum('ij,ajai', gamma[mu], Sinv) for mu in range(4)])
            denom = 12 * sum([np.sin(2 * np.pi * (p[mu] + bvec[mu]) / LL[mu]) ** 2 for mu in range(4)])
            Zq[pstring][b] = (1j) * (num / denom) * hypervolume
    return Zq

# Compute \Gamma_{Born}(p). Should be a function of p with Dirac indices. For the mom frac
# paper, the Born term is i(\gamma_\mu p_\nu + \gamma_nu p_\mu) (equation B3). Returns Born
# term with index structure (mu, i, nu, j)
def born_term(mu, momenta = mom_list):
    Gamma_B = {}
    Gamma_B_inv = {}
    # for p in momenta:
    #     pstring = plist_to_string(p)
    #     Gamma_B[pstring] = (-1j) * 2 * (2 * np.pi * p[mu] / LL[mu]) * gamma[mu]
    #     Gamma_B_inv[pstring] = np.linalg.inv(Gamma_B[pstring])
    # gets much closer, but still artifacts. These are likely from mixing in the other tensor structure
    for k in momenta:
        p_lat = to_lattice_momentum(k)
        pstring = plist_to_string(k)
        Gamma_B[pstring] = (-1j) * 2 * p_lat[mu] * gamma[mu]
        # Gamma_B[pstring] = (-1j) * (2 * p_lat[mu] * gamma[mu] - (1/2) * slash(p_lat))
        # Gamma_B_inv[pstring] = np.linalg.inv(Gamma_B[pstring])
    return Gamma_B#, Gamma_B_inv

def born_term_numerical(mu, momenta = mom_list):
    Gamma_B = {}
    Gamma_B_inv = {}
    file = '/Users/theoares/lqcd/npr_momfrac/born_term.h5'
    f = h5py.File(file, 'r')
    props, threepts = {}, {}
    for p in momenta:
        pstring = plist_to_string(p)
        prop_path = 'prop/' + pstring + '/cfg200'
        threept_path = 'O' + str(mu + 1) + str(mu + 1) + '/' + pstring + '/cfg200'

        # reshape and add bootstrap + configuration dimensions
        x = np.einsum('ijab->aibj', f[prop_path][()])
        y = np.einsum('ijab->aibj', f[threept_path][()])
        props[pstring] = np.expand_dims(x, axis = 0)
        threepts[pstring] = np.expand_dims(y, axis = 0)
    f.close()
    props_inv = invert_prop(props, B = 1)
    Gamma = amputate(props_inv, threepts, B = 1)

    pkeys = [plist_to_string(p) for p in momenta]
    eps = 1e-6    # tolerance for which terms to set to 0
    for p in pkeys:
        Gamma[p][np.abs(Gamma[p]) < eps] = 0

    Gamma_B = {p : Gamma[p][0, 0, :, 0, :] for p in pkeys}    # strip off extra indices
    # Gamma_B_inv = {p : np.linalg.inv(Gamma_B[p]) for p in pkeys}
    return Gamma_B#, Gamma_B_inv

def born_term_irreps(momenta):
    Gamma_B = [{}, {}, {}]
    Gamma_B_inv = [{}, {}, {}]
    file = '/Users/theoares/lqcd/npr_momfrac/born_term.h5'
    f = h5py.File(file, 'r')
    props, threepts = {}, [{}, {}, {}]
    for p in momenta:
        pstring = plist_to_string(p)
        prop_path = 'prop/' + pstring + '/cfg200'
        x = np.einsum('ijab->aibj', f[prop_path][()])    # reshape and add bootstrap + configuration dimensions
        props[pstring] = np.expand_dims(x, axis = 0)

        threept_vec = [0] * 4
        for mu in range(4):
            threept_path = 'O' + str(mu + 1) + str(mu + 1) + '/' + pstring + '/cfg200'
            y = np.einsum('ijab->aibj', f[threept_path][()])
            threept_vec[mu] = np.expand_dims(y, axis = 0)
        threepts[0][pstring] = (threept_vec[2] - threept_vec[3]) / np.sqrt(2)
        threepts[1][pstring] = (threept_vec[0] - threept_vec[1]) / np.sqrt(2)
        threepts[2][pstring] = (threept_vec[0] + threept_vec[1] - threept_vec[2] - threept_vec[3]) / 2
    f.close()
    props_inv = invert_prop(props, B = 1)

    Gamma = []
    for i in range(3):
        Gamma.append(amputate(props_inv, threepts[i], B = 1))
    pkeys = [plist_to_string(p) for p in momenta]
    eps = 1e-6    # tolerance for which terms to set to 0
    Gamma_B, Gamma_B_inv = [{}, {}, {}], [{}, {}, {}]
    for i in range(3):
        for p in pkeys:
            Gamma[i][p][np.abs(Gamma[i][p]) < eps] = 0
            Gamma_B[i][p] = Gamma[i][p][0, 0, :, 0, :]
            # print(p)
            # print(Gamma_B[i][p])
            Gamma_B_inv[i][p] = np.linalg.inv(Gamma_B[i][p])
    # Gamma_B = [{p : Gamma[i][p][0, 0, :, 0, :] for p in pkeys} for i in range(3)]    # strip off extra indices
    # Gamma_B_inv = [{p : np.linalg.inv(Gamma_B[i][p]) for p in pkeys} for i in range(3)]
    return Gamma_B, Gamma_B_inv

# uses the
def born_term_irreps2(momenta):
    Gamma_B = [{}, {}, {}]
    Gamma_B_inv = [{}, {}, {}]
    file = '/Users/theoares/lqcd/npr_momfrac/born_term_irreps.h5'
    f = h5py.File(file, 'r')
    props, threepts = {}, [{}, {}, {}]
    for p in momenta:
        pstring = plist_to_string(p)
        prop_path = 'prop/' + pstring
        x = np.einsum('ijab->aibj', f[prop_path][()])    # reshape and add bootstrap + configuration dimensions
        props[pstring] = np.expand_dims(x, axis = 0)

        for j in range(3):
            threept_path = 'O' + str(j + 1) + '/' + pstring
            y = np.einsum('ijab->aibj', f[threept_path][()])
            threepts[j][pstring] = np.expand_dims(y, axis = 0)
    f.close()
    props_inv = invert_prop(props, B = 1)

    Gamma = []
    for i in range(3):
        Gamma.append(amputate(props_inv, threepts[i], B = 1))
    pkeys = [plist_to_string(p) for p in momenta]
    eps = 1e-6    # tolerance for which terms to set to 0
    Gamma_B, Gamma_B_inv = [{}, {}, {}], [{}, {}, {}]
    for i in range(3):
        for p in pkeys:
            Gamma[i][p][np.abs(Gamma[i][p]) < eps] = 0
            Gamma_B[i][p] = Gamma[i][p][0, 0, :, 0, :]
            Gamma_B_inv[i][p] = np.linalg.inv(Gamma_B[i][p])
    return Gamma_B, Gamma_B_inv

# Compute operator renormalization Z(p)
def get_Z(Zq, Gamma, Gamma_B_inv):
    Z = {}
    for p in mom_str_list:
        # Z[p] = np.zeros((n_boot, num_cfgs), dtype = np.complex64)
        Z[p] = np.zeros((n_boot), dtype = np.complex64)
        for b in range(n_boot):
            trace = np.einsum('aiaj,ji', Gamma[p][b], Gamma_B_inv[p])
            Z[p][b] = 12 * Zq[p][b] / trace
    return Z

# Gets the nth element of tau13 from a tensor O[mu, nu]
def tau13_irrep(O, n):
    assert n in [0, 1, 2]
    if n == 0:
        return (O[2, 2] - O[3, 3]) / np.sqrt(2)
    elif n == 1:
        return (O[0, 0] - O[1, 1]) / np.sqrt(2)
    else:
        return (O[0, 0] + O[1, 1] - O[2, 2] - O[3, 3]) / 2

# Assumes O2 can have a tensor structure with Dirac indices on the lect.
def inner(O1, O2):
    # tau13_irrep(O1, n) has a Dirac structure. tau13_irrep(O2, n) has a Dirac structure and color structure
    traces = [np.einsum('ij,ji', tau13_irrep(O1, n), tau13_irrep(O2, n)) for n in range(3)]
    # return sum(traces)# / 4
    return sum(traces)

# Returns the matrix A_{ab} discussed in Sergei's thesis.
def A_ab(p):
    return np.array([
        [inner(Lambda1(p), Lambda1(p)), inner(Lambda1(p), Lambda2(p))],
        [inner(Lambda2(p), Lambda1(p)), inner(Lambda2(p), Lambda2(p))]
    ])

# p is lattice momentum
def detA(p):
    return inner(Lambda1(p), Lambda1(p)) * inner(Lambda2(p), Lambda2(p)) - inner(Lambda1(p), Lambda2(p)) ** 2

def A_inv_ab(p):
    A11 = inner(Lambda1(p), Lambda1(p))
    A12 = inner(Lambda1(p), Lambda2(p))
    A22 = inner(Lambda2(p), Lambda2(p))
    det = A11 * A22 - A12 * A12
    return (1 / det) * np.array([
        [A22, -A12],
        [-A12, A11]
    ])

# Returns a in units of GeV^{-1}
def fm_to_GeV(a):
    return a / .197327

# Returns the anomalous dimension in Landau gauge in RI'
def gamma_RI_Landau(alpha = .2956):
    a = alpha / (4 * np.pi)
    nf = 3
    z3 = zeta(3)
    return 32 * a / 9 - (4 / 243) * (378 * nf - 6005) * (a ** 2) \
        + (8 / 6561) * (10998 * (nf ** 2) - 6318 * z3 * nf - 467148 * nf - 524313 * z3 + 3691019) * (a ** 3)


# returns the conversion factor Z_{RI'} / Z_{MSbar} in Landau gauge, from J.A. Gracey's paper.
# Note that this is the inverse of the conversion factor that we want, and in Gracey's convenction
# the coupling is a = g^2 / (16*pi^2) = alpha / (4*pi), and we are matching at mu_0 = 2 GeV
def C_O(alpha = .2956):
    a, cf, ca, tf, nf = alpha / (4 * np.pi), 4/3, 3, 1/2, 3
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    return 1 + 31 * cf * a / 9 + \
        ((-1782 * z3 + 6404) * ca + (1296 * z3 - 228) * cf - 2668 * tf * nf) * cf * (a ** 2) / 162 \
        + ((-11944044 * z3 + 746496 * z4 + 524880 * z5 + 38226589) * (ca ** 2) \
          + (-4914432 * z3 - 2239488 * z4 + 8864640 * z5 + 3993332) * ca * cf \
          + (369792 * z3 - 1492992 * z4 - 24752896) * ca * tf * nf \
          + (10737792 * z3 + 1492992 * z4 - 9331200 * z5 - 3848760) * (cf ** 2) \
          - (-3234816 * z3 - 1492992 * z4 + 9980032) * cf * tf * nf \
          + (221184 * z3 + 3391744) * (tf ** 2) * (nf ** 2)) * cf * (a ** 3) / 69984

# Returns Z_{RI'} / Z_{MSbar} in Landau gauge for the quark renormalization Z_q.
def C_q(alpha = .2956):
    a, cf, ca, tf, nf = alpha / (4 * np.pi), 4/3, 3, 1/2, 3
    z3, z4, z5 = zeta(3), zeta(4), zeta(5)
    return 1 + (5 * cf - (24 * z3 + 82) * ca + 28 * tf * nf) * cf * (a ** 2) / 8 \
        + ((678024 * z3 + 22356 * z4 - 213840 * z5 - 1274056) * (ca ** 2) \
        + (-228096 * z3 - 31104 * z4 + 103680 * z5 + 215352) * (ca * cf) \
        + 31536 * (cf ** 2) + (-89856 * z3 + 760768) * ca * tf * nf \
        + (68256 - 82944 * z3) * cf * tf * nf - 100480 * (tf ** 2) * (nf ** 2)) * (cf * (a ** 3) / 5184)

# pass in Z before we do statistics
def to_MSbar(Z, momenta):
    #Zms = {}
    Zms = []
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
    for i, k in enumerate(momenta):
        kstring = plist_to_string(k)
        p_lat = to_lattice_momentum(k)
        # Adjusted R in table X to fit the operator I'm using.
        # R = ((p[2] ** 2 - p[3] ** 2) ** 2) / (2 * square(p) * (p[2] ** 2 + p[3] ** 2))
        # TODO how should I work with the R in this expression? (Table X in Gockeler)
        R0 = (p_lat[3] ** 2 - (p_lat[0] ** 2 + p_lat[1] ** 2 + p_lat[2] ** 2) / 3) ** 2 / (2*square(p_lat) * (p_lat[3] ** 2 + (p_lat[0] ** 2 + p_lat[1] ** 2 + p_lat[2] ** 2) / 9))
        # R1 = = ((p_lat[2] ** 2 - p_lat[3] ** 2) / np.sqrt(2)) ** 2 / (2 * square(p_lat) * ())
        R = R0
        c1 = c11 + c12 * R
        c2 = c21 + b2 + b2 + c22 * R
        c3 = c31 + b2 * c11 + b3 + (c32 + b2 * c12) * R
        x = (g ** 2) / (16 * (np.pi ** 2))
        Zconv = 1 + c1 * x + c2 * (x ** 2) + c3 * (x ** 3)
        #Zms[pstring] = Zconv * Z[pstring]
        Zms.append(Zconv * Z[i])
    return Zms

# Converts a wavevector to an energy scale using ptwid. Lattice parameter is a = A femtometers.
def k_to_mu_ptwid(k, A = .1167):
    aGeV = fm_to_GeV(A)
    return 2 / aGeV * np.sqrt(sum([np.sin(np.pi * k[mu] / LL[mu]) ** 2 for mu in range(4)]))

def k_to_mu_p(k, A = .1167):
    aGeV = fm_to_GeV(A)
    return (2 * np.pi / aGeV) * np.sqrt(sum([(k[mu] / L[mu]) ** 2 for mu in range(4)]))

def save_mu_sigma(mu, sigma, directory, clear_path = False):
    mu_file = directory + '/mu.npy'
    sigma_file = directory + '/sigma.npy'
    if clear_path:
        os.remove(mu_file)
        os.remove(sigma_file)
    np.save(mu_file, mu)
    np.save(sigma_file, sigma)
    return True

def load_data_h5(file):
    print('Loading ' + str(file) + '.')
    f = h5py.File(file, 'r')
    k_list = f['momenta'][()]
    # p_list = np.array([2 * np.sin(np.pi * np.array([k[mu] / LL[mu] for mu in range(4)])) for k in k_list])
    p_list = np.array([to_lattice_momentum(k) for k in k_list])
    Z = f['Z'][()]
    Zq = f['Zq'][()]
    cfgnum = f['cfgnum'][()]
    f.close()
    return k_list, p_list, Z, Zq, cfgnum

def load_Zq(file):
    f = h5py.File(file, 'r')
    k_list = f['momenta'][()]
    # p_list = np.array([2 * np.sin(np.pi * np.array([k[mu] / LL[mu] for mu in range(4)])) for k in k_list])
    p_list = np.array([to_lattice_momentum(k) for k in k_list])
    Zq = f['Zq'][()]
    f.close()
    return k_list, p_list, Zq

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

def get_point_list(file):
    f = h5py.File(file, 'r')
    allpts = f['prop']
    pts = []
    for ptstr in allpts.keys():    # should be of the form x#y#z#t#
        parts = re.split('x|y|z|t', ptstr)[1:]
        pt = [int(x) for x in parts]
        pts.append(pt)
    f.close()
    return pts

# partition mom_list into orbits by O(3) norm and p[3]
def get_O3_orbits(k_list):
    orbits, psquared_p3_list = [], []    # these should always be the same size
    for p in k_list:
        psquared = p[0] ** 2 + p[1] ** 2 + p[2] ** 2
        if (psquared, p[3]) in psquared_p3_list:    # then add to existing orbit
            idx = psquared_p3_list.index((psquared, p[3]))
            orbits[idx].append(p)
        else:
            psquared_p3_list.append((psquared, p[3]))
            orbits.append([p])
    return orbits, psquared_p3_list

# Z is a (n_momentum x n_boot) matrix
def average_O3_orbits(Z, k_list):
    orbits, psquared_p3_list = get_O3_orbits(k_list)
    k_rep_list, Zavg, sigma = [], [], []
    for coset in orbits:
        # idx_list = [k_list.index(p) for p in coset]
        idx_list = [np.where(k_list == p) for p in coset]
        k_rep_list.append(coset[0])    # take the first representative
        Zavg.append(np.mean(Z[idx_list, :]))
        sigma.append(np.std(Z[idx_list, :]))
    return k_rep_list, Zavg, sigma

# Averages point propagators to determin S(p) and G(p). Also attempts to determine
# the error in these respective quantities.
def average_over_points(S_list, G_list, pt_list):
    N = len(pt_list)
    S_ave, G_ave = {}, {}
    sigma_S, sigma_G = {}, {}
    # global mom_str_list
    for pstring in mom_str_list:
        S_ave[pstring], G_ave[pstring] = np.zeros(S_list[0][pstring].shape, dtype = np.complex64), \
                    np.zeros(G_list[0][pstring].shape, dtype = np.complex64)
        for idx, pt in enumerate(pt_list):
            S_ave[pstring] += S_list[idx][pstring]
            G_ave[pstring] += G_list[idx][pstring]
        S_ave[pstring] *= hypervolume / N
        G_ave[pstring] *= hypervolume / N

        sigma_S[pstring], sigma_G[pstring] = np.zeros(S_list[0][pstring].shape, dtype = np.complex64), \
                    np.zeros(G_list[0][pstring].shape, dtype = np.complex64)
        var_S = np.sum([np.square(S_ave[pstring] - S_list[idx][pstring]) for idx in range(len(pt_list))])
        sigma_S[pstring] = np.sqrt(var_S)
        var_G = np.sum([np.square(G_ave[pstring] - G_list[idx][pstring]) for idx in range(len(pt_list))])
        sigma_S[pstring] = np.sqrt(var_S)
    return S_ave, G_ave, sigma_S, sigma_G

# Run analysis on a set of point sources by averaging over the S and G's. Assumes that
# three point functions are tied up in QLUA.
def run_analysis_point_sources(directory, momenta, mu = None, s = 0):
    momenta_str_list = [plist_to_string(p) for p in momenta]
    start = time.time()
    Z_list, Zq_list = [], []
    # Gamma_B, Gamma_B_inv = born_term(mu = mu, momenta = momenta)
    Gamma_B, Gamma_B_inv = born_term_numerical(mu = mu, momenta = momenta)
    global mom_list
    global mom_str_list
    for idx, p in enumerate(momenta_str_list):
        print('Computing for sink momentum ' + str(idx) + ' out of ' \
                    + str(len(momenta_str_list)) + '. Value: ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]
        S_ave, G_ave, N = readfile(directory, pointsrc = True, mu = mu, sink_momenta = [p])
        props_boot = bootstrap(S_ave, seed = s)
        threept_boot = bootstrap(G_ave, seed = s)
        props_inv = invert_prop(props_boot)
        Gamma = amputate(props_inv, threept_boot)
        Zq = quark_renorm(props_inv)
        Z = get_Z(Zq, Gamma, Gamma_B_inv)
        Z_list.append(Z[p])
        Zq_list.append(Zq[p])
        # Time per iteration
        print('Elapsed time: ' + str(time.time() - start))
    return np.array(Z_list), np.array(Zq_list)

# Run analysis on the H(4) irreps
def run_analysis_irreps(directory, momenta, s = 0):
    momenta_str_list = [plist_to_string(p) for p in momenta]
    start = time.time()
    Z_list = [[], [], []]
    Zq_list = []
    Gamma_B_list, Gamma_B_inv_list = born_term_irreps(momenta)
    global mom_list
    global mom_str_list
    for idx, p in enumerate(momenta_str_list):
        print('Computing for sink momentum ' + str(idx) + ' out of ' \
                    + str(len(momenta_str_list)) + '. Value: ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]

        G = [0] * 4
        for mu in range(4):
            S, G[mu], N = readfile(directory, pointsrc = True, mu = mu, sink_momenta = [p])
        G_irreps = [
            {p : (G[2][p] - G[3][p]) / np.sqrt(2) for p in S.keys()},
            {p : (G[0][p] - G[1][p]) / np.sqrt(2) for p in S.keys()},
            {p : (G[0][p] + G[1][p] - G[2][p] - G[3][p]) / 2 for p in S.keys()}
        ]
        props_boot = bootstrap(S, seed = s)
        props_inv = invert_prop(props_boot)
        Zq = quark_renorm(props_inv)     # quark_renorm returns a dictionary

        for i, Gi in enumerate(G_irreps):
            Gamma_B, Gamma_B_inv = Gamma_B_list[i], Gamma_B_inv_list[i]
            threept_boot = bootstrap(Gi, seed = s)
            Gamma = amputate(props_inv, threept_boot)
            Z_list[i].append(get_Z(Zq, Gamma, Gamma_B_inv)[p])
        Zq_list.append(Zq[p])
        # Time per iteration
        print('Elapsed time: ' + str(time.time() - start))
    return np.array(Z_list), np.array(Zq_list)

def run_analysis_mixing(directory, momenta, s = 0):    #, mixing_list = None, s = 0):
    momenta_str_list = [plist_to_string(p) for p in momenta]
    # if mixing_list:
    #     Ainv_list, L1_list, L2_list = mixing_list
    # else:
    #     Ainv_list, L1_list, L2_list = [], [], []
    #     for p in momenta:
    #         p_lat = to_lattice_momentum(p)
    #         L1_list.append(Lambda1(p_lat))
    #         L2_list.append(Lambda2(p_lat))
    #         Ainv_list.append(A_inv_ab(p_lat))
    start = time.time()
    # Z11_list, Z12_list, Zq_list = [], [], []
    Pi_11_list, Pi_12_list, Zq_list = [], [], []

    Gamma_munu_list = []

    Gamma_Dirac_list = []
    global mom_list
    global mom_str_list
    for idx, p in enumerate(momenta_str_list):
        print('Computing for sink momentum ' + str(idx) + ' out of ' \
                    + str(len(momenta_str_list)) + '. Value: ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]
        p_lat = to_lattice_momentum(pstring_to_list(p))    # converts k to ptwiddle
        # print([2*np.pi*pstring_to_list(p)[mu] / LL[mu] for mu in range(4)])
        # print(p_lat)
        G = [0] * 4
        for mu in range(4):
            S, G[mu], N = readfile(directory, pointsrc = True, mu = mu, sink_momenta = [p])

        props_boot = bootstrap(S, seed = s)
        props_inv = invert_prop(props_boot)
        Zq = quark_renorm(props_inv)[p]     # quark_renorm returns a dictionary
        Zq_list.append(Zq)

        G_irreps = [
            {p : (G[2][p] - G[3][p]) / np.sqrt(2) for p in S.keys()},
            {p : (G[0][p] - G[1][p]) / np.sqrt(2) for p in S.keys()},
            {p : (G[0][p] + G[1][p] - G[2][p] - G[3][p]) / 2 for p in S.keys()}
        ]
        # Gamma = []

        # for mu, Gmu in enumerate(G):
        #     print('mu is: ' + str(mu) + '. First bootstrapped sample of Gamma_mu is: ')
        #     threept_boot = bootstrap(Gmu)
        #     x = amputate(props_inv, threept_boot)[p]
        #     color_trace = np.einsum('baiaj->bij', x) / 3
        #     print(color_trace[0, :, :])

        Gamma_munu = []
        for mu in range(4):
            threept_boot_munu = bootstrap(G[mu], seed = s)
            x_munu = amputate(props_inv, threept_boot_munu)[p]
            # Gamma.append(x)
            color_trace_munu = np.einsum('baiaj->bij', x_munu) / 3
            Gamma_munu.append(color_trace_munu)    # trace color indices
        Gamma_munu_list.append(Gamma_munu)

        Gamma_Dirac = []
        for Gi in G_irreps:
            threept_boot = bootstrap(Gi, seed = s)
            x = amputate(props_inv, threept_boot)[p]
            # Gamma.append(x)
            color_trace = np.einsum('baiaj->bij', x) / 3
            Gamma_Dirac.append(color_trace)    # trace color indices
        A = A_ab(p_lat)
        # A_inv, L1, L2 = Ainv_list[idx], L1_list[idx], L2_list[idx]
        A_inv, L1, L2 = A_inv_ab(p_lat), Lambda1(p_lat), Lambda2(p_lat)
        Gamma_Dirac_list.append(Gamma_Dirac)
        # print('A is:')
        # print(A)
        #
        # print('Ainv is:')
        # print(A_inv)

        # Lambda's agree and A, Ainv agrees. See if Gamma and v agree
        # Mathematica confirms that for the 0 boot idx, v, Pi_{11}, and Pi_{12} are all correct.

        # print('Gamma Dirac boot index 0 is:')
        # print('First element of irrep')
        # print(Gamma_Dirac[0][0, :, :])
        #
        # print('Second element of irrep')
        # print(Gamma_Dirac[1][0, :, :])
        #
        # print('Third element of irrep')
        # print(Gamma_Dirac[2][0, :, :])

        # trace over color indices first?

        # v = np.array([
        #     sum([np.einsum('ij,bajai->b', tau13_irrep(L1, n), Gamma[n]) for n in range(3)]),
        #     sum([np.einsum('ij,bajai->b', tau13_irrep(L2, n), Gamma[n]) for n in range(3)]),
        # ]) / 3    # TODO normalization. Should probably color average, and if I dirac average so does the other inner product

        v = np.array([
            sum([np.einsum('ij,bji->b', tau13_irrep(L1, n), Gamma_Dirac[n]) for n in range(3)]),
            sum([np.einsum('ij,bji->b', tau13_irrep(L2, n), Gamma_Dirac[n]) for n in range(3)]),
        ])

        # print('v is: ')
        # print(v[:, 0])

        Pi_11, Pi_12 = A_inv.dot(v)
        # print(Pi_11.shape)

        # Check that \Gamma = Pi_11 Lambda1 + Pi_22 Lambda 2 (shouldn't be exactly true, this is
        # the projection of \Gamma onto the subspace spanned by Lambda1 and Lambda2)
        # PiLambda = np.einsum('b,mnij->mnbij', Pi_11, L1) + np.einsum('b,mnij->mnbij', Pi_12, L2)
        # print(PiLambda.shape)
        # PiLambda_irr = [
        #     (PiLambda[2, 2] - PiLambda[3, 3]) / np.sqrt(2),
        #     (PiLambda[0, 0] - PiLambda[1, 1]) / np.sqrt(2),
        #     (PiLambda[0, 0] + PiLambda[1, 1] - PiLambda[2, 2] - PiLambda[3, 3]) / 2
        # ]

        # print('Gamma at 0 is: ')
        # print(Gamma_Dirac[0])
        # print('Difference in Gamma and Pi_ij Lambda_j for irrep 0 is: ')
        # print(PiLambda_irr[0] - Gamma_Dirac[0])
        # print('Difference in Gamma and Pi_ij Lambda_j for irrep 1 is: ')
        # print(PiLambda_irr[1] - Gamma_Dirac[1])
        # print('Difference in Gamma and Pi_ij Lambda_j for irrep 2 is: ')
        # print(PiLambda_irr[2] - Gamma_Dirac[2])

        # print(Zq)
        # print('Pi_11 is:')
        # print(Pi_11)
        # print('Pi_12 is:')
        # print(Pi_12)

        # Z11, Z12 = Zq / Pi_11, Zq / Pi_12
        # Z11, Z12 = Zq * Pi_11, Zq * Pi_12    # this makes a lot more sense

        # print(Z11)
        # Z11_list.append(Z11)
        # Z12_list.append(Z12)
        # print(Zq)
        Pi_11_list.append(Pi_11)
        Pi_12_list.append(Pi_12)
        print('Z computed. Elapsed time: ' + str(time.time() - start))
    # return np.array(Z11_list), np.array(Z12_list), np.array(Zq_list)
    return np.array(Pi_11_list), np.array(Pi_12_list), np.array(Zq_list), np.array(Gamma_munu_list)#np.array(Gamma_Dirac_list)

# Gets irreps explicitly from being run on QLUA
def run_analysis_irreps2(directory, momenta, s = 0):
    momenta_str_list = [plist_to_string(p) for p in momenta]
    start = time.time()
    Z_list = [[], [], []]
    Zq_list = []
    Gamma_B_list, Gamma_B_inv_list = born_term_irreps2(momenta)
    global mom_list
    global mom_str_list
    for idx, p in enumerate(momenta_str_list):
        print('Computing for sink momentum ' + str(idx) + ' out of ' \
                    + str(len(momenta_str_list)) + '. Value: ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]

        G_irreps = [0] * 3
        for j in range(3):
            S, G_irreps[j], N = readfile(directory, j = j, sink_momenta = [p])
        props_boot = bootstrap(S, seed = s)
        props_inv = invert_prop(props_boot)
        Zq = quark_renorm(props_inv)     # quark_renorm returns a dictionary

        for i, Gi in enumerate(G_irreps):
            Gamma_B, Gamma_B_inv = Gamma_B_list[i], Gamma_B_inv_list[i]
            threept_boot = bootstrap(Gi, seed = s)
            Gamma = amputate(props_inv, threept_boot)
            Z_list[i].append(get_Z(Zq, Gamma, Gamma_B_inv)[p])
        Zq_list.append(Zq[p])
        # Time per iteration
        print('Elapsed time: ' + str(time.time() - start))
    return np.array(Z_list), np.array(Zq_list)

def Zq_analysis(directory, momenta, N = None, s = 0):
    start = time.time()
    Zq_list = []
    global mom_list
    global mom_str_list
    mom_str_list_cp = mom_str_list

    for idx, p in enumerate(mom_str_list_cp):
        print('Computing for sink momentum ' + str(idx) + ' out of ' \
                    + str(len(mom_str_list_cp)) + '. Value: ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]
        S_list, G_list = [], []
        props, threepts, N = readfile(directory, pointsrc = True, mu = -1, sink_momenta = mom_str_list)
        props_boot = bootstrap(props)
        props_inv = invert_prop(props_boot)
        Zq = quark_renorm(props_inv)
        Zq_list.append(Zq[p])

        # Time per iteration
        print('Elapsed time: ' + str(time.time() - start))
    return np.array(Zq_list)

def current_analysis(directory, momenta, s = 5):
    momenta_str_list = [plist_to_string(p) for p in momenta]
    start = time.time()
    ZV_list, ZA_list = [[], [], [], []], [[], [], [], []]    # NOTE [[]] * 4 MAKES THE SAME LIST 4 TIMES, NOT 4 SEPARATE ONES
    Zq_list = []
    # GVB, GAB = 2 * gamma, np.array([2 * np.dot(gamma[mu], gamma5) for mu in range(4)])
    # AFTER RUNNING FREE FIELD, Born term should just be gamma[mu] and gamma[mu] * gamma5
    GVB, GAB = gamma, np.array([np.dot(gamma[mu], gamma5) for mu in range(4)])
    GammaV_B_inv = np.array([np.linalg.inv(x) for x in GVB])
    GammaA_B_inv = np.array([np.linalg.inv(x) for x in GAB])
    # convert to a dictionary because my code is dumb
    GVB_inv = [{p : GammaV_B_inv[mu] for p in momenta_str_list} for mu in range(4)]
    GAB_inv = [{p : GammaA_B_inv[mu] for p in momenta_str_list} for mu in range(4)]
    global mom_list
    global mom_str_list
    for idx, p in enumerate(momenta_str_list):
        print('Computing for sink momentum ' + str(idx) + ' out of ' \
                    + str(len(momenta_str_list)) + '. Value: ' + p)
        mom_list = [pstring_to_list(p)]
        mom_str_list = [p]

        GV, GA = [0] * 4, [0] * 4
        for mu in range(4):
            S, GV[mu], N = readfile(directory, j = mu, sink_momenta = [p])
            S, GA[mu], N = readfile(directory, j = 50 + mu, sink_momenta = [p])
            # S, GV[mu], N = readfile(directory, j = mu, dpath = '/q1111', sink_momenta = [p])
            # S, GA[mu], N = readfile(directory, j = 50 + mu, dpath = '/q1111', sink_momenta = [p])
        props_boot = bootstrap(S, seed = s)
        props_inv = invert_prop(props_boot)
        Zq = quark_renorm(props_inv)     # quark_renorm returns a dictionary

        # print(GV[0]['p1111'][2, 2])
        # print('GA is here:')
        # print(GA[0]['p1111'][2, 2])
        # break

        for mu in range(4):
            threeptV_boot = bootstrap(GV[mu], seed = s)
            threeptA_boot = bootstrap(GA[mu], seed = s)
            GammaV = amputate(props_inv, threeptV_boot)
            GammaA = amputate(props_inv, threeptA_boot)
            ZV_list[mu].append(get_Z(Zq, GammaV, GVB_inv[mu])[p])
            ZA_list[mu].append(get_Z(Zq, GammaA, GAB_inv[mu])[p])
        Zq_list.append(Zq[p])
        # Time per iteration
        print('Elapsed time: ' + str(time.time() - start))
    return np.array(ZV_list), np.array(ZA_list), np.array(Zq_list)

def get_props_threepts(directory, dpath = ''):
    files = []
    for (dirpath, dirnames, file) in os.walk(directory):
        files.extend(file)
    global num_cfgs
    num_cfgs = len(files)
    props = np.zeros((num_cfgs, 3, 4, 3, 4), dtype = np.complex64)
    # cfgnum, mu, color, Dirac, color, Dirac
    threepts = np.zeros((num_cfgs, 4, 3, 4, 3, 4), dtype = np.complex64)
    idx = 0
    for file in files:
        path_to_file = directory + '/' + file
        f = h5py.File(path_to_file, 'r')
        prop_path = 'prop/' + dpath
        threept_paths = ['threept' + str(i + 1) + str(i + 1) + '/' + dpath for i in range(4)]

        prop = f[prop_path][()]
        threepts = np.array([f[threept_paths[i]][()] for i in range(4)])
        # TODO what does the new propagator index structure look like?
        props[idx, :, :, :, :] = np.einsum('ijab->aibj', prop)
        threepts[idx, :, :, :, :, :] = np.einsum('mijab->maibj', threept)
        idx += 1
    return props, threepts
