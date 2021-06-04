import numpy as np
from scipy.optimize import root
import h5py
import os
import xml.etree.ElementTree as ET
from utils import *

Nd = 4
Nc = 3

# This script takes in XML input and reformats it as HDF5. Output will be taken
# from the dir directory, which is at the level of meas/24I/ml0p01/cfg200 in the
# example file tree. The parsed output will be stored in the hdf5 directory in the
# corresponding parent folder, with the file name cfg200.h5.
# TODO in the future, dir will be the folder labeled with all the output from a config,
# and enumerate all the momentum labels (ex cfg100/ will have mom_2.dat.xml and mom_3.dat.xml).
# So, ex file structure:
#                                 meas
#                               /     \
#                             24I     32I
#                            /  \        \
#                      ml0p01    ml0p005       <----------- parent
#                   /   |    \
#              hdf5 cfg100 ...  cfg200          <----------- dir = meas/24I/ml0p01/cfg200
#             /              /  |   \
#   cfg200.h5   mom_1.dat.xml  ...   mom_6.dat.xml

# INPUT PARAMETERS
home = '/Users/theoares'
# home = '/Users/poare'
cfg_root = 'cfgEXAMPLE'                                     # 'cfg200'
mom_idx_list = [2]                                          #[1, 2, ..., 6]

# FIXED SETUP
parent = home + '/Dropbox (MIT)/research/0nubb/tests/'      # meas/24I/ml0p01
dir = parent + cfg_root                                # meas/24I/ml0p01/cfg200
out_file = parent + 'hdf5/' + cfg_root + '.h5'
f = h5py.File(out_file, 'w')

for mom_idx in mom_idx_list:
    xml_file = dir + '/mom_' + str(mom_idx) + '.dat.xml'
    print('Reading data from ' + xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()

    hadspec = root[6][0]       # access hadspec/elem
    for child in hadspec:
        print(child)

    props = hadspec[1]
    GV = hadspec[2]
    GA = hadspec[3]
    four_point = hadspec[4]

    # Turns a multi1d<int> into a vector of ints. Input text should be a string
    toVec = lambda text : np.array([int(token) for token in str.split(text)])

    # pass in the text from the first matrix tag in the propagator as full_mat
    def parse_prop_matrix(full_mat):
        prop_mat = np.zeros((Nd, Nd, Nc, Nc), dtype = np.complex64)
        for ii in range(Nd * Nd):
            dirac_index = full_mat[ii].attrib
            i, j = int(dirac_index['row']), int(dirac_index['col'])
            color_mat = full_mat[ii][0] # length 9
            for cc in range(Nc * Nc):
                color_index = color_mat[cc].attrib
                c, d = int(color_index['row']), int(color_index['col'])
                re_elem = color_mat[cc][0].text
                im_elem = color_mat[cc][1].text
                prop_mat[i, j, c, d] = np.complex(float(re_elem), float(im_elem))
        return prop_mat

    # parse propagators
    mom_labels = ['k1', 'k2', 'q']
    k_list = [0, 0, 0]
    S_list = [0, 0, 0] # readfiles reads in as i j a b
    for i in range(3):
        k_list[i] = toVec(props[i][0].text)
        Ski_mat = props[i][1][0]            # extra 0 to strip off <Matrix> tag. len(Ski) is 16 = Nd * Nd
        S_list[i] = parse_prop_matrix(Ski_mat)
    k_out = np.array(k_list)
    S_out = np.array(S_list)

    # parse GV
    GV_list = [0, 0, 0, 0]
    for mu in range(4):
        assert mu == int(GV[mu][0].text), 'Bad data format passed in.'
        GV_mu_mat = GV[mu][1][0]        # correlator
        GV_list[mu] = parse_prop_matrix(GV_mu_mat)
    GV_out = np.array(GV_list)

    # parse GA
    GA_list = [0, 0, 0, 0]
    for mu in range(4):
        assert mu == int(GA[mu][0].text), 'Bad data format passed in.'
        GA_mu_mat = GA[mu][1][0]        # correlator
        GA_list[mu] = parse_prop_matrix(GA_mu_mat)
    GA_out = np.array(GA_list)

    # parse four-points
    n_gammas = 16
    four_point_out = np.zeros((n_gammas, Nd, Nd, Nd, Nd, Nc, Nc, Nc, Nc), dtype = np.complex64)
    for n in range(n_gammas):
        assert n == int(four_point[n][0].text), 'Bad data format passed in.'
        for comp in four_point[n]:
            if comp.tag == 'gamma_value':
                continue
            alpha = int(comp[0].text)
            beta = int(comp[1].text)
            rho = int(comp[2].text)
            sigma = int(comp[3].text)
            a = int(comp[4].text)
            b = int(comp[5].text)
            c = int(comp[6].text)
            d = int(comp[7].text)
            re_val = float(comp[8][0].text)
            im_val = float(comp[8][1].text)
            four_point_out[n, alpha, beta, rho, sigma, a, b, c, d] = np.complex(float(re_val), float(im_val))

    # write to file (use same format that read_files uses to read things in)
    qstr = klist_to_string(k_out[2], 'q')
    for i in range(2):
        f['moms/' + qstr + '/' + mom_labels[i]] = k_out[i]
    for i in range(3):
        f['prop_' + mom_labels[i] + '/' + qstr] = S_out[i]
    for mu in range(Nd):
        f['GV' + str(mu + 1) + '/' + qstr] = GV_out[mu]
        f['GA' + str(mu + 1) + '/' + qstr] = GA_out[mu]
    for n in range(n_gammas):
        f['Gn' + str(n) + '/' + qstr] = four_point_out[n]

f.close()
print('Output file written to: ' + out_file)
