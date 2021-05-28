import numpy as np
from scipy.optimize import root
import h5py
import os
import xml.etree.ElementTree as ET
from utils import *

Nd = 4
Nc = 3

# This script takes in XML input and reformats it as HDF5
xml_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/tests/zero_nubb_test.dat.xml'
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

# write to file
out_file = ''
