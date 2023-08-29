#####################################################################
# Code for 8.871 Problem Set 3. Note that this uses a variety of    #
# code written for my research which is stored in the               #
# '/Users/theoares/lqcd/utilities' folder. I will include relevant  #
# code snippets in my writeup if they are imported from another     #
# script that I wrote.                                              #
#                                                                   #
# Note that I will use GPT to generate configurations and compute   #
# observables on these configurations.                              #
#####################################################################

n_boot = 100
import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *
from formattools import *
import plottools as pt

import gpt

out_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/figs/'
cfg_dir = '/Users/theoares/Dropbox (MIT)/classes/Fall 2022/8.871/ps3/cfgs/'
np.random.seed(20)                 # seed the RNG for reproducibility
Nc = 3
Nd = 4
LL = [16, 16, 16, 32]
beta = 6.0
n_therm = 5
n_cfgs = 100
grid = gpt.grid(LL, gpt.double)

rng = gpt.random("test")
grid_eo = grid.checkerboarded(gpt.redblack)
mask_rb = gpt.complex(grid_eo)
mask_rb[:] = 1
mask = gpt.complex(grid)
def staple(U, mu):
    st = gpt.lattice(U[0])
    st[:] = 0
    for nu in range(Nd):
        if mu != nu:
            st += beta * gpt.qcd.gauge.staple(U, mu, nu) / U[0].otype.Nc
    return st
gpt.default.push_verbose("su2_heat_bath", False)

markov = gpt.algorithms.markov.su2_heat_bath(rng)

n_corr = 50
cfgs = []
plaqs, rects = [], []
configs = []
U = gpt.qcd.gauge.unit(grid)

#####################################################################
############################# THERMALIZE ############################
#####################################################################

for _ in range(n_therm):
    for it in range(n_corr):
        plaqs.append(gpt.qcd.gauge.plaquette(U))
        rects.append(gpt.qcd.gauge.rectangle(U, 2, 1))
        gpt.message(f"SU(2)-subgroup heatbath {it} has P = {plaqs[it]}, R_2x1 = {rects[it]}")
        for cb in [gpt.even, gpt.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            gpt.set_checkerboard(mask, mask_rb)
            for mu in range(Nd):
                st = gpt.eval(staple(U, mu))
                markov(U[mu], st, mask)
print('Thermalization complete.')

#####################################################################
########################## GENERATE CONFIGS #########################
#####################################################################

for icfg in range(n_cfgs):
    for it in range(n_corr):
        plaqs.append(gpt.qcd.gauge.plaquette(U))
        rects.append(gpt.qcd.gauge.rectangle(U, 2, 1))
        gpt.message(f"SU(2)-subgroup heatbath {it} has P = {plaqs[it]}, R_2x1 = {rects[it]}")
        for cb in [gpt.even, gpt.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            gpt.set_checkerboard(mask, mask_rb)

            for mu in range(Nd):
                st = gpt.eval(staple(U, mu))
                markov(U[mu], st, mask)
    configs.append(U)
    path = cfg_dir + 'cfg' + str(icfg) + '.lat'
    gpt.save(path, U, gpt.format.nersc())
    print('Configuration saved at: ' + path)
