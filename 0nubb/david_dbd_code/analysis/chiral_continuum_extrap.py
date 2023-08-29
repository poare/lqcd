#!/usr/bin/python3

import numpy as np
from scipy.optimize import minimize

import superjack
import SD_chpt

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

fit_O1  = False
fit_O2  = False
fit_O3  = False
fit_O1p = False
fit_O2p = True

plt.style.use('seaborn-pastel')
sns.set_style('ticks')
plt.rcParams['font.sans-serif'] = 'Liberation Sans'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['errorbar.capsize'] = 4
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

tableau10 = [ (31.0,119.0,180.0), (255.0,127.0,14.0) , (44.0,160.0,44.0)  , (214.0,39.0,40.0) , (148.0,103.0,189.0), \
              (140.0,86.0,75.0) , (227.0,119.0,194.0), (127.0,127.0,127.0), (188.0,189.0,34.0), (23.0,190.0,207.0)   ]

for i in range(len(tableau10)):
    r,g,b = tableau10[i]
    tableau10[i] = (r/255.0, g/255.0, b/255.0)

ensembles  = { "16I":110, "24I_ml0p01":52, "24I_ml0p005":53, \
               "32I_ml0p008":33, "32I_ml0p006":42, "32I_ml0p004":48 }
# data_paths = { "16I":         "/home/dmurphyiv/Dropbox/MIT_LQCD/Projects/Double_Beta_Decay/Pi_Pi/16I/prod", \
#                "24I_ml0p01":  "/home/dmurphyiv/Dropbox/MIT_LQCD/Projects/Double_Beta_Decay/Pi_Pi/24I/ml_0p01", \
#                "24I_ml0p005": "/home/dmurphyiv/Dropbox/MIT_LQCD/Projects/Double_Beta_Decay/Pi_Pi/24I/ml_0p005", \
#                "32I_ml0p008": "/home/dmurphyiv/Dropbox/MIT_LQCD/Projects/Double_Beta_Decay/Pi_Pi/32I/ml0p008", \
#                "32I_ml0p006": "/home/dmurphyiv/Dropbox/MIT_LQCD/Projects/Double_Beta_Decay/Pi_Pi/32I/ml0p006", \
#                "32I_ml0p004": "/home/dmurphyiv/Dropbox/MIT_LQCD/Projects/Double_Beta_Decay/Pi_Pi/32I/ml0p004" }
# remoe 16I data
data_paths = { "24I_ml0p01":  "/Users/theoares/Pi_Pi/24I/ml_0p01", \
               "24I_ml0p005": "/Users/theoares/Pi_Pi/24I/ml_0p005", \
               "32I_ml0p008": "/Users/theoares/Pi_Pi/32I/ml0p008", \
               "32I_ml0p006": "/Users/theoares/Pi_Pi/32I/ml0p006", \
               "32I_ml0p004": "/Users/theoares/Pi_Pi/32I/ml0p004" }

# Total number of superjackknife samples
Nsuperboots = 0
for key in ensembles:
 Nsuperboots += ensembles[key]

# Make fake superjackknife distributions for physical mpi and fpi
mpi_pdg   = [ 0.13957018, 0.00000035 ]
fpi_pdg   = [ 0.13041, np.sqrt(0.00003**2 + 0.00020**2) ]
mpi_phys  = superjack.distribution(ensembles)
fpi_phys  = superjack.distribution(ensembles)
mpi_phys.gen_fake_data(mpi_pdg[0], mpi_pdg[1])
# print(mpi_phys.data['24I_ml0p01'])
fpi_phys.gen_fake_data(fpi_pdg[0], fpi_pdg[1])
epi2_phys = mpi_phys**2 / fpi_phys**2
epi2_phys.rescale(1.0/8.0/np.pi**2)

# Refs.: -- arXiv:1106.2714 (16I)
#        -- arXiv:1511.01950 (24I, 32I)
scale_16I = [ 1.7290, 0.0280 ]
scale_24I = [ 1.7844, 0.0049 ]
scale_32I = [ 2.3820, 0.0084 ]

# Make fake superjackknife distributions for 1/a
ainv_16I = superjack.distribution(ensembles)
ainv_24I = superjack.distribution(ensembles)
ainv_32I = superjack.distribution(ensembles)
ainv_16I.gen_fake_data(scale_16I[0], scale_16I[1])
ainv_24I.gen_fake_data(scale_24I[0], scale_24I[1])
ainv_32I.gen_fake_data(scale_32I[0], scale_32I[1])
print("\nGot lattice scales:")
print("\t-- 16I: a^(-1) = {0:.4f} +/- {1:.4f} (GeV)".format(ainv_16I.cv, ainv_16I.std))
print("\t-- 24I: a^(-1) = {0:.4f} +/- {1:.4f} (GeV)".format(ainv_24I.cv, ainv_24I.std))
print("\t-- 32I: a^(-1) = {0:.4f} +/- {1:.4f} (GeV)".format(ainv_32I.cv, ainv_32I.std))

# Get a2
a2_16I = ainv_16I**(-2)
a2_24I = ainv_24I**(-2)
a2_32I = ainv_32I**(-2)
a2_16I.rescale(0.1973269631**2)                 # a2_ENS is valued in fm^2 for the fit, but ainv_ENS is valued in GeV
a2_24I.rescale(0.1973269631**2)
a2_32I.rescale(0.1973269631**2)
print("\nGot lattice scales:")
print("\t-- 16I: a^2 = {0:.4f} +/- {1:.4f} (fm^2)".format(a2_16I.cv, a2_16I.std))
print("\t-- 24I: a^2 = {0:.4f} +/- {1:.4f} (fm^2)".format(a2_24I.cv, a2_24I.std))
print("\t-- 32I: a^2 = {0:.4f} +/- {1:.4f} (fm^2)".format(a2_32I.cv, a2_32I.std))

# Get L
L_16I = ainv_16I**(-1)
L_24I = ainv_24I**(-1)
L_32I = ainv_32I**(-1)
L_16I.rescale(16.0)             # L_ENS = L a (GeV^-1)
L_24I.rescale(24.0)
L_32I.rescale(32.0)
print("\nGot volumes:")
print("\t-- 16I: L = {0:.4f} +/- {1:.4f}".format(L_16I.cv, L_16I.std))
print("\t-- 24I: L = {0:.4f} +/- {1:.4f}".format(L_24I.cv, L_24I.std))
print("\t-- 32I: L = {0:.4f} +/- {1:.4f}".format(L_32I.cv, L_32I.std))

# Debug: check LL
# LL_16I = int( L_16I.cv / np.sqrt(a2_16I.cv) )
# LL_24I = int( L_24I.cv / np.sqrt(a2_24I.cv) )
# LL_32I = int( L_32I.cv / np.sqrt(a2_32I.cv) )
# print("\nGot volumes:")
# print("\t-- 16I: LL = {0:d}".format(LL_16I))
# print("\t-- 24I: LL = {0:d}".format(LL_24I))
# print("\t-- 32I: LL = {0:d}".format(LL_32I))

# Get mpi in physical units
# mpi_16I            = superjack.distribution(ensembles)
mpi_24I_ml0p01     = superjack.distribution(ensembles)
mpi_24I_ml0p005    = superjack.distribution(ensembles)
mpi_32I_ml0p008    = superjack.distribution(ensembles)
mpi_32I_ml0p006    = superjack.distribution(ensembles)
mpi_32I_ml0p004    = superjack.distribution(ensembles)
# mpi_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/mpi.dat" )
mpi_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/mpi.dat" )
mpi_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/mpi.dat" )
mpi_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/mpi.dat" )
mpi_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/mpi.dat" )
mpi_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/mpi.dat" )
# mpi_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/mpi_jacks.dat" ) )
mpi_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/mpi_jacks.dat" ) )
mpi_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/mpi_jacks.dat" ) )
mpi_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/mpi_jacks.dat" ) )
mpi_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/mpi_jacks.dat" ) )
mpi_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/mpi_jacks.dat" ) )
# mpi_16I.set_zeros_to_cv()
mpi_24I_ml0p01.set_zeros_to_cv()
mpi_24I_ml0p005.set_zeros_to_cv()
mpi_32I_ml0p008.set_zeros_to_cv()
mpi_32I_ml0p006.set_zeros_to_cv()
mpi_32I_ml0p004.set_zeros_to_cv()
# mpi_16I.calc_mean()
mpi_24I_ml0p01.calc_mean()
mpi_24I_ml0p005.calc_mean()
mpi_32I_ml0p008.calc_mean()
mpi_32I_ml0p006.calc_mean()
mpi_32I_ml0p004.calc_mean()
# mpi_16I.calc_std()
mpi_24I_ml0p01.calc_std()
mpi_24I_ml0p005.calc_std()
mpi_32I_ml0p008.calc_std()
mpi_32I_ml0p006.calc_std()
mpi_32I_ml0p004.calc_std()
# mpi_16I         = ainv_16I * mpi_16I
mpi_24I_ml0p01  = ainv_24I * mpi_24I_ml0p01
mpi_24I_ml0p005 = ainv_24I * mpi_24I_ml0p005
mpi_32I_ml0p008 = ainv_32I * mpi_32I_ml0p008
mpi_32I_ml0p006 = ainv_32I * mpi_32I_ml0p006
mpi_32I_ml0p004 = ainv_32I * mpi_32I_ml0p004
print("\nGot mpi:")
# print("\t-- 16I:             mpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*mpi_16I.cv        , 1000.0*mpi_16I.std        ))
print("\t-- 24I, ml = 0.01:  mpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*mpi_24I_ml0p01.cv , 1000.0*mpi_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: mpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*mpi_24I_ml0p005.cv, 1000.0*mpi_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: mpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*mpi_32I_ml0p008.cv, 1000.0*mpi_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: mpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*mpi_32I_ml0p006.cv, 1000.0*mpi_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: mpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*mpi_32I_ml0p004.cv, 1000.0*mpi_32I_ml0p004.std))

# Debug: check mpi*L
# mpi_L_16I         = mpi_16I         * L_16I
# mpi_L_24I_ml0p01  = mpi_24I_ml0p01  * L_24I
# mpi_L_24I_ml0p005 = mpi_24I_ml0p005 * L_24I
# mpi_L_32I_ml0p008 = mpi_32I_ml0p008 * L_32I
# mpi_L_32I_ml0p006 = mpi_32I_ml0p006 * L_32I
# mpi_L_32I_ml0p004 = mpi_32I_ml0p004 * L_32I
# print("\nGot mpi*L:")
# print("\t-- 16I:             mpi*L = {0:.1f} +/- {1:.1f} (MeV)".format(mpi_L_16I.cv        , mpi_L_16I.std        ))
# print("\t-- 24I, ml = 0.01:  mpi*L = {0:.1f} +/- {1:.1f} (MeV)".format(mpi_L_24I_ml0p01.cv , mpi_L_24I_ml0p01.std ))
# print("\t-- 24I, ml = 0.005: mpi*L = {0:.1f} +/- {1:.1f} (MeV)".format(mpi_L_24I_ml0p005.cv, mpi_L_24I_ml0p005.std))
# print("\t-- 32I, ml = 0.008: mpi*L = {0:.1f} +/- {1:.1f} (MeV)".format(mpi_L_32I_ml0p008.cv, mpi_L_32I_ml0p008.std))
# print("\t-- 32I, ml = 0.006: mpi*L = {0:.1f} +/- {1:.1f} (MeV)".format(mpi_L_32I_ml0p006.cv, mpi_L_32I_ml0p006.std))
# print("\t-- 32I, ml = 0.004: mpi*L = {0:.1f} +/- {1:.1f} (MeV)".format(mpi_L_32I_ml0p004.cv, mpi_L_32I_ml0p004.std))
# exit(0)

# Get fpi in physical units
# fpi_16I            = superjack.distribution(ensembles)
fpi_24I_ml0p01     = superjack.distribution(ensembles)
fpi_24I_ml0p005    = superjack.distribution(ensembles)
fpi_32I_ml0p008    = superjack.distribution(ensembles)
fpi_32I_ml0p006    = superjack.distribution(ensembles)
fpi_32I_ml0p004    = superjack.distribution(ensembles)
# fpi_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/fpi.dat" )
fpi_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/fpi.dat" )
fpi_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/fpi.dat" )
fpi_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/fpi.dat" )
fpi_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/fpi.dat" )
fpi_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/fpi.dat" )
# fpi_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/fpi_jacks.dat" ) )
fpi_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/fpi_jacks.dat" ) )
fpi_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/fpi_jacks.dat" ) )
fpi_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/fpi_jacks.dat" ) )
fpi_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/fpi_jacks.dat" ) )
fpi_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/fpi_jacks.dat" ) )
# fpi_16I.set_zeros_to_cv()
fpi_24I_ml0p01.set_zeros_to_cv()
fpi_24I_ml0p005.set_zeros_to_cv()
fpi_32I_ml0p008.set_zeros_to_cv()
fpi_32I_ml0p006.set_zeros_to_cv()
fpi_32I_ml0p004.set_zeros_to_cv()
# fpi_16I.calc_mean()
fpi_24I_ml0p01.calc_mean()
fpi_24I_ml0p005.calc_mean()
fpi_32I_ml0p008.calc_mean()
fpi_32I_ml0p006.calc_mean()
fpi_32I_ml0p004.calc_mean()
# fpi_16I.calc_std()
fpi_24I_ml0p01.calc_std()
fpi_24I_ml0p005.calc_std()
fpi_32I_ml0p008.calc_std()
fpi_32I_ml0p006.calc_std()
fpi_32I_ml0p004.calc_std()
# fpi_16I         = ainv_16I * fpi_16I
fpi_24I_ml0p01  = ainv_24I * fpi_24I_ml0p01
fpi_24I_ml0p005 = ainv_24I * fpi_24I_ml0p005
fpi_32I_ml0p008 = ainv_32I * fpi_32I_ml0p008
fpi_32I_ml0p006 = ainv_32I * fpi_32I_ml0p006
fpi_32I_ml0p004 = ainv_32I * fpi_32I_ml0p004
print("\nGot fpi:")
# print("\t-- 16I:             fpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*fpi_16I.cv        , 1000.0*fpi_16I.std        ))
print("\t-- 24I, ml = 0.01:  fpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*fpi_24I_ml0p01.cv , 1000.0*fpi_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: fpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*fpi_24I_ml0p005.cv, 1000.0*fpi_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: fpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*fpi_32I_ml0p008.cv, 1000.0*fpi_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: fpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*fpi_32I_ml0p006.cv, 1000.0*fpi_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: fpi = {0:.1f} +/- {1:.1f} (MeV)".format(1000.0*fpi_32I_ml0p004.cv, 1000.0*fpi_32I_ml0p004.std))

# Get O1 in physical units
# O1_16I            = superjack.distribution(ensembles)
O1_24I_ml0p01     = superjack.distribution(ensembles)
O1_24I_ml0p005    = superjack.distribution(ensembles)
O1_32I_ml0p008    = superjack.distribution(ensembles)
O1_32I_ml0p006    = superjack.distribution(ensembles)
O1_32I_ml0p004    = superjack.distribution(ensembles)
# O1_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/O_1.dat" )
O1_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/O_1.dat" )
O1_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_1.dat" )
O1_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_1.dat" )
O1_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_1.dat" )
O1_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_1.dat" )
# O1_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/O_1_jacks.dat" ) )
O1_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/O_1_jacks.dat" ) )
O1_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_1_jacks.dat" ) )
O1_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_1_jacks.dat" ) )
O1_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_1_jacks.dat" ) )
O1_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_1_jacks.dat" ) )
# O1_16I.set_zeros_to_cv()
O1_24I_ml0p01.set_zeros_to_cv()
O1_24I_ml0p005.set_zeros_to_cv()
O1_32I_ml0p008.set_zeros_to_cv()
O1_32I_ml0p006.set_zeros_to_cv()
O1_32I_ml0p004.set_zeros_to_cv()
# O1_16I.calc_mean()
O1_24I_ml0p01.calc_mean()
O1_24I_ml0p005.calc_mean()
O1_32I_ml0p008.calc_mean()
O1_32I_ml0p006.calc_mean()
O1_32I_ml0p004.calc_mean()
# O1_16I.calc_std()
O1_24I_ml0p01.calc_std()
O1_24I_ml0p005.calc_std()
O1_32I_ml0p008.calc_std()
O1_32I_ml0p006.calc_std()
O1_32I_ml0p004.calc_std()
# O1_16I         = ainv_16I**4 * O1_16I
O1_24I_ml0p01  = ainv_24I**4 * O1_24I_ml0p01
O1_24I_ml0p005 = ainv_24I**4 * O1_24I_ml0p005
O1_32I_ml0p008 = ainv_32I**4 * O1_32I_ml0p008
O1_32I_ml0p006 = ainv_32I**4 * O1_32I_ml0p006
O1_32I_ml0p004 = ainv_32I**4 * O1_32I_ml0p004
# O1_16I.rescale(0.0625)
O1_24I_ml0p01.rescale(0.0625)
O1_24I_ml0p005.rescale(0.0625)
O1_32I_ml0p008.rescale(0.0625)
O1_32I_ml0p006.rescale(0.0625)
O1_32I_ml0p004.rescale(0.0625)
print("\nGot O1:")
# print("\t-- 16I:             O1 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1_16I.cv        , O1_16I.std        ))
print("\t-- 24I, ml = 0.01:  O1 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1_24I_ml0p01.cv , O1_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: O1 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1_24I_ml0p005.cv, O1_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: O1 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1_32I_ml0p008.cv, O1_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: O1 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1_32I_ml0p006.cv, O1_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: O1 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1_32I_ml0p004.cv, O1_32I_ml0p004.std))

# Get O2 in physical units
# O2_16I            = superjack.distribution(ensembles)
O2_24I_ml0p01     = superjack.distribution(ensembles)
O2_24I_ml0p005    = superjack.distribution(ensembles)
O2_32I_ml0p008    = superjack.distribution(ensembles)
O2_32I_ml0p006    = superjack.distribution(ensembles)
O2_32I_ml0p004    = superjack.distribution(ensembles)
# O2_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/O_2.dat" )
O2_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/O_2.dat" )
O2_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_2.dat" )
O2_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_2.dat" )
O2_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_2.dat" )
O2_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_2.dat" )
# O2_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/O_2_jacks.dat" ) )
O2_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/O_2_jacks.dat" ) )
O2_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_2_jacks.dat" ) )
O2_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_2_jacks.dat" ) )
O2_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_2_jacks.dat" ) )
O2_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_2_jacks.dat" ) )
# O2_16I.set_zeros_to_cv()
O2_24I_ml0p01.set_zeros_to_cv()
O2_24I_ml0p005.set_zeros_to_cv()
O2_32I_ml0p008.set_zeros_to_cv()
O2_32I_ml0p006.set_zeros_to_cv()
O2_32I_ml0p004.set_zeros_to_cv()
# O2_16I.calc_mean()
O2_24I_ml0p01.calc_mean()
O2_24I_ml0p005.calc_mean()
O2_32I_ml0p008.calc_mean()
O2_32I_ml0p006.calc_mean()
O2_32I_ml0p004.calc_mean()
# O2_16I.calc_std()
O2_24I_ml0p01.calc_std()
O2_24I_ml0p005.calc_std()
O2_32I_ml0p008.calc_std()
O2_32I_ml0p006.calc_std()
O2_32I_ml0p004.calc_std()
# O2_16I         = ainv_16I**4 * O2_16I
O2_24I_ml0p01  = ainv_24I**4 * O2_24I_ml0p01
O2_24I_ml0p005 = ainv_24I**4 * O2_24I_ml0p005
O2_32I_ml0p008 = ainv_32I**4 * O2_32I_ml0p008
O2_32I_ml0p006 = ainv_32I**4 * O2_32I_ml0p006
O2_32I_ml0p004 = ainv_32I**4 * O2_32I_ml0p004
# O2_16I.rescale(0.125)
O2_24I_ml0p01.rescale(0.125)
O2_24I_ml0p005.rescale(0.125)
O2_32I_ml0p008.rescale(0.125)
O2_32I_ml0p006.rescale(0.125)
O2_32I_ml0p004.rescale(0.125)
print("\nGot O2:")
# print("\t-- 16I:             O2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2_16I.cv        , O2_16I.std        ))
print("\t-- 24I, ml = 0.01:  O2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2_24I_ml0p01.cv , O2_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: O2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2_24I_ml0p005.cv, O2_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: O2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2_32I_ml0p008.cv, O2_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: O2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2_32I_ml0p006.cv, O2_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: O2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2_32I_ml0p004.cv, O2_32I_ml0p004.std))

print(O2_24I_ml0p01.cv)
print(O2_24I_ml0p01.mean)

# Get O3 in physical units
# O3_16I            = superjack.distribution(ensembles)
O3_24I_ml0p01     = superjack.distribution(ensembles)
O3_24I_ml0p005    = superjack.distribution(ensembles)
O3_32I_ml0p008    = superjack.distribution(ensembles)
O3_32I_ml0p006    = superjack.distribution(ensembles)
O3_32I_ml0p004    = superjack.distribution(ensembles)
# O3_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/O_3.dat" )
O3_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/O_3.dat" )
O3_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_3.dat" )
O3_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_3.dat" )
O3_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_3.dat" )
O3_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_3.dat" )
# O3_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/O_3_jacks.dat" ) )
O3_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/O_3_jacks.dat" ) )
O3_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_3_jacks.dat" ) )
O3_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_3_jacks.dat" ) )
O3_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_3_jacks.dat" ) )
O3_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_3_jacks.dat" ) )
# O3_16I.set_zeros_to_cv()
O3_24I_ml0p01.set_zeros_to_cv()
O3_24I_ml0p005.set_zeros_to_cv()
O3_32I_ml0p008.set_zeros_to_cv()
O3_32I_ml0p006.set_zeros_to_cv()
O3_32I_ml0p004.set_zeros_to_cv()
# O3_16I.calc_mean()
O3_24I_ml0p01.calc_mean()
O3_24I_ml0p005.calc_mean()
O3_32I_ml0p008.calc_mean()
O3_32I_ml0p006.calc_mean()
O3_32I_ml0p004.calc_mean()
# O3_16I.calc_std()
O3_24I_ml0p01.calc_std()
O3_24I_ml0p005.calc_std()
O3_32I_ml0p008.calc_std()
O3_32I_ml0p006.calc_std()
O3_32I_ml0p004.calc_std()
# O3_16I         = ainv_16I**4 * O3_16I
O3_24I_ml0p01  = ainv_24I**4 * O3_24I_ml0p01
O3_24I_ml0p005 = ainv_24I**4 * O3_24I_ml0p005
O3_32I_ml0p008 = ainv_32I**4 * O3_32I_ml0p008
O3_32I_ml0p006 = ainv_32I**4 * O3_32I_ml0p006
O3_32I_ml0p004 = ainv_32I**4 * O3_32I_ml0p004
# O3_16I.rescale(0.125)
O3_24I_ml0p01.rescale(0.125)
O3_24I_ml0p005.rescale(0.125)
O3_32I_ml0p008.rescale(0.125)
O3_32I_ml0p006.rescale(0.125)
O3_32I_ml0p004.rescale(0.125)
print("\nGot O3:")
# print("\t-- 16I:             O3 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O3_16I.cv        , O3_16I.std        ))
print("\t-- 24I, ml = 0.01:  O3 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O3_24I_ml0p01.cv , O3_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: O3 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O3_24I_ml0p005.cv, O3_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: O3 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O3_32I_ml0p008.cv, O3_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: O3 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O3_32I_ml0p006.cv, O3_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: O3 = {0:.4e} +/- {1:.4e} (GeV^4)".format(O3_32I_ml0p004.cv, O3_32I_ml0p004.std))

# Get O1p in physical units
# O1p_16I            = superjack.distribution(ensembles)
O1p_24I_ml0p01     = superjack.distribution(ensembles)
O1p_24I_ml0p005    = superjack.distribution(ensembles)
O1p_32I_ml0p008    = superjack.distribution(ensembles)
O1p_32I_ml0p006    = superjack.distribution(ensembles)
O1p_32I_ml0p004    = superjack.distribution(ensembles)
# populate each ENS.cv to the mean value (in the table in his analysis writeup)
# O1p_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/O_1p.dat" )
O1p_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/O_1p.dat" )
O1p_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_1p.dat" )
O1p_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_1p.dat" )
O1p_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_1p.dat" )
O1p_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_1p.dat" )
# O1p_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/O_1p_jacks.dat" ) )
O1p_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/O_1p_jacks.dat" ) )
O1p_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_1p_jacks.dat" ) )
O1p_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_1p_jacks.dat" ) )
O1p_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_1p_jacks.dat" ) )
O1p_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_1p_jacks.dat" ) )
# set every value which has not been set (everything in other ensembles, i.e. in O1p_24I_ml_0p01['32I_ml0p008']) to cv value (which is the ensemble's mean)
# O1p_16I.set_zeros_to_cv()
O1p_24I_ml0p01.set_zeros_to_cv()
O1p_24I_ml0p005.set_zeros_to_cv()
O1p_32I_ml0p008.set_zeros_to_cv()
O1p_32I_ml0p006.set_zeros_to_cv()
O1p_32I_ml0p004.set_zeros_to_cv()
# O1p_16I.calc_mean()
O1p_24I_ml0p01.calc_mean()
O1p_24I_ml0p005.calc_mean()
O1p_32I_ml0p008.calc_mean()
O1p_32I_ml0p006.calc_mean()
O1p_32I_ml0p004.calc_mean()
# O1p_16I.calc_std()
O1p_24I_ml0p01.calc_std()
O1p_24I_ml0p005.calc_std()
O1p_32I_ml0p008.calc_std()
O1p_32I_ml0p006.calc_std()
O1p_32I_ml0p004.calc_std()
# O1p_16I         = ainv_16I**4 * O1p_16I
O1p_24I_ml0p01  = ainv_24I**4 * O1p_24I_ml0p01
O1p_24I_ml0p005 = ainv_24I**4 * O1p_24I_ml0p005
O1p_32I_ml0p008 = ainv_32I**4 * O1p_32I_ml0p008
O1p_32I_ml0p006 = ainv_32I**4 * O1p_32I_ml0p006
O1p_32I_ml0p004 = ainv_32I**4 * O1p_32I_ml0p004
# O1p_16I.rescale(0.0625)
O1p_24I_ml0p01.rescale(0.0625)
O1p_24I_ml0p005.rescale(0.0625)
O1p_32I_ml0p008.rescale(0.0625)
O1p_32I_ml0p006.rescale(0.0625)
O1p_32I_ml0p004.rescale(0.0625)
print("\nGot O1p:")
# print("\t-- 16I:             O1p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1p_16I.cv        , O1p_16I.std        ))
print("\t-- 24I, ml = 0.01:  O1p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1p_24I_ml0p01.cv , O1p_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: O1p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1p_24I_ml0p005.cv, O1p_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: O1p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1p_32I_ml0p008.cv, O1p_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: O1p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1p_32I_ml0p006.cv, O1p_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: O1p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O1p_32I_ml0p004.cv, O1p_32I_ml0p004.std))

# Get O2p in physical units
# O2p_16I            = superjack.distribution(ensembles)
O2p_24I_ml0p01     = superjack.distribution(ensembles)
O2p_24I_ml0p005    = superjack.distribution(ensembles)
O2p_32I_ml0p008    = superjack.distribution(ensembles)
O2p_32I_ml0p006    = superjack.distribution(ensembles)
O2p_32I_ml0p004    = superjack.distribution(ensembles)
# O2p_16I.cv         = np.genfromtxt( data_paths["16I"] + "/fits/results/fit_params/O_2p.dat" )
O2p_24I_ml0p01.cv  = np.genfromtxt( data_paths["24I_ml0p01"] + "/fits/results/fit_params/O_2p.dat" )
O2p_24I_ml0p005.cv = np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_2p.dat" )
O2p_32I_ml0p008.cv = np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_2p.dat" )
O2p_32I_ml0p006.cv = np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_2p.dat" )
O2p_32I_ml0p004.cv = np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_2p.dat" )
# O2p_16I.set_jacks(        "16I"        , np.genfromtxt( data_paths["16I"]         + "/fits/results/fit_params/O_2p_jacks.dat" ) )
O2p_24I_ml0p01.set_jacks( "24I_ml0p01" , np.genfromtxt( data_paths["24I_ml0p01"]  + "/fits/results/fit_params/O_2p_jacks.dat" ) )
O2p_24I_ml0p005.set_jacks("24I_ml0p005", np.genfromtxt( data_paths["24I_ml0p005"] + "/fits/results/fit_params/O_2p_jacks.dat" ) )
O2p_32I_ml0p008.set_jacks("32I_ml0p008", np.genfromtxt( data_paths["32I_ml0p008"] + "/fits/results/fit_params/O_2p_jacks.dat" ) )
O2p_32I_ml0p006.set_jacks("32I_ml0p006", np.genfromtxt( data_paths["32I_ml0p006"] + "/fits/results/fit_params/O_2p_jacks.dat" ) )
O2p_32I_ml0p004.set_jacks("32I_ml0p004", np.genfromtxt( data_paths["32I_ml0p004"] + "/fits/results/fit_params/O_2p_jacks.dat" ) )
# O2p_16I.set_zeros_to_cv()
O2p_24I_ml0p01.set_zeros_to_cv()
O2p_24I_ml0p005.set_zeros_to_cv()
O2p_32I_ml0p008.set_zeros_to_cv()
O2p_32I_ml0p006.set_zeros_to_cv()
O2p_32I_ml0p004.set_zeros_to_cv()
# O2p_16I.calc_mean()
O2p_24I_ml0p01.calc_mean()
O2p_24I_ml0p005.calc_mean()
O2p_32I_ml0p008.calc_mean()
O2p_32I_ml0p006.calc_mean()
O2p_32I_ml0p004.calc_mean()
# O2p_16I.calc_std()
O2p_24I_ml0p01.calc_std()
O2p_24I_ml0p005.calc_std()
O2p_32I_ml0p008.calc_std()
O2p_32I_ml0p006.calc_std()
O2p_32I_ml0p004.calc_std()
# O2p_16I         = ainv_16I**4 * O2p_16I
O2p_24I_ml0p01  = ainv_24I**4 * O2p_24I_ml0p01
O2p_24I_ml0p005 = ainv_24I**4 * O2p_24I_ml0p005
O2p_32I_ml0p008 = ainv_32I**4 * O2p_32I_ml0p008
O2p_32I_ml0p006 = ainv_32I**4 * O2p_32I_ml0p006
O2p_32I_ml0p004 = ainv_32I**4 * O2p_32I_ml0p004
# O2p_16I.rescale(0.125)
O2p_24I_ml0p01.rescale(0.125)
O2p_24I_ml0p005.rescale(0.125)
O2p_32I_ml0p008.rescale(0.125)
O2p_32I_ml0p006.rescale(0.125)
O2p_32I_ml0p004.rescale(0.125)
print("\nGot O2p:")
# print("\t-- 16I:             O2p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2p_16I.cv        , O2p_16I.std        ))
print("\t-- 24I, ml = 0.01:  O2p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2p_24I_ml0p01.cv , O2p_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: O2p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2p_24I_ml0p005.cv, O2p_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: O2p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2p_32I_ml0p008.cv, O2p_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: O2p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2p_32I_ml0p006.cv, O2p_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: O2p = {0:.4e} +/- {1:.4e} (GeV^4)".format(O2p_32I_ml0p004.cv, O2p_32I_ml0p004.std))

# Compute eps_pi
# epi2_16I         = mpi_16I**2         / fpi_16I**2
epi2_24I_ml0p01  = mpi_24I_ml0p01**2  / fpi_24I_ml0p01**2
epi2_24I_ml0p005 = mpi_24I_ml0p005**2 / fpi_24I_ml0p005**2
epi2_32I_ml0p008 = mpi_32I_ml0p008**2 / fpi_32I_ml0p008**2
epi2_32I_ml0p006 = mpi_32I_ml0p006**2 / fpi_32I_ml0p006**2
epi2_32I_ml0p004 = mpi_32I_ml0p004**2 / fpi_32I_ml0p004**2
# epi2_16I.rescale(0.125/np.pi**2)
epi2_24I_ml0p01.rescale(0.125/np.pi**2)
epi2_24I_ml0p005.rescale(0.125/np.pi**2)
epi2_32I_ml0p008.rescale(0.125/np.pi**2)
epi2_32I_ml0p006.rescale(0.125/np.pi**2)
epi2_32I_ml0p004.rescale(0.125/np.pi**2)
print("\nGot epi**2:")
# print("\t-- 16I:             epi**2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(epi2_16I.cv        , epi2_16I.std        ))
print("\t-- 24I, ml = 0.01:  epi**2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(epi2_24I_ml0p01.cv , epi2_24I_ml0p01.std ))
print("\t-- 24I, ml = 0.005: epi**2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(epi2_24I_ml0p005.cv, epi2_24I_ml0p005.std))
print("\t-- 32I, ml = 0.008: epi**2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(epi2_32I_ml0p008.cv, epi2_32I_ml0p008.std))
print("\t-- 32I, ml = 0.006: epi**2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(epi2_32I_ml0p006.cv, epi2_32I_ml0p006.std))
print("\t-- 32I, ml = 0.004: epi**2 = {0:.4e} +/- {1:.4e} (GeV^4)".format(epi2_32I_ml0p004.cv, epi2_32I_ml0p004.std))

# exit()

#####################################################################################################
# Fit O1

if fit_O1:

  print("\n===== Fitting to O1 =====")
  xx                               = np.linspace(0.0, 0.5, 101)
  epi2_xx                          = xx**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  y1                               = np.zeros((len(xx)-1))
  dy1                              = np.zeros((len(xx)-1))
  b1_jacks                         = superjack.distribution(ensembles)
  c1_jacks                         = superjack.distribution(ensembles)
  c1a_jacks                        = superjack.distribution(ensembles)
  chi2_O1_jacks                    = superjack.distribution(ensembles)
  chi2perdof_O1_jacks              = superjack.distribution(ensembles)
  O1_phys_jacks                    = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p004_jacks = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p006_jacks = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p008_jacks = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p005_jacks = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p01_jacks  = superjack.distribution(ensembles)
  O1_corrected_32I_ml0p004_jacks   = superjack.distribution(ensembles)
  O1_corrected_32I_ml0p006_jacks   = superjack.distribution(ensembles)
  O1_corrected_32I_ml0p008_jacks   = superjack.distribution(ensembles)
  O1_corrected_24I_ml0p005_jacks   = superjack.distribution(ensembles)
  O1_corrected_24I_ml0p01_jacks    = superjack.distribution(ensembles)
  y1_jacks            = []
  for idx in range(0, len(xx)-1):
    y1_jacks.append(superjack.distribution(ensembles))

  # Fit to central values
  print("\t-- Fitting to central value...")
  this_mpi  = np.array( [ mpi_32I_ml0p004.cv, mpi_24I_ml0p005.cv, mpi_32I_ml0p006.cv, mpi_32I_ml0p008.cv, mpi_24I_ml0p01.cv ] )
  this_fpi  = np.array( [ fpi_32I_ml0p004.cv, fpi_24I_ml0p005.cv, fpi_32I_ml0p006.cv, fpi_32I_ml0p008.cv, fpi_24I_ml0p01.cv ] )
  this_a2   = np.array( [ a2_32I.cv, a2_24I.cv, a2_32I.cv, a2_32I.cv, a2_24I.cv ] )
  # TODO THIS LOOKS LIKE A BUG: USING L = 32 IN THE CODE FOR F1
  this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p005.cv/ainv_24I.cv, 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.cv/ainv_24I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p01.cv/ainv_24I.cv , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.cv/ainv_24I.cv, 32.0) ] )
  this_fv = this_f0
  this_O1   = np.array( [ O1_32I_ml0p004.cv, O1_24I_ml0p005.cv, O1_32I_ml0p006.cv, O1_32I_ml0p008.cv, O1_24I_ml0p01.cv ] )
  this_dO1  = np.array( [ O1_32I_ml0p004.std, O1_24I_ml0p005.std, O1_32I_ml0p006.std, O1_32I_ml0p008.std, O1_24I_ml0p01.std ] )
  # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O1, this_dO1), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})

  # SOME TESTING I WAS PLAYING WITH
  # print('mpi')
  # print(this_mpi)
  # print('fpi')
  # print(this_fpi)
  # print('a_sq')
  # print(this_a2)
  # print('fv')
  # print(this_fv)
  # print('O1')
  # print(this_O1)
  # print('dO1')
  # print(this_dO1)
  # print('chi2 value at minimum:')
  # print(SD_chpt.chi2_O12_chptfv_a2(np.array([2, 3, 1]), np.array([4, 4, 4, 4, 4]), np.array([5, 5, 5, 5, 5]), np.array([6, 6, 6, 6, 6]), np.array([7, 7, 7, 7, 7]), np.array([8, 8, 8, 8, 8]), np.array([9, 9, 9, 9, 9])))

  # x0tmp = [-1.7,-0.8,-1.3]
  # mtmp = np.array([0.30137702, 0.3413693,  0.3591327,  0.41153151, 0.43111663])
  # ftmp = np.array([0.14754673, 0.15159201, 0.15429037, 0.16202998, 0.16375955])
  # asqtmp = np.array([0.00686261, 0.01222893, 0.00686261, 0.00686261, 0.01222893])
  # fvtmp = np.array([0.15470272, 0.04469975, 0.05700776, 0.02402461, 0.01071661])
  # O1tmp = np.array([-0.02364856, -0.02473052, -0.02626278, -0.02969915, -0.02944992])
  # dO1tmp = np.array([0.00036667, 0.00030666, 0.00040225, 0.00043925, 0.00037243])
  # print(SD_chpt.chi2_O12_chptfv_a2(x0tmp, mtmp, ftmp, asqtmp, fvtmp, O1tmp, dO1tmp))
  #
  # mtmp2 = np.array([0.4310717, 0.34125885, 0.41164456, 0.35922994, 0.30164958])
  # ftmp2 = np.array([0.16374257, 0.15154161, 0.16206748, 0.15432774, 0.14767583])
  # asqtmp2 = np.array([0.01223167, 0.01223704, 0.00685934, 0.00685911, 0.00685041])
  # fvtmp2 = np.array([0.01071635, 0.04469897, 0.02402092, 0.05700604, 0.15469871])
  # O1tmp2 = np.array([-0.02944849, -0.02476874, -0.02962412, -0.02634128, -0.02371357])
  # dO1tmp2 = np.array([0.0003789, 0.00029522, 0.00044951, 0.00041236, 0.00032347])
  # print(SD_chpt.chi2_O12_chptfv_a2(x0tmp, mtmp2, ftmp2, asqtmp2, fvtmp2, O1tmp2, dO1tmp2))

  # fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [-1.7,-0.8,-1.3], args=(this_mpi, this_fpi, this_a2, this_fv, this_O1, this_dO1), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [1.0, 1.0, 1.0], args=(this_mpi, this_fpi, this_a2, this_fv, this_O1, this_dO1), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  b1_jacks.cv              = fit.x[0]
  c1_jacks.cv              = fit.x[1]
  c1a_jacks.cv             = fit.x[2]
  print(fit.x[2], fit.x[0], fit.x[1])
  # print(SD_chpt.f0(7, 32))
  # print(SD_chpt.f1(3, 16))
  # print(SD_chpt.O12_chptfv_a2(2, 3, 1, 4, 5, 7, -SD_chpt.f0(4, 6) + 2 * SD_chpt.f1(4, 6)))
  chi2_O1_jacks.cv         = fit.fun
  chi2perdof_O1_jacks.cv   = chi2_O1_jacks.cv / ( 5.0 - 3.0 )
  epi2_corrected_32I_ml0p004_jacks.cv = mpi_32I_ml0p004.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p006_jacks.cv = mpi_32I_ml0p006.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p008_jacks.cv = mpi_32I_ml0p008.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p005_jacks.cv = mpi_24I_ml0p005.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p01_jacks.cv  = mpi_24I_ml0p01.cv**2  / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  O1_corrected_32I_ml0p004_jacks.cv = this_O1[0] - ( SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[0], fpi_phys.cv, 0.0, 0.0) )
  O1_corrected_24I_ml0p005_jacks.cv = this_O1[1] - ( SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[1], fpi_phys.cv, 0.0, 0.0) )
  O1_corrected_32I_ml0p006_jacks.cv = this_O1[2] - ( SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[2], fpi_phys.cv, 0.0, 0.0) )
  O1_corrected_32I_ml0p008_jacks.cv = this_O1[3] - ( SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[3], fpi_phys.cv, 0.0, 0.0) )
  O1_corrected_24I_ml0p01_jacks.cv  = this_O1[4] - ( SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, this_mpi[4], fpi_phys.cv, 0.0, 0.0) )
  O1_phys_jacks.cv = SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, mpi_phys.cv, fpi_phys.cv, 0.0, 0.0)
  for idx in range(0, len(xx)-1):
    y1_jacks[idx].cv = SD_chpt.O12_chptfv_a2(b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv, xx[idx+1], fpi_phys.cv, 0.0, 0.0)
    y1[idx] = y1_jacks[idx].cv

  # Fit to superjackknife samples
  this_boot = 0
  for ens in ensembles:     # ens is '16I', '24I_ml0p01', ... ensembles is a dict with label -> n_cfgs
    Nboots = ensembles[ens]
    for idx in range(0, Nboots):
      print("\t-- Fitting to boot {0} of {1}...".format(this_boot+1, Nsuperboots))
      # When ens is not the same distribution, mpi_ENS1.data[ENS2][idx] returns mpi_ENS1.cv, which is the ensemble average for mpi on ENSEMBLE 1
      this_mpi  = np.array( [ mpi_32I_ml0p004.data[ens][idx], mpi_24I_ml0p005.data[ens][idx], mpi_32I_ml0p006.data[ens][idx], mpi_32I_ml0p008.data[ens][idx], mpi_24I_ml0p01.data[ens][idx] ] )
      this_fpi  = np.array( [ fpi_32I_ml0p004.data[ens][idx], fpi_24I_ml0p005.data[ens][idx], fpi_32I_ml0p006.data[ens][idx], fpi_32I_ml0p008.data[ens][idx], fpi_24I_ml0p01.data[ens][idx] ] )
      this_a2   = np.array( [ a2_32I.data[ens][idx], a2_24I.data[ens][idx], a2_32I.data[ens][idx], a2_32I.data[ens][idx], a2_24I.data[ens][idx] ] )
      this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx] , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx], 32.0) ] )
      this_fv   = this_f0
      this_O1   = np.array( [ O1_32I_ml0p004.data[ens][idx], O1_24I_ml0p005.data[ens][idx], O1_32I_ml0p006.data[ens][idx], O1_32I_ml0p008.data[ens][idx], O1_24I_ml0p01.data[ens][idx] ] )
      this_dO1  = np.array( [ O1_32I_ml0p004.std, O1_24I_ml0p005.std, O1_32I_ml0p006.std, O1_32I_ml0p008.std, O1_24I_ml0p01.std ] )
      # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O1, this_dO1), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [b1_jacks.cv, c1_jacks.cv, c1a_jacks.cv], args=(this_mpi, this_fpi, this_a2, this_fv, this_O1, this_dO1), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      b1_jacks.data[ens][idx]  = fit.x[0]
      c1_jacks.data[ens][idx]  = fit.x[1]
      c1a_jacks.data[ens][idx] = fit.x[2]
      chi2_O1_jacks.data[ens][idx] = fit.fun
      chi2perdof_O1_jacks.data[ens][idx] = chi2_O1_jacks.data[ens][idx] / ( 5.0 - 3.0 )
      O1_phys_jacks.data[ens][idx] = SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], mpi_phys.data[ens][idx], fpi_phys.data[ens][idx], 0.0, 0.0)
      epi2_corrected_32I_ml0p004_jacks.data[ens][idx] = mpi_32I_ml0p004.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p006_jacks.data[ens][idx] = mpi_32I_ml0p006.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p008_jacks.data[ens][idx] = mpi_32I_ml0p008.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p005_jacks.data[ens][idx] = mpi_24I_ml0p005.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p01_jacks.data[ens][idx]  = mpi_24I_ml0p01.data[ens][idx]**2  / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      O1_corrected_32I_ml0p004_jacks.data[ens][idx] = this_O1[0] - ( SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[0], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1_corrected_24I_ml0p005_jacks.data[ens][idx] = this_O1[1] - ( SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[1], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1_corrected_32I_ml0p006_jacks.data[ens][idx] = this_O1[2] - ( SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[2], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1_corrected_32I_ml0p008_jacks.data[ens][idx] = this_O1[3] - ( SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[3], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1_corrected_24I_ml0p01_jacks.data[ens][idx]  = this_O1[4] - ( SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], this_mpi[4], fpi_phys.data[ens][idx], 0.0, 0.0) )
      for ii in range(0, len(xx)-1):
        y1_jacks[ii].data[ens][idx] = SD_chpt.O12_chptfv_a2(b1_jacks.data[ens][idx], c1_jacks.data[ens][idx], c1a_jacks.data[ens][idx], xx[ii+1], fpi_phys.data[ens][idx], 0.0, 0.0)
      this_boot += 1

      # O1_corrected... doesn't extrapolate the pion mass (probably so we can plot against epsilon_corrected), and O1_phys does extrapolate to the physical pion mass

  # Compute errors
  b1_jacks.calc_mean()
  b1_jacks.calc_std()
  c1_jacks.calc_mean()
  c1_jacks.calc_std()
  c1a_jacks.calc_mean()
  c1a_jacks.calc_std()
  chi2_O1_jacks.calc_mean()
  chi2_O1_jacks.calc_std()
  chi2perdof_O1_jacks.calc_mean()
  chi2perdof_O1_jacks.calc_std()
  O1_phys_jacks.calc_mean()
  O1_phys_jacks.calc_std()
  epi2_corrected_32I_ml0p004_jacks.calc_mean()
  epi2_corrected_32I_ml0p004_jacks.calc_std()
  epi2_corrected_32I_ml0p006_jacks.calc_mean()
  epi2_corrected_32I_ml0p006_jacks.calc_std()
  epi2_corrected_32I_ml0p008_jacks.calc_mean()
  epi2_corrected_32I_ml0p008_jacks.calc_std()
  epi2_corrected_24I_ml0p005_jacks.calc_mean()
  epi2_corrected_24I_ml0p005_jacks.calc_std()
  epi2_corrected_24I_ml0p01_jacks.calc_mean()
  epi2_corrected_24I_ml0p01_jacks.calc_std()
  O1_corrected_32I_ml0p004_jacks.calc_mean()
  O1_corrected_32I_ml0p004_jacks.calc_std()
  O1_corrected_32I_ml0p006_jacks.calc_mean()
  O1_corrected_32I_ml0p006_jacks.calc_std()
  O1_corrected_32I_ml0p008_jacks.calc_mean()
  O1_corrected_32I_ml0p008_jacks.calc_std()
  O1_corrected_24I_ml0p005_jacks.calc_mean()
  O1_corrected_24I_ml0p005_jacks.calc_std()
  O1_corrected_24I_ml0p01_jacks.calc_mean()
  O1_corrected_24I_ml0p01_jacks.calc_std()
  for ii in range(0, len(xx)-1):
    y1_jacks[ii].calc_mean()
    y1_jacks[ii].calc_std()
    dy1[ii] = y1_jacks[ii].std

  print("\t-- Result: O1 = {0:.8e} +/- {1:.8e}".format(O1_phys_jacks.cv, O1_phys_jacks.std))
  print("\t-- Result: b1 = {0:.8e} +/- {1:.8e}".format(b1_jacks.cv, b1_jacks.std))
  print("\t-- Result: c1 = {0:.8e} +/- {1:.8e}".format(c1_jacks.cv, c1_jacks.std))
  print("\t-- Result: c1a = {0:.8e} +/- {1:.8e}".format(c1a_jacks.cv, c1a_jacks.std))
  print("\t-- Result: chi2pdof = {0:.8e} +/- {1:.8e}".format(chi2perdof_O1_jacks.cv, chi2perdof_O1_jacks.std))

  with open("./results/O1_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p004.cv, epi2_32I_ml0p004.std, O1_32I_ml0p004.cv, O1_32I_ml0p004.std), file=f)

  with open("./results/O1_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p006.cv, epi2_32I_ml0p006.std, O1_32I_ml0p006.cv, O1_32I_ml0p006.std), file=f)

  with open("./results/O1_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p008.cv, epi2_32I_ml0p008.std, O1_32I_ml0p008.cv, O1_32I_ml0p008.std), file=f)

  with open("./results/O1_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p005.cv, epi2_24I_ml0p005.std, O1_24I_ml0p005.cv, O1_24I_ml0p005.std), file=f)

  with open("./results/O1_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p01.cv, epi2_24I_ml0p01.std, O1_24I_ml0p01.cv, O1_24I_ml0p01.std), file=f)

  with open("./results/O1_corrected_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p004_jacks.cv, epi2_corrected_32I_ml0p004_jacks.std, O1_corrected_32I_ml0p004_jacks.cv, O1_corrected_32I_ml0p004_jacks.std), file=f)

  with open("./results/O1_corrected_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p006_jacks.cv, epi2_corrected_32I_ml0p006_jacks.std, O1_corrected_32I_ml0p006_jacks.cv, O1_corrected_32I_ml0p006_jacks.std), file=f)

  with open("./results/O1_corrected_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p008_jacks.cv, epi2_corrected_32I_ml0p008_jacks.std, O1_corrected_32I_ml0p008_jacks.cv, O1_corrected_32I_ml0p008_jacks.std), file=f)

  with open("./results/O1_corrected_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p005_jacks.cv, epi2_corrected_24I_ml0p005_jacks.std, O1_corrected_24I_ml0p005_jacks.cv, O1_corrected_24I_ml0p005_jacks.std), file=f)

  with open("./results/O1_corrected_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p01_jacks.cv, epi2_corrected_24I_ml0p01_jacks.std, O1_corrected_24I_ml0p01_jacks.cv, O1_corrected_24I_ml0p01_jacks.std), file=f)

  with open("./results/O1_phys.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_phys.cv, epi2_phys.std, O1_phys_jacks.cv, O1_phys_jacks.std), file=f)

  with open("./results/O1_extrap.dat", "w") as f:
    for idx in range(0, len(xx)-1):
      print("{0:.8e} {1:.8e} {2:.8e}".format(epi2_xx[idx+1], y1[idx], dy1[idx]), file=f)

#####################################################################################################
# Fit O2

if fit_O2:

  print("\n===== Fitting to O2 =====")
  xx                               = np.linspace(0.0, 0.5, 101)
  epi2_xx                          = xx**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  y2                               = np.zeros((len(xx)-1))
  dy2                              = np.zeros((len(xx)-1))
  b2_jacks                         = superjack.distribution(ensembles)
  c2_jacks                         = superjack.distribution(ensembles)
  c2a_jacks                        = superjack.distribution(ensembles)
  chi2_O2_jacks                    = superjack.distribution(ensembles)
  chi2perdof_O2_jacks              = superjack.distribution(ensembles)
  O2_phys_jacks                    = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p004_jacks = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p006_jacks = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p008_jacks = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p005_jacks = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p01_jacks  = superjack.distribution(ensembles)
  O2_corrected_32I_ml0p004_jacks   = superjack.distribution(ensembles)
  O2_corrected_32I_ml0p006_jacks   = superjack.distribution(ensembles)
  O2_corrected_32I_ml0p008_jacks   = superjack.distribution(ensembles)
  O2_corrected_24I_ml0p005_jacks   = superjack.distribution(ensembles)
  O2_corrected_24I_ml0p01_jacks    = superjack.distribution(ensembles)
  y2_jacks            = []
  for idx in range(0, len(xx)-1):
    y2_jacks.append(superjack.distribution(ensembles))

  # Fit to central values
  print("\t-- Fitting to central value...")
  this_mpi  = np.array( [ mpi_32I_ml0p004.cv, mpi_24I_ml0p005.cv, mpi_32I_ml0p006.cv, mpi_32I_ml0p008.cv, mpi_24I_ml0p01.cv ] )
  this_fpi  = np.array( [ fpi_32I_ml0p004.cv, fpi_24I_ml0p005.cv, fpi_32I_ml0p006.cv, fpi_32I_ml0p008.cv, fpi_24I_ml0p01.cv ] )
  this_a2   = np.array( [ a2_32I.cv, a2_24I.cv, a2_32I.cv, a2_32I.cv, a2_24I.cv ] )
  this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p005.cv/ainv_24I.cv, 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.cv/ainv_24I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p01.cv/ainv_24I.cv , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.cv/ainv_24I.cv, 32.0) ] )
  this_fv = this_f0
  this_O2   = np.array( [ O2_32I_ml0p004.cv, O2_24I_ml0p005.cv, O2_32I_ml0p006.cv, O2_32I_ml0p008.cv, O2_24I_ml0p01.cv ] )
  this_dO2  = np.array( [ O2_32I_ml0p004.std, O2_24I_ml0p005.std, O2_32I_ml0p006.std, O2_32I_ml0p008.std, O2_24I_ml0p01.std ] )
  # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O2, this_dO2), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [-5.0,-1.1,10.3], args=(this_mpi, this_fpi, this_a2, this_fv, this_O2, this_dO2), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  print(fit.x[2], fit.x[0], fit.x[1])
  b2_jacks.cv              = fit.x[0]
  c2_jacks.cv              = fit.x[1]
  c2a_jacks.cv             = fit.x[2]
  chi2_O2_jacks.cv         = fit.fun
  chi2perdof_O2_jacks.cv   = chi2_O2_jacks.cv / ( 5.0 - 3.0 )
  epi2_corrected_32I_ml0p004_jacks.cv = mpi_32I_ml0p004.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p006_jacks.cv = mpi_32I_ml0p006.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p008_jacks.cv = mpi_32I_ml0p008.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p005_jacks.cv = mpi_24I_ml0p005.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p01_jacks.cv  = mpi_24I_ml0p01.cv**2  / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  O2_corrected_32I_ml0p004_jacks.cv = this_O2[0] - ( SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[0], fpi_phys.cv, 0.0, 0.0) )
  O2_corrected_24I_ml0p005_jacks.cv = this_O2[1] - ( SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[1], fpi_phys.cv, 0.0, 0.0) )
  O2_corrected_32I_ml0p006_jacks.cv = this_O2[2] - ( SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[2], fpi_phys.cv, 0.0, 0.0) )
  O2_corrected_32I_ml0p008_jacks.cv = this_O2[3] - ( SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[3], fpi_phys.cv, 0.0, 0.0) )
  O2_corrected_24I_ml0p01_jacks.cv  = this_O2[4] - ( SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, this_mpi[4], fpi_phys.cv, 0.0, 0.0) )
  O2_phys_jacks.cv = SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, mpi_phys.cv, fpi_phys.cv, 0.0, 0.0)
  for idx in range(0, len(xx)-1):
    y2_jacks[idx].cv = SD_chpt.O12_chptfv_a2(b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv, xx[idx+1], fpi_phys.cv, 0.0, 0.0)
    y2[idx] = y2_jacks[idx].cv

  # Fit to superjackknife samples
  this_boot = 0
  for ens in ensembles:
    Nboots = ensembles[ens]
    for idx in range(0, Nboots):
      print("\t-- Fitting to boot {0} of {1}...".format(this_boot+1, Nsuperboots))
      this_mpi  = np.array( [ mpi_32I_ml0p004.data[ens][idx], mpi_24I_ml0p005.data[ens][idx], mpi_32I_ml0p006.data[ens][idx], mpi_32I_ml0p008.data[ens][idx], mpi_24I_ml0p01.data[ens][idx] ] )
      this_fpi  = np.array( [ fpi_32I_ml0p004.data[ens][idx], fpi_24I_ml0p005.data[ens][idx], fpi_32I_ml0p006.data[ens][idx], fpi_32I_ml0p008.data[ens][idx], fpi_24I_ml0p01.data[ens][idx] ] )
      this_a2   = np.array( [ a2_32I.data[ens][idx], a2_24I.data[ens][idx], a2_32I.data[ens][idx], a2_32I.data[ens][idx], a2_24I.data[ens][idx] ] )
      this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx] , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx], 32.0) ] )
      this_fv   = this_f0
      this_O2   = np.array( [ O2_32I_ml0p004.data[ens][idx], O2_24I_ml0p005.data[ens][idx], O2_32I_ml0p006.data[ens][idx], O2_32I_ml0p008.data[ens][idx], O2_24I_ml0p01.data[ens][idx] ] )
      this_dO2  = np.array( [ O2_32I_ml0p004.std, O2_24I_ml0p005.std, O2_32I_ml0p006.std, O2_32I_ml0p008.std, O2_24I_ml0p01.std ] )
      # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O2, this_dO2), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [b2_jacks.cv, c2_jacks.cv, c2a_jacks.cv], args=(this_mpi, this_fpi, this_a2, this_fv, this_O2, this_dO2), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      b2_jacks.data[ens][idx]  = fit.x[0]
      c2_jacks.data[ens][idx]  = fit.x[1]
      c2a_jacks.data[ens][idx] = fit.x[2]
      chi2_O2_jacks.data[ens][idx] = fit.fun
      chi2perdof_O2_jacks.data[ens][idx] = chi2_O2_jacks.data[ens][idx] / ( 5.0 - 3.0 )
      epi2_corrected_32I_ml0p004_jacks.data[ens][idx] = mpi_32I_ml0p004.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p006_jacks.data[ens][idx] = mpi_32I_ml0p006.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p008_jacks.data[ens][idx] = mpi_32I_ml0p008.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p005_jacks.data[ens][idx] = mpi_24I_ml0p005.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p01_jacks.data[ens][idx]  = mpi_24I_ml0p01.data[ens][idx]**2  / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      O2_phys_jacks.data[ens][idx] = SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], mpi_phys.data[ens][idx], fpi_phys.data[ens][idx], 0.0, 0.0)
      O2_corrected_32I_ml0p004_jacks.data[ens][idx] = this_O2[0] - ( SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[0], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2_corrected_24I_ml0p005_jacks.data[ens][idx] = this_O2[1] - ( SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[1], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2_corrected_32I_ml0p006_jacks.data[ens][idx] = this_O2[2] - ( SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[2], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2_corrected_32I_ml0p008_jacks.data[ens][idx] = this_O2[3] - ( SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[3], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2_corrected_24I_ml0p01_jacks.data[ens][idx]  = this_O2[4] - ( SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], this_mpi[4], fpi_phys.data[ens][idx], 0.0, 0.0) )
      for ii in range(0, len(xx)-1):
        y2_jacks[ii].data[ens][idx] = SD_chpt.O12_chptfv_a2(b2_jacks.data[ens][idx], c2_jacks.data[ens][idx], c2a_jacks.data[ens][idx], xx[ii+1], fpi_phys.data[ens][idx], 0.0, 0.0)
      this_boot += 1

  # Compute errors
  b2_jacks.calc_mean()
  b2_jacks.calc_std()
  c2_jacks.calc_mean()
  c2_jacks.calc_std()
  c2a_jacks.calc_mean()
  c2a_jacks.calc_std()
  chi2_O2_jacks.calc_mean()
  chi2_O2_jacks.calc_std()
  chi2perdof_O2_jacks.calc_mean()
  chi2perdof_O2_jacks.calc_std()
  epi2_corrected_32I_ml0p004_jacks.calc_mean()
  epi2_corrected_32I_ml0p004_jacks.calc_std()
  epi2_corrected_32I_ml0p006_jacks.calc_mean()
  epi2_corrected_32I_ml0p006_jacks.calc_std()
  epi2_corrected_32I_ml0p008_jacks.calc_mean()
  epi2_corrected_32I_ml0p008_jacks.calc_std()
  epi2_corrected_24I_ml0p005_jacks.calc_mean()
  epi2_corrected_24I_ml0p005_jacks.calc_std()
  epi2_corrected_24I_ml0p01_jacks.calc_mean()
  epi2_corrected_24I_ml0p01_jacks.calc_std()
  O2_phys_jacks.calc_mean()
  O2_phys_jacks.calc_std()
  O2_corrected_32I_ml0p004_jacks.calc_mean()
  O2_corrected_32I_ml0p004_jacks.calc_std()
  O2_corrected_32I_ml0p006_jacks.calc_mean()
  O2_corrected_32I_ml0p006_jacks.calc_std()
  O2_corrected_32I_ml0p008_jacks.calc_mean()
  O2_corrected_32I_ml0p008_jacks.calc_std()
  O2_corrected_24I_ml0p005_jacks.calc_mean()
  O2_corrected_24I_ml0p005_jacks.calc_std()
  O2_corrected_24I_ml0p01_jacks.calc_mean()
  O2_corrected_24I_ml0p01_jacks.calc_std()
  for ii in range(0, len(xx)-1):
    y2_jacks[ii].calc_mean()
    y2_jacks[ii].calc_std()
    dy2[ii] = y2_jacks[ii].std

  print("\t-- Result: O2 = {0:.8e} +/- {1:.8e}".format(O2_phys_jacks.cv, O2_phys_jacks.std))
  print("\t-- Result: b2 = {0:.8e} +/- {1:.8e}".format(b2_jacks.cv, b2_jacks.std))
  print("\t-- Result: c2 = {0:.8e} +/- {1:.8e}".format(c2_jacks.cv, c2_jacks.std))
  print("\t-- Result: c2a = {0:.8e} +/- {1:.8e}".format(c2a_jacks.cv, c2a_jacks.std))
  print("\t-- Result: chi2pdof = {0:.8e} +/- {1:.8e}".format(chi2perdof_O2_jacks.cv, chi2perdof_O2_jacks.std))

  with open("./results/O2_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p004.cv, epi2_32I_ml0p004.std, O2_32I_ml0p004.cv, O2_32I_ml0p004.std), file=f)

  with open("./results/O2_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p006.cv, epi2_32I_ml0p006.std, O2_32I_ml0p006.cv, O2_32I_ml0p006.std), file=f)

  with open("./results/O2_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p008.cv, epi2_32I_ml0p008.std, O2_32I_ml0p008.cv, O2_32I_ml0p008.std), file=f)

  with open("./results/O2_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p005.cv, epi2_24I_ml0p005.std, O2_24I_ml0p005.cv, O2_24I_ml0p005.std), file=f)

  with open("./results/O2_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p01.cv, epi2_24I_ml0p01.std, O2_24I_ml0p01.cv, O2_24I_ml0p01.std), file=f)

  with open("./results/O2_corrected_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p004_jacks.cv, epi2_corrected_32I_ml0p004_jacks.std, O2_corrected_32I_ml0p004_jacks.cv, O2_corrected_32I_ml0p004_jacks.std), file=f)

  with open("./results/O2_corrected_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p006_jacks.cv, epi2_corrected_32I_ml0p006_jacks.std, O2_corrected_32I_ml0p006_jacks.cv, O2_corrected_32I_ml0p006_jacks.std), file=f)

  with open("./results/O2_corrected_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p008_jacks.cv, epi2_corrected_32I_ml0p008_jacks.std, O2_corrected_32I_ml0p008_jacks.cv, O2_corrected_32I_ml0p008_jacks.std), file=f)

  with open("./results/O2_corrected_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p005_jacks.cv, epi2_corrected_24I_ml0p005_jacks.std, O2_corrected_24I_ml0p005_jacks.cv, O2_corrected_24I_ml0p005_jacks.std), file=f)

  with open("./results/O2_corrected_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p01_jacks.cv, epi2_corrected_24I_ml0p01_jacks.std, O2_corrected_24I_ml0p01_jacks.cv, O2_corrected_24I_ml0p01_jacks.std), file=f)

  with open("./results/O2_phys.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_phys.cv, epi2_phys.std, O2_phys_jacks.cv, O2_phys_jacks.std), file=f)

  with open("./results/O2_extrap.dat", "w") as f:
    for idx in range(0, len(xx)-1):
      print("{0:.8e} {1:.8e} {2:.8e}".format(epi2_xx[idx+1], y2[idx], dy2[idx]), file=f)

#####################################################################################################
# Fit O3

if fit_O3:

  print("\n===== Fitting to O3 =====")
  xx                               = np.linspace(0.0, 0.5, 101)
  epi2_xx                          = xx**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  y3                               = np.zeros((len(xx)-1))
  dy3                              = np.zeros((len(xx)-1))
  b3_jacks                         = superjack.distribution(ensembles)
  c3_jacks                         = superjack.distribution(ensembles)
  c3a_jacks                        = superjack.distribution(ensembles)
  chi2_O3_jacks                    = superjack.distribution(ensembles)
  chi2perdof_O3_jacks              = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p004_jacks = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p006_jacks = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p008_jacks = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p005_jacks = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p01_jacks  = superjack.distribution(ensembles)
  O3_phys_jacks                    = superjack.distribution(ensembles)
  O3_corrected_32I_ml0p004_jacks   = superjack.distribution(ensembles)
  O3_corrected_32I_ml0p006_jacks   = superjack.distribution(ensembles)
  O3_corrected_32I_ml0p008_jacks   = superjack.distribution(ensembles)
  O3_corrected_24I_ml0p005_jacks   = superjack.distribution(ensembles)
  O3_corrected_24I_ml0p01_jacks    = superjack.distribution(ensembles)
  y3_jacks            = []
  for idx in range(0, len(xx)-1):
    y3_jacks.append(superjack.distribution(ensembles))

  # Fit to central values
  print("\t-- Fitting to central value...")
  this_mpi  = np.array( [ mpi_32I_ml0p004.cv, mpi_24I_ml0p005.cv, mpi_32I_ml0p006.cv, mpi_32I_ml0p008.cv, mpi_24I_ml0p01.cv ] )
  this_fpi  = np.array( [ fpi_32I_ml0p004.cv, fpi_24I_ml0p005.cv, fpi_32I_ml0p006.cv, fpi_32I_ml0p008.cv, fpi_24I_ml0p01.cv ] )
  this_a2   = np.array( [ a2_32I.cv, a2_24I.cv, a2_32I.cv, a2_32I.cv, a2_24I.cv ] )
  this_f0 = np.array( [ SD_chpt.f0(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0), \
                        SD_chpt.f0(mpi_24I_ml0p005.cv/ainv_24I.cv, 24.0), \
                        SD_chpt.f0(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0), \
                        SD_chpt.f0(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0), \
                        SD_chpt.f0(mpi_24I_ml0p01.cv/ainv_24I.cv , 24.0) ] )
  this_f1 = np.array( [ SD_chpt.f1(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0), \
                        SD_chpt.f1(mpi_24I_ml0p005.cv/ainv_24I.cv, 24.0), \
                        SD_chpt.f1(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0), \
                        SD_chpt.f1(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0), \
                        SD_chpt.f1(mpi_24I_ml0p01.cv/ainv_24I.cv , 24.0) ] )
  this_fv = 2.0*this_f1 + this_f0
  this_O3   = np.array( [ O3_32I_ml0p004.cv, O3_24I_ml0p005.cv, O3_32I_ml0p006.cv, O3_32I_ml0p008.cv, O3_24I_ml0p01.cv ] )
  this_dO3  = np.array( [ O3_32I_ml0p004.std, O3_24I_ml0p005.std, O3_32I_ml0p006.std, O3_32I_ml0p008.std, O3_24I_ml0p01.std ] )
  # fit = minimize(SD_chpt.chi2_O3_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O3, this_dO3), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  fit = minimize(SD_chpt.chi2_O3_chptfv_a2, [0.6,1.7,51.4], args=(this_mpi, this_fpi, this_a2, this_fv, this_O3, this_dO3), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  print(fit.x[2], fit.x[0], fit.x[1])
  b3_jacks.cv              = fit.x[0]
  c3_jacks.cv              = fit.x[1]
  c3a_jacks.cv             = fit.x[2]
  chi2_O3_jacks.cv         = fit.fun
  chi2perdof_O3_jacks.cv   = chi2_O3_jacks.cv / ( 5.0 - 3.0 )
  epi2_corrected_32I_ml0p004_jacks.cv = mpi_32I_ml0p004.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p006_jacks.cv = mpi_32I_ml0p006.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p008_jacks.cv = mpi_32I_ml0p008.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p005_jacks.cv = mpi_24I_ml0p005.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p01_jacks.cv  = mpi_24I_ml0p01.cv**2  / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  O3_corrected_32I_ml0p004_jacks.cv = this_O3[0] - ( SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[0], fpi_phys.cv, 0.0, 0.0) )
  O3_corrected_24I_ml0p005_jacks.cv = this_O3[1] - ( SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[1], fpi_phys.cv, 0.0, 0.0) )
  O3_corrected_32I_ml0p006_jacks.cv = this_O3[2] - ( SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[2], fpi_phys.cv, 0.0, 0.0) )
  O3_corrected_32I_ml0p008_jacks.cv = this_O3[3] - ( SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[3], fpi_phys.cv, 0.0, 0.0) )
  O3_corrected_24I_ml0p01_jacks.cv  = this_O3[4] - ( SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, this_mpi[4], fpi_phys.cv, 0.0, 0.0) )
  O3_phys_jacks.cv = SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, mpi_phys.cv, fpi_phys.cv, 0.0, 0.0)
  for idx in range(0, len(xx)-1):
    y3_jacks[idx].cv = SD_chpt.O3_chptfv_a2(b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv, xx[idx+1], fpi_phys.cv, 0.0, 0.0)
    y3[idx] = y3_jacks[idx].cv

  # Fit to superjackknife samples
  this_boot = 0
  for ens in ensembles:
    Nboots = ensembles[ens]
    for idx in range(0, Nboots):
      print("\t-- Fitting to boot {0} of {1}...".format(this_boot+1, Nsuperboots))
      this_mpi  = np.array( [ mpi_32I_ml0p004.data[ens][idx], mpi_24I_ml0p005.data[ens][idx], mpi_32I_ml0p006.data[ens][idx], mpi_32I_ml0p008.data[ens][idx], mpi_24I_ml0p01.data[ens][idx] ] )
      this_fpi  = np.array( [ fpi_32I_ml0p004.data[ens][idx], fpi_24I_ml0p005.data[ens][idx], fpi_32I_ml0p006.data[ens][idx], fpi_32I_ml0p008.data[ens][idx], fpi_24I_ml0p01.data[ens][idx] ] )
      this_a2   = np.array( [ a2_32I.data[ens][idx], a2_24I.data[ens][idx], a2_32I.data[ens][idx], a2_32I.data[ens][idx], a2_24I.data[ens][idx] ] )
      this_f0 = np.array( [ SD_chpt.f0(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                            SD_chpt.f0(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 24.0), \
                            SD_chpt.f0(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                            SD_chpt.f0(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                            SD_chpt.f0(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx] , 24.0) ] )
      this_f1 = np.array( [ SD_chpt.f1(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                            SD_chpt.f1(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 24.0), \
                            SD_chpt.f1(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                            SD_chpt.f1(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                            SD_chpt.f1(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx] , 24.0) ] )
      this_fv = 2.0*this_f1 + this_f0
      this_O3   = np.array( [ O3_32I_ml0p004.data[ens][idx], O3_24I_ml0p005.data[ens][idx], O3_32I_ml0p006.data[ens][idx], O3_32I_ml0p008.data[ens][idx], O3_24I_ml0p01.data[ens][idx] ] )
      this_dO3  = np.array( [ O3_32I_ml0p004.std, O3_24I_ml0p005.std, O3_32I_ml0p006.std, O3_32I_ml0p008.std, O3_24I_ml0p01.std ] )
      # fit = minimize(SD_chpt.chi2_O3_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O3, this_dO3), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      fit = minimize(SD_chpt.chi2_O3_chptfv_a2, [b3_jacks.cv, c3_jacks.cv, c3a_jacks.cv], args=(this_mpi, this_fpi, this_a2, this_fv, this_O3, this_dO3), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      b3_jacks.data[ens][idx]  = fit.x[0]
      c3_jacks.data[ens][idx]  = fit.x[1]
      c3a_jacks.data[ens][idx] = fit.x[2]
      chi2_O3_jacks.data[ens][idx] = fit.fun
      chi2perdof_O3_jacks.data[ens][idx] = chi2_O3_jacks.data[ens][idx] / ( 5.0 - 3.0 )
      O3_phys_jacks.data[ens][idx] = SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], mpi_phys.data[ens][idx], fpi_phys.data[ens][idx], 0.0, 0.0)
      epi2_corrected_32I_ml0p004_jacks.data[ens][idx] = mpi_32I_ml0p004.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p006_jacks.data[ens][idx] = mpi_32I_ml0p006.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p008_jacks.data[ens][idx] = mpi_32I_ml0p008.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p005_jacks.data[ens][idx] = mpi_24I_ml0p005.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p01_jacks.data[ens][idx]  = mpi_24I_ml0p01.data[ens][idx]**2  / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      O3_corrected_32I_ml0p004_jacks.data[ens][idx] = this_O3[0] - ( SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[0], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O3_corrected_24I_ml0p005_jacks.data[ens][idx] = this_O3[1] - ( SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[1], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O3_corrected_32I_ml0p006_jacks.data[ens][idx] = this_O3[2] - ( SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[2], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O3_corrected_32I_ml0p008_jacks.data[ens][idx] = this_O3[3] - ( SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[3], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O3_corrected_24I_ml0p01_jacks.data[ens][idx]  = this_O3[4] - ( SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], this_mpi[4], fpi_phys.data[ens][idx], 0.0, 0.0) )
      for ii in range(0, len(xx)-1):
        y3_jacks[ii].data[ens][idx] = SD_chpt.O3_chptfv_a2(b3_jacks.data[ens][idx], c3_jacks.data[ens][idx], c3a_jacks.data[ens][idx], xx[ii+1], fpi_phys.data[ens][idx], 0.0, 0.0)
      this_boot += 1

  # Compute errors
  b3_jacks.calc_mean()
  b3_jacks.calc_std()
  c3_jacks.calc_mean()
  c3_jacks.calc_std()
  c3a_jacks.calc_mean()
  c3a_jacks.calc_std()
  chi2_O3_jacks.calc_mean()
  chi2_O3_jacks.calc_std()
  chi2perdof_O3_jacks.calc_mean()
  chi2perdof_O3_jacks.calc_std()
  O3_phys_jacks.calc_mean()
  O3_phys_jacks.calc_std()
  epi2_corrected_32I_ml0p004_jacks.calc_mean()
  epi2_corrected_32I_ml0p004_jacks.calc_std()
  epi2_corrected_32I_ml0p006_jacks.calc_mean()
  epi2_corrected_32I_ml0p006_jacks.calc_std()
  epi2_corrected_32I_ml0p008_jacks.calc_mean()
  epi2_corrected_32I_ml0p008_jacks.calc_std()
  epi2_corrected_24I_ml0p005_jacks.calc_mean()
  epi2_corrected_24I_ml0p005_jacks.calc_std()
  epi2_corrected_24I_ml0p01_jacks.calc_mean()
  epi2_corrected_24I_ml0p01_jacks.calc_std()
  O3_corrected_32I_ml0p004_jacks.calc_mean()
  O3_corrected_32I_ml0p004_jacks.calc_std()
  O3_corrected_32I_ml0p006_jacks.calc_mean()
  O3_corrected_32I_ml0p006_jacks.calc_std()
  O3_corrected_32I_ml0p008_jacks.calc_mean()
  O3_corrected_32I_ml0p008_jacks.calc_std()
  O3_corrected_24I_ml0p005_jacks.calc_mean()
  O3_corrected_24I_ml0p005_jacks.calc_std()
  O3_corrected_24I_ml0p01_jacks.calc_mean()
  O3_corrected_24I_ml0p01_jacks.calc_std()
  for ii in range(0, len(xx)-1):
    y3_jacks[ii].calc_mean()
    y3_jacks[ii].calc_std()
    dy3[ii] = y3_jacks[ii].std

  print("\t-- Result: O3 = {0:.8e} +/- {1:.8e}".format(O3_phys_jacks.cv, O3_phys_jacks.std))
  print("\t-- Result: b3 = {0:.8e} +/- {1:.8e}".format(b3_jacks.cv, b3_jacks.std))
  print("\t-- Result: c3 = {0:.8e} +/- {1:.8e}".format(c3_jacks.cv, c3_jacks.std))
  print("\t-- Result: c3a = {0:.8e} +/- {1:.8e}".format(c3a_jacks.cv, c3a_jacks.std))
  print("\t-- Result: chi2pdof = {0:.8e} +/- {1:.8e}".format(chi2perdof_O3_jacks.cv, chi2perdof_O3_jacks.std))

  with open("./results/O3_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p004.cv, epi2_32I_ml0p004.std, O3_32I_ml0p004.cv, O3_32I_ml0p004.std), file=f)

  with open("./results/O3_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p006.cv, epi2_32I_ml0p006.std, O3_32I_ml0p006.cv, O3_32I_ml0p006.std), file=f)

  with open("./results/O3_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p008.cv, epi2_32I_ml0p008.std, O3_32I_ml0p008.cv, O3_32I_ml0p008.std), file=f)

  with open("./results/O3_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p005.cv, epi2_24I_ml0p005.std, O3_24I_ml0p005.cv, O3_24I_ml0p005.std), file=f)

  with open("./results/O3_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p01.cv, epi2_24I_ml0p01.std, O3_24I_ml0p01.cv, O3_24I_ml0p01.std), file=f)

  with open("./results/O3_corrected_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p004_jacks.cv, epi2_corrected_32I_ml0p004_jacks.std, O3_corrected_32I_ml0p004_jacks.cv, O3_corrected_32I_ml0p004_jacks.std), file=f)

  with open("./results/O3_corrected_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p006_jacks.cv, epi2_corrected_32I_ml0p006_jacks.std, O3_corrected_32I_ml0p006_jacks.cv, O3_corrected_32I_ml0p006_jacks.std), file=f)

  with open("./results/O3_corrected_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p008_jacks.cv, epi2_corrected_32I_ml0p008_jacks.std, O3_corrected_32I_ml0p008_jacks.cv, O3_corrected_32I_ml0p008_jacks.std), file=f)

  with open("./results/O3_corrected_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p005_jacks.cv, epi2_corrected_24I_ml0p005_jacks.std, O3_corrected_24I_ml0p005_jacks.cv, O3_corrected_24I_ml0p005_jacks.std), file=f)

  with open("./results/O3_corrected_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p01_jacks.cv, epi2_corrected_24I_ml0p01_jacks.std, O3_corrected_24I_ml0p01_jacks.cv, O3_corrected_24I_ml0p01_jacks.std), file=f)

  with open("./results/O3_phys.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_phys.cv, epi2_phys.std, O3_phys_jacks.cv, O3_phys_jacks.std), file=f)

  with open("./results/O3_extrap.dat", "w") as f:
    for idx in range(0, len(xx)-1):
      print("{0:.8e} {1:.8e} {2:.8e}".format(epi2_xx[idx+1], y3[idx], dy3[idx]), file=f)

#####################################################################################################
# Fit O1p

if fit_O1p:

  print("\n===== Fitting to O1p =====")
  xx                               = np.linspace(0.0, 0.5, 101)
  epi2_xx                          = xx**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  y1p                               = np.zeros((len(xx)-1))
  dy1p                              = np.zeros((len(xx)-1))
  b1p_jacks                         = superjack.distribution(ensembles)
  c1p_jacks                         = superjack.distribution(ensembles)
  c1pa_jacks                        = superjack.distribution(ensembles)
  chi2_O1p_jacks                    = superjack.distribution(ensembles)
  chi2perdof_O1p_jacks              = superjack.distribution(ensembles)
  O1p_phys_jacks                    = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p004_jacks  = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p006_jacks  = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p008_jacks  = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p005_jacks  = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p01_jacks   = superjack.distribution(ensembles)
  O1p_corrected_32I_ml0p004_jacks   = superjack.distribution(ensembles)
  O1p_corrected_32I_ml0p006_jacks   = superjack.distribution(ensembles)
  O1p_corrected_32I_ml0p008_jacks   = superjack.distribution(ensembles)
  O1p_corrected_24I_ml0p005_jacks   = superjack.distribution(ensembles)
  O1p_corrected_24I_ml0p01_jacks    = superjack.distribution(ensembles)
  y1p_jacks            = []
  for idx in range(0, len(xx)-1):
    y1p_jacks.append(superjack.distribution(ensembles))

  # Fit to central values
  print("\t-- Fitting to central value...")
  this_mpi  = np.array( [ mpi_32I_ml0p004.cv, mpi_24I_ml0p005.cv, mpi_32I_ml0p006.cv, mpi_32I_ml0p008.cv, mpi_24I_ml0p01.cv ] )
  this_fpi  = np.array( [ fpi_32I_ml0p004.cv, fpi_24I_ml0p005.cv, fpi_32I_ml0p006.cv, fpi_32I_ml0p008.cv, fpi_24I_ml0p01.cv ] )
  this_a2   = np.array( [ a2_32I.cv, a2_24I.cv, a2_32I.cv, a2_32I.cv, a2_24I.cv ] )
  this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p005.cv/ainv_24I.cv, 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.cv/ainv_24I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p01.cv/ainv_24I.cv , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.cv/ainv_24I.cv, 32.0) ] )
  this_fv = this_f0
  this_O1p   = np.array( [ O1p_32I_ml0p004.cv, O1p_24I_ml0p005.cv, O1p_32I_ml0p006.cv, O1p_32I_ml0p008.cv, O1p_24I_ml0p01.cv ] )
  this_dO1p  = np.array( [ O1p_32I_ml0p004.std, O1p_24I_ml0p005.std, O1p_32I_ml0p006.std, O1p_32I_ml0p008.std, O1p_24I_ml0p01.std ] )
  # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O1p, this_dO1p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [-5.5,-1.1,-1.5], args=(this_mpi, this_fpi, this_a2, this_fv, this_O1p, this_dO1p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  print(fit.x[2], fit.x[0], fit.x[1])
  b1p_jacks.cv              = fit.x[0]
  c1p_jacks.cv              = fit.x[1]
  c1pa_jacks.cv             = fit.x[2]
  chi2_O1p_jacks.cv         = fit.fun
  chi2perdof_O1p_jacks.cv   = chi2_O1p_jacks.cv / ( 5.0 - 3.0 )
  epi2_corrected_32I_ml0p004_jacks.cv = mpi_32I_ml0p004.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p006_jacks.cv = mpi_32I_ml0p006.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p008_jacks.cv = mpi_32I_ml0p008.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p005_jacks.cv = mpi_24I_ml0p005.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p01_jacks.cv  = mpi_24I_ml0p01.cv**2  / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  O1p_corrected_32I_ml0p004_jacks.cv = this_O1p[0] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[0], fpi_phys.cv, 0.0, 0.0) )
  O1p_corrected_24I_ml0p005_jacks.cv = this_O1p[1] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[1], fpi_phys.cv, 0.0, 0.0) )
  O1p_corrected_32I_ml0p006_jacks.cv = this_O1p[2] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[2], fpi_phys.cv, 0.0, 0.0) )
  O1p_corrected_32I_ml0p008_jacks.cv = this_O1p[3] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[3], fpi_phys.cv, 0.0, 0.0) )
  O1p_corrected_24I_ml0p01_jacks.cv  = this_O1p[4] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, this_mpi[4], fpi_phys.cv, 0.0, 0.0) )
  O1p_phys_jacks.cv = SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, mpi_phys.cv, fpi_phys.cv, 0.0, 0.0)
  for idx in range(0, len(xx)-1):
    y1p_jacks[idx].cv = SD_chpt.O12_chptfv_a2(b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv, xx[idx+1], fpi_phys.cv, 0.0, 0.0)
    y1p[idx] = y1p_jacks[idx].cv

  # Fit to superjackknife samples
  this_boot = 0
  for ens in ensembles:
    Nboots = ensembles[ens]
    for idx in range(0, Nboots):
      print("\t-- Fitting to boot {0} of {1}...".format(this_boot+1, Nsuperboots))
      this_mpi  = np.array( [ mpi_32I_ml0p004.data[ens][idx], mpi_24I_ml0p005.data[ens][idx], mpi_32I_ml0p006.data[ens][idx], mpi_32I_ml0p008.data[ens][idx], mpi_24I_ml0p01.data[ens][idx] ] )
      this_fpi  = np.array( [ fpi_32I_ml0p004.data[ens][idx], fpi_24I_ml0p005.data[ens][idx], fpi_32I_ml0p006.data[ens][idx], fpi_32I_ml0p008.data[ens][idx], fpi_24I_ml0p01.data[ens][idx] ] )
      this_a2   = np.array( [ a2_32I.data[ens][idx], a2_24I.data[ens][idx], a2_32I.data[ens][idx], a2_32I.data[ens][idx], a2_24I.data[ens][idx] ] )
      this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx] , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx], 32.0) ] )
      this_fv   = this_f0
      this_O1p   = np.array( [ O1p_32I_ml0p004.data[ens][idx], O1p_24I_ml0p005.data[ens][idx], O1p_32I_ml0p006.data[ens][idx], O1p_32I_ml0p008.data[ens][idx], O1p_24I_ml0p01.data[ens][idx] ] )
      this_dO1p  = np.array( [ O1p_32I_ml0p004.std, O1p_24I_ml0p005.std, O1p_32I_ml0p006.std, O1p_32I_ml0p008.std, O1p_24I_ml0p01.std ] )
      # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O1p, this_dO1p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [b1p_jacks.cv, c1p_jacks.cv, c1pa_jacks.cv], args=(this_mpi, this_fpi, this_a2, this_fv, this_O1p, this_dO1p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      b1p_jacks.data[ens][idx]  = fit.x[0]
      c1p_jacks.data[ens][idx]  = fit.x[1]
      c1pa_jacks.data[ens][idx] = fit.x[2]
      chi2_O1p_jacks.data[ens][idx] = fit.fun
      chi2perdof_O1p_jacks.data[ens][idx] = chi2_O1p_jacks.data[ens][idx] / ( 5.0 - 3.0 )
      epi2_corrected_32I_ml0p004_jacks.data[ens][idx] = mpi_32I_ml0p004.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p006_jacks.data[ens][idx] = mpi_32I_ml0p006.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p008_jacks.data[ens][idx] = mpi_32I_ml0p008.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p005_jacks.data[ens][idx] = mpi_24I_ml0p005.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p01_jacks.data[ens][idx]  = mpi_24I_ml0p01.data[ens][idx]**2  / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      O1p_phys_jacks.data[ens][idx] = SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], mpi_phys.data[ens][idx], fpi_phys.data[ens][idx], 0.0, 0.0)
      O1p_corrected_32I_ml0p004_jacks.data[ens][idx] = this_O1p[0] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[0], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1p_corrected_24I_ml0p005_jacks.data[ens][idx] = this_O1p[1] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[1], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1p_corrected_32I_ml0p006_jacks.data[ens][idx] = this_O1p[2] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[2], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1p_corrected_32I_ml0p008_jacks.data[ens][idx] = this_O1p[3] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[3], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O1p_corrected_24I_ml0p01_jacks.data[ens][idx]  = this_O1p[4] - ( SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], this_mpi[4], fpi_phys.data[ens][idx], 0.0, 0.0) )
      for ii in range(0, len(xx)-1):
        y1p_jacks[ii].data[ens][idx] = SD_chpt.O12_chptfv_a2(b1p_jacks.data[ens][idx], c1p_jacks.data[ens][idx], c1pa_jacks.data[ens][idx], xx[ii+1], fpi_phys.data[ens][idx], 0.0, 0.0)
      this_boot += 1

  # Compute errors
  b1p_jacks.calc_mean()
  b1p_jacks.calc_std()
  c1p_jacks.calc_mean()
  c1p_jacks.calc_std()
  c1pa_jacks.calc_mean()
  c1pa_jacks.calc_std()
  chi2_O1p_jacks.calc_mean()
  chi2_O1p_jacks.calc_std()
  chi2perdof_O1p_jacks.calc_mean()
  chi2perdof_O1p_jacks.calc_std()
  O1p_phys_jacks.calc_mean()
  O1p_phys_jacks.calc_std()
  epi2_corrected_32I_ml0p004_jacks.calc_mean()
  epi2_corrected_32I_ml0p004_jacks.calc_std()
  epi2_corrected_32I_ml0p006_jacks.calc_mean()
  epi2_corrected_32I_ml0p006_jacks.calc_std()
  epi2_corrected_32I_ml0p008_jacks.calc_mean()
  epi2_corrected_32I_ml0p008_jacks.calc_std()
  epi2_corrected_24I_ml0p005_jacks.calc_mean()
  epi2_corrected_24I_ml0p005_jacks.calc_std()
  epi2_corrected_24I_ml0p01_jacks.calc_mean()
  epi2_corrected_24I_ml0p01_jacks.calc_std()
  O1p_corrected_32I_ml0p004_jacks.calc_mean()
  O1p_corrected_32I_ml0p004_jacks.calc_std()
  O1p_corrected_32I_ml0p006_jacks.calc_mean()
  O1p_corrected_32I_ml0p006_jacks.calc_std()
  O1p_corrected_32I_ml0p008_jacks.calc_mean()
  O1p_corrected_32I_ml0p008_jacks.calc_std()
  O1p_corrected_24I_ml0p005_jacks.calc_mean()
  O1p_corrected_24I_ml0p005_jacks.calc_std()
  O1p_corrected_24I_ml0p01_jacks.calc_mean()
  O1p_corrected_24I_ml0p01_jacks.calc_std()
  for ii in range(0, len(xx)-1):
    y1p_jacks[ii].calc_mean()
    y1p_jacks[ii].calc_std()
    dy1p[ii] = y1p_jacks[ii].std

  print("\t-- Result: O1p = {0:.8e} +/- {1:.8e}".format(O1p_phys_jacks.cv, O1p_phys_jacks.std))
  print("\t-- Result: b1p = {0:.8e} +/- {1:.8e}".format(b1p_jacks.cv, b1p_jacks.std))
  print("\t-- Result: c1p = {0:.8e} +/- {1:.8e}".format(c1p_jacks.cv, c1p_jacks.std))
  print("\t-- Result: c1pa = {0:.8e} +/- {1:.8e}".format(c1pa_jacks.cv, c1pa_jacks.std))
  print("\t-- Result: chi2pdof = {0:.8e} +/- {1:.8e}".format(chi2perdof_O1p_jacks.cv, chi2perdof_O1p_jacks.std))

  with open("./results/O1p_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p004.cv, epi2_32I_ml0p004.std, O1p_32I_ml0p004.cv, O1p_32I_ml0p004.std), file=f)

  with open("./results/O1p_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p006.cv, epi2_32I_ml0p006.std, O1p_32I_ml0p006.cv, O1p_32I_ml0p006.std), file=f)

  with open("./results/O1p_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p008.cv, epi2_32I_ml0p008.std, O1p_32I_ml0p008.cv, O1p_32I_ml0p008.std), file=f)

  with open("./results/O1p_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p005.cv, epi2_24I_ml0p005.std, O1p_24I_ml0p005.cv, O1p_24I_ml0p005.std), file=f)

  with open("./results/O1p_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p01.cv, epi2_24I_ml0p01.std, O1p_24I_ml0p01.cv, O1p_24I_ml0p01.std), file=f)

  with open("./results/O1p_corrected_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p004_jacks.cv, epi2_corrected_32I_ml0p004_jacks.std, O1p_corrected_32I_ml0p004_jacks.cv, O1p_corrected_32I_ml0p004_jacks.std), file=f)

  with open("./results/O1p_corrected_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p006_jacks.cv, epi2_corrected_32I_ml0p006_jacks.std, O1p_corrected_32I_ml0p006_jacks.cv, O1p_corrected_32I_ml0p006_jacks.std), file=f)

  with open("./results/O1p_corrected_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p008_jacks.cv, epi2_corrected_32I_ml0p008_jacks.std, O1p_corrected_32I_ml0p008_jacks.cv, O1p_corrected_32I_ml0p008_jacks.std), file=f)

  with open("./results/O1p_corrected_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p005_jacks.cv, epi2_corrected_24I_ml0p005_jacks.std, O1p_corrected_24I_ml0p005_jacks.cv, O1p_corrected_24I_ml0p005_jacks.std), file=f)

  with open("./results/O1p_corrected_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p01_jacks.cv, epi2_corrected_24I_ml0p01_jacks.std, O1p_corrected_24I_ml0p01_jacks.cv, O1p_corrected_24I_ml0p01_jacks.std), file=f)

  with open("./results/O1p_phys.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_phys.cv, epi2_phys.std, O1p_phys_jacks.cv, O1p_phys_jacks.std), file=f)

  with open("./results/O1p_extrap.dat", "w") as f:
    for idx in range(0, len(xx)-1):
      print("{0:.8e} {1:.8e} {2:.8e}".format(epi2_xx[idx+1], y1p[idx], dy1p[idx]), file=f)

#####################################################################################################
# Fit O2p

if fit_O2p:

  print("\n===== Fitting to O2p =====")
  xx                               = np.linspace(0.0, 0.5, 101)
  epi2_xx                          = xx**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  y2p                               = np.zeros((len(xx)-1))
  dy2p                              = np.zeros((len(xx)-1))
  b2p_jacks                         = superjack.distribution(ensembles)
  c2p_jacks                         = superjack.distribution(ensembles)
  c2pa_jacks                        = superjack.distribution(ensembles)
  chi2_O2p_jacks                    = superjack.distribution(ensembles)
  chi2perdof_O2p_jacks              = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p004_jacks  = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p006_jacks  = superjack.distribution(ensembles)
  epi2_corrected_32I_ml0p008_jacks  = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p005_jacks  = superjack.distribution(ensembles)
  epi2_corrected_24I_ml0p01_jacks   = superjack.distribution(ensembles)
  O2p_phys_jacks                    = superjack.distribution(ensembles)
  O2p_corrected_32I_ml0p004_jacks   = superjack.distribution(ensembles)
  O2p_corrected_32I_ml0p006_jacks   = superjack.distribution(ensembles)
  O2p_corrected_32I_ml0p008_jacks   = superjack.distribution(ensembles)
  O2p_corrected_24I_ml0p005_jacks   = superjack.distribution(ensembles)
  O2p_corrected_24I_ml0p01_jacks    = superjack.distribution(ensembles)
  y2p_jacks            = []
  for idx in range(0, len(xx)-1):
    y2p_jacks.append(superjack.distribution(ensembles))

  # Fit to central values
  print("\t-- Fitting to central value...")
  this_mpi  = np.array( [ mpi_32I_ml0p004.cv, mpi_24I_ml0p005.cv, mpi_32I_ml0p006.cv, mpi_32I_ml0p008.cv, mpi_24I_ml0p01.cv ] )
  this_fpi  = np.array( [ fpi_32I_ml0p004.cv, fpi_24I_ml0p005.cv, fpi_32I_ml0p006.cv, fpi_32I_ml0p008.cv, fpi_24I_ml0p01.cv ] )
  this_a2   = np.array( [ a2_32I.cv, a2_24I.cv, a2_32I.cv, a2_32I.cv, a2_24I.cv ] )
  this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p005.cv/ainv_24I.cv, 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.cv/ainv_24I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.cv/ainv_32I.cv, 32.0), \
                          -SD_chpt.f0(mpi_24I_ml0p01.cv/ainv_24I.cv , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.cv/ainv_24I.cv, 32.0) ] )
  this_fv = this_f0
  this_O2p   = np.array( [ O2p_32I_ml0p004.cv, O2p_24I_ml0p005.cv, O2p_32I_ml0p006.cv, O2p_32I_ml0p008.cv, O2p_24I_ml0p01.cv ] )
  this_dO2p  = np.array( [ O2p_32I_ml0p004.std, O2p_24I_ml0p005.std, O2p_32I_ml0p006.std, O2p_32I_ml0p008.std, O2p_24I_ml0p01.std ] )
  fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [1.3,-1.1,7.7], args=(this_mpi, this_fpi, this_a2, this_fv, this_O2p, this_dO2p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
  print(fit.x[2], fit.x[0], fit.x[1])
  b2p_jacks.cv              = fit.x[0]
  c2p_jacks.cv              = fit.x[1]
  c2pa_jacks.cv             = fit.x[2]
  chi2_O2p_jacks.cv         = fit.fun
  chi2perdof_O2p_jacks.cv   = chi2_O2p_jacks.cv / ( 5.0 - 3.0 )
  epi2_corrected_32I_ml0p004_jacks.cv = mpi_32I_ml0p004.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p006_jacks.cv = mpi_32I_ml0p006.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_32I_ml0p008_jacks.cv = mpi_32I_ml0p008.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p005_jacks.cv = mpi_24I_ml0p005.cv**2 / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  epi2_corrected_24I_ml0p01_jacks.cv  = mpi_24I_ml0p01.cv**2  / ( 8.0 * np.pi**2 * fpi_phys.cv**2 )
  O2p_corrected_32I_ml0p004_jacks.cv = this_O2p[0] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[0], fpi_phys.cv, 0.0, 0.0) )
  O2p_corrected_24I_ml0p005_jacks.cv = this_O2p[1] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[1], fpi_phys.cv, 0.0, 0.0) )
  O2p_corrected_32I_ml0p006_jacks.cv = this_O2p[2] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[2], fpi_phys.cv, 0.0, 0.0) )
  O2p_corrected_32I_ml0p008_jacks.cv = this_O2p[3] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[3], fpi_phys.cv, 0.0, 0.0) )
  O2p_corrected_24I_ml0p01_jacks.cv  = this_O2p[4] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, this_mpi[4], fpi_phys.cv, 0.0, 0.0) )
  O2p_phys_jacks.cv = SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, mpi_phys.cv, fpi_phys.cv, 0.0, 0.0)
  for idx in range(0, len(xx)-1):
    y2p_jacks[idx].cv = SD_chpt.O12_chptfv_a2(b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv, xx[idx+1], fpi_phys.cv, 0.0, 0.0)
    y2p[idx] = y2p_jacks[idx].cv

  # Fit to superjackknife samples
  this_boot = 0
  for ens in ensembles:
    Nboots = ensembles[ens]
    for idx in range(0, Nboots):
      print("\t-- Fitting to boot {0} of {1}...".format(this_boot+1, Nsuperboots))
      this_mpi  = np.array( [ mpi_32I_ml0p004.data[ens][idx], mpi_24I_ml0p005.data[ens][idx], mpi_32I_ml0p006.data[ens][idx], mpi_32I_ml0p008.data[ens][idx], mpi_24I_ml0p01.data[ens][idx] ] )
      this_fpi  = np.array( [ fpi_32I_ml0p004.data[ens][idx], fpi_24I_ml0p005.data[ens][idx], fpi_32I_ml0p006.data[ens][idx], fpi_32I_ml0p008.data[ens][idx], fpi_24I_ml0p01.data[ens][idx] ] )
      this_a2   = np.array( [ a2_32I.data[ens][idx], a2_24I.data[ens][idx], a2_32I.data[ens][idx], a2_32I.data[ens][idx], a2_24I.data[ens][idx] ] )
      this_f0   = np.array( [ -SD_chpt.f0(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p004.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p005.data[ens][idx]/ainv_24I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p006.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0) + 2.0*SD_chpt.f1(mpi_32I_ml0p008.data[ens][idx]/ainv_32I.data[ens][idx], 32.0), \
                              -SD_chpt.f0(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx] , 24.0) + 2.0*SD_chpt.f1(mpi_24I_ml0p01.data[ens][idx]/ainv_24I.data[ens][idx], 32.0) ] )
      this_fv   = this_f0
      this_O2p   = np.array( [ O2p_32I_ml0p004.data[ens][idx], O2p_24I_ml0p005.data[ens][idx], O2p_32I_ml0p006.data[ens][idx], O2p_32I_ml0p008.data[ens][idx], O2p_24I_ml0p01.data[ens][idx] ] )
      this_dO2p  = np.array( [ O2p_32I_ml0p004.std, O2p_24I_ml0p005.std, O2p_32I_ml0p006.std, O2p_32I_ml0p008.std, O2p_24I_ml0p01.std ] )
      # fit = minimize(SD_chpt.chi2_O12_chpt_a2, [1.0,1.0,1.0], args=(this_mpi, this_fpi, this_a2, this_O2p, this_dO2p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      fit = minimize(SD_chpt.chi2_O12_chptfv_a2, [b2p_jacks.cv, c2p_jacks.cv, c2pa_jacks.cv], args=(this_mpi, this_fpi, this_a2, this_fv, this_O2p, this_dO2p), method='Powell', options={'maxiter':10000, 'ftol':1.0e-08})
      b2p_jacks.data[ens][idx]  = fit.x[0]
      c2p_jacks.data[ens][idx]  = fit.x[1]
      c2pa_jacks.data[ens][idx] = fit.x[2]
      chi2_O2p_jacks.data[ens][idx] = fit.fun
      chi2perdof_O2p_jacks.data[ens][idx] = chi2_O2p_jacks.data[ens][idx] / ( 5.0 - 3.0 )
      epi2_corrected_32I_ml0p004_jacks.data[ens][idx] = mpi_32I_ml0p004.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p006_jacks.data[ens][idx] = mpi_32I_ml0p006.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_32I_ml0p008_jacks.data[ens][idx] = mpi_32I_ml0p008.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p005_jacks.data[ens][idx] = mpi_24I_ml0p005.data[ens][idx]**2 / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      epi2_corrected_24I_ml0p01_jacks.data[ens][idx]  = mpi_24I_ml0p01.data[ens][idx]**2  / ( 8.0 * np.pi**2 * fpi_phys.data[ens][idx]**2 )
      O2p_phys_jacks.data[ens][idx] = SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], mpi_phys.data[ens][idx], fpi_phys.data[ens][idx], 0.0, 0.0)
      O2p_corrected_32I_ml0p004_jacks.data[ens][idx] = this_O2p[0] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[0], this_fpi[0], this_a2[0], this_fv[0]) - SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[0], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2p_corrected_24I_ml0p005_jacks.data[ens][idx] = this_O2p[1] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[1], this_fpi[1], this_a2[1], this_fv[1]) - SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[1], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2p_corrected_32I_ml0p006_jacks.data[ens][idx] = this_O2p[2] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[2], this_fpi[2], this_a2[2], this_fv[2]) - SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[2], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2p_corrected_32I_ml0p008_jacks.data[ens][idx] = this_O2p[3] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[3], this_fpi[3], this_a2[3], this_fv[3]) - SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[3], fpi_phys.data[ens][idx], 0.0, 0.0) )
      O2p_corrected_24I_ml0p01_jacks.data[ens][idx]  = this_O2p[4] - ( SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[4], this_fpi[4], this_a2[4], this_fv[4]) - SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], this_mpi[4], fpi_phys.data[ens][idx], 0.0, 0.0) )
      for ii in range(0, len(xx)-1):
        y2p_jacks[ii].data[ens][idx] = SD_chpt.O12_chptfv_a2(b2p_jacks.data[ens][idx], c2p_jacks.data[ens][idx], c2pa_jacks.data[ens][idx], xx[ii+1], fpi_phys.data[ens][idx], 0.0, 0.0)
      this_boot += 1

  # Compute errors
  b2p_jacks.calc_mean()
  b2p_jacks.calc_std()
  c2p_jacks.calc_mean()
  c2p_jacks.calc_std()
  c2pa_jacks.calc_mean()
  c2pa_jacks.calc_std()
  chi2_O2p_jacks.calc_mean()
  chi2_O2p_jacks.calc_std()
  chi2perdof_O2p_jacks.calc_mean()
  chi2perdof_O2p_jacks.calc_std()
  O2p_phys_jacks.calc_mean()
  O2p_phys_jacks.calc_std()
  epi2_corrected_32I_ml0p004_jacks.calc_mean()
  epi2_corrected_32I_ml0p004_jacks.calc_std()
  epi2_corrected_32I_ml0p006_jacks.calc_mean()
  epi2_corrected_32I_ml0p006_jacks.calc_std()
  epi2_corrected_32I_ml0p008_jacks.calc_mean()
  epi2_corrected_32I_ml0p008_jacks.calc_std()
  epi2_corrected_24I_ml0p005_jacks.calc_mean()
  epi2_corrected_24I_ml0p005_jacks.calc_std()
  epi2_corrected_24I_ml0p01_jacks.calc_mean()
  epi2_corrected_24I_ml0p01_jacks.calc_std()
  O2p_corrected_32I_ml0p004_jacks.calc_mean()
  O2p_corrected_32I_ml0p004_jacks.calc_std()
  O2p_corrected_32I_ml0p006_jacks.calc_mean()
  O2p_corrected_32I_ml0p006_jacks.calc_std()
  O2p_corrected_32I_ml0p008_jacks.calc_mean()
  O2p_corrected_32I_ml0p008_jacks.calc_std()
  O2p_corrected_24I_ml0p005_jacks.calc_mean()
  O2p_corrected_24I_ml0p005_jacks.calc_std()
  O2p_corrected_24I_ml0p01_jacks.calc_mean()
  O2p_corrected_24I_ml0p01_jacks.calc_std()
  for ii in range(0, len(xx)-1):
    y2p_jacks[ii].calc_mean()
    y2p_jacks[ii].calc_std()
    dy2p[ii] = y2p_jacks[ii].std

  print("\t-- Result: O2p = {0:.8e} +/- {1:.8e}".format(O2p_phys_jacks.cv, O2p_phys_jacks.std))
  print("\t-- Result: b2p = {0:.8e} +/- {1:.8e}".format(b2p_jacks.cv, b2p_jacks.std))
  print("\t-- Result: c2p = {0:.8e} +/- {1:.8e}".format(c2p_jacks.cv, c2p_jacks.std))
  print("\t-- Result: c2pa = {0:.8e} +/- {1:.8e}".format(c2pa_jacks.cv, c2pa_jacks.std))
  print("\t-- Result: chi2pdof = {0:.8e} +/- {1:.8e}".format(chi2perdof_O2p_jacks.cv, chi2perdof_O2p_jacks.std))

  with open("./results/O2p_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p004.cv, epi2_32I_ml0p004.std, O2p_32I_ml0p004.cv, O2p_32I_ml0p004.std), file=f)

  with open("./results/O2p_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p006.cv, epi2_32I_ml0p006.std, O2p_32I_ml0p006.cv, O2p_32I_ml0p006.std), file=f)

  with open("./results/O2p_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_32I_ml0p008.cv, epi2_32I_ml0p008.std, O2p_32I_ml0p008.cv, O2p_32I_ml0p008.std), file=f)

  with open("./results/O2p_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p005.cv, epi2_24I_ml0p005.std, O2p_24I_ml0p005.cv, O2p_24I_ml0p005.std), file=f)

  with open("./results/O2p_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_24I_ml0p01.cv, epi2_24I_ml0p01.std, O2p_24I_ml0p01.cv, O2p_24I_ml0p01.std), file=f)

  with open("./results/O2p_corrected_32I_ml0p004.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p004_jacks.cv, epi2_corrected_32I_ml0p004_jacks.std, O2p_corrected_32I_ml0p004_jacks.cv, O2p_corrected_32I_ml0p004_jacks.std), file=f)

  with open("./results/O2p_corrected_32I_ml0p006.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p006_jacks.cv, epi2_corrected_32I_ml0p006_jacks.std, O2p_corrected_32I_ml0p006_jacks.cv, O2p_corrected_32I_ml0p006_jacks.std), file=f)

  with open("./results/O2p_corrected_32I_ml0p008.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_32I_ml0p008_jacks.cv, epi2_corrected_32I_ml0p008_jacks.std, O2p_corrected_32I_ml0p008_jacks.cv, O2p_corrected_32I_ml0p008_jacks.std), file=f)

  with open("./results/O2p_corrected_24I_ml0p005.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p005_jacks.cv, epi2_corrected_24I_ml0p005_jacks.std, O2p_corrected_24I_ml0p005_jacks.cv, O2p_corrected_24I_ml0p005_jacks.std), file=f)

  with open("./results/O2p_corrected_24I_ml0p01.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_corrected_24I_ml0p01_jacks.cv, epi2_corrected_24I_ml0p01_jacks.std, O2p_corrected_24I_ml0p01_jacks.cv, O2p_corrected_24I_ml0p01_jacks.std), file=f)

  with open("./results/O2p_phys.dat", "w") as f:
    print("{0:.8e} {1:.8e} {2:.8e} {3:.8e}".format(epi2_phys.cv, epi2_phys.std, O2p_phys_jacks.cv, O2p_phys_jacks.std), file=f)

  with open("./results/O2p_extrap.dat", "w") as f:
    for idx in range(0, len(xx)-1):
      print("{0:.8e} {1:.8e} {2:.8e}".format(epi2_xx[idx+1], y2p[idx], dy2p[idx]), file=f)

#####################################################################################################

# Plot O1

if fit_O1:

  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p01_jacks.cv, O1_corrected_24I_ml0p01_jacks.cv, xerr=epi2_corrected_24I_ml0p01_jacks.std, yerr=O1_corrected_24I_ml0p01_jacks.std, marker="o", color=tableau10[0], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="24I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p005_jacks.cv, O1_corrected_24I_ml0p005_jacks.cv, xerr=epi2_corrected_24I_ml0p005_jacks.std, yerr=O1_corrected_24I_ml0p005_jacks.std, marker="o", color=tableau10[0], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p008_jacks.cv, O1_corrected_32I_ml0p008_jacks.cv, xerr=epi2_corrected_32I_ml0p008_jacks.std, yerr=O1_corrected_32I_ml0p008_jacks.std, marker="s", color=tableau10[0], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p006_jacks.cv, O1_corrected_32I_ml0p006_jacks.cv, xerr=epi2_corrected_32I_ml0p006_jacks.std, yerr=O1_corrected_32I_ml0p006_jacks.std, marker="s", color=tableau10[0], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p004_jacks.cv, O1_corrected_32I_ml0p004_jacks.cv, xerr=epi2_corrected_32I_ml0p004_jacks.std, yerr=O1_corrected_32I_ml0p004_jacks.std, marker="s", color=tableau10[0], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="32I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_phys.cv, O1_phys_jacks.cv, xerr=epi2_phys.std, yerr=O1_phys_jacks.std, marker="D", color="k", ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="Physical pt.")
  for cap in caps:
    cap.set_markeredgewidth(2)
  plt.fill_between(epi2_xx[1:], y1-dy1, y1+dy1, color=tableau10[0], alpha=0.5)
  plt.xlabel("$\\epsilon_{\\pi}^{2} = m_{\\pi}^{2} / 8 \\pi^{2} f_{\\pi}^{2}$", fontsize=18)
  plt.ylabel("$\\langle \\pi \\vert \\mathscr{O}_{1+}^{++} \\vert \\pi \\rangle$ (GeV$^{4}$)", fontsize=18)
  plt.legend(loc="upper left", numpoints=1, frameon=True)
  plt.xlim(0.0,0.16)
  plt.ylim(-0.35,-0.10)
  plt.savefig("./plots/O1_extrap.pdf", bbox_inches="tight")
  plt.clf()

#####################################################################################################

# Plot O2

if fit_O2:

  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p01_jacks.cv, O2_corrected_24I_ml0p01_jacks.cv, xerr=epi2_corrected_24I_ml0p01_jacks.std, yerr=O2_corrected_24I_ml0p01_jacks.std, marker="o", color=tableau10[1], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="24I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p005_jacks.cv, O2_corrected_24I_ml0p005_jacks.cv, xerr=epi2_corrected_24I_ml0p005_jacks.std, yerr=O2_corrected_24I_ml0p005_jacks.std, marker="o", color=tableau10[1], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p008_jacks.cv, O2_corrected_32I_ml0p008_jacks.cv, xerr=epi2_corrected_32I_ml0p008_jacks.std, yerr=O2_corrected_32I_ml0p008_jacks.std, marker="s", color=tableau10[1], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p006_jacks.cv, O2_corrected_32I_ml0p006_jacks.cv, xerr=epi2_corrected_32I_ml0p006_jacks.std, yerr=O2_corrected_32I_ml0p006_jacks.std, marker="s", color=tableau10[1], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p004_jacks.cv, O2_corrected_32I_ml0p004_jacks.cv, xerr=epi2_corrected_32I_ml0p004_jacks.std, yerr=O2_corrected_32I_ml0p004_jacks.std, marker="s", color=tableau10[1], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="32I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_phys.cv, O2_phys_jacks.cv, xerr=epi2_phys.std, yerr=O2_phys_jacks.std, marker="D", color="k", ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="Physical pt.")
  for cap in caps:
    cap.set_markeredgewidth(2)
  plt.fill_between(epi2_xx[1:], y2-dy2, y2+dy2, color=tableau10[1], alpha=0.5)
  plt.xlabel("$\\epsilon_{\\pi}^{2} = m_{\\pi}^{2} / 8 \\pi^{2} f_{\\pi}^{2}$", fontsize=18)
  plt.ylabel("$\\langle \\pi \\vert \\mathscr{O}_{2+}^{++} \\vert \\pi \\rangle$ (GeV$^{4}$)", fontsize=18)
  plt.legend(loc="upper left", numpoints=1, frameon=True)
  plt.xlim(0.0,0.16)
  plt.ylim(-0.50,-0.15)
  plt.savefig("./plots/O2_extrap.pdf", bbox_inches="tight")
  plt.clf()

#####################################################################################################

# Plot O3

if fit_O3:

  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p01_jacks.cv, O3_corrected_24I_ml0p01_jacks.cv, xerr=epi2_corrected_24I_ml0p01_jacks.std, yerr=O3_corrected_24I_ml0p01_jacks.std, marker="o", color=tableau10[2], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="24I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p005_jacks.cv, O3_corrected_24I_ml0p005_jacks.cv, xerr=epi2_corrected_24I_ml0p005_jacks.std, yerr=O3_corrected_24I_ml0p005_jacks.std, marker="o", color=tableau10[2], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p008_jacks.cv, O3_corrected_32I_ml0p008_jacks.cv, xerr=epi2_corrected_32I_ml0p008_jacks.std, yerr=O3_corrected_32I_ml0p008_jacks.std, marker="s", color=tableau10[2], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p006_jacks.cv, O3_corrected_32I_ml0p006_jacks.cv, xerr=epi2_corrected_32I_ml0p006_jacks.std, yerr=O3_corrected_32I_ml0p006_jacks.std, marker="s", color=tableau10[2], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p004_jacks.cv, O3_corrected_32I_ml0p004_jacks.cv, xerr=epi2_corrected_32I_ml0p004_jacks.std, yerr=O3_corrected_32I_ml0p004_jacks.std, marker="s", color=tableau10[2], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="32I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_phys.cv, O3_phys_jacks.cv, xerr=epi2_phys.std, yerr=O3_phys_jacks.std, marker="D", color="k", ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="Physical pt.")
  for cap in caps:
    cap.set_markeredgewidth(2)
  plt.fill_between(epi2_xx[1:], y3-dy3, y3+dy3, color=tableau10[2], alpha=0.5)
  plt.xlabel("$\\epsilon_{\\pi}^{2} = m_{\\pi}^{2} / 8 \\pi^{2} f_{\\pi}^{2}$", fontsize=18)
  plt.ylabel("$\\langle \\pi \\vert \\mathscr{O}_{3+}^{++} \\vert \\pi \\rangle$ (GeV$^{4}$)", fontsize=18)
  plt.legend(loc="upper left", numpoints=1, frameon=True)
  plt.xlim(0.0,0.16)
  plt.ylim(0.0,0.0025)
  plt.savefig("./plots/O3_extrap.pdf", bbox_inches="tight")
  plt.clf()

#####################################################################################################

# Plot O1p

if fit_O1p:

  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p01_jacks.cv, O1p_corrected_24I_ml0p01_jacks.cv, xerr=epi2_corrected_24I_ml0p01_jacks.std, yerr=O1p_corrected_24I_ml0p01_jacks.std, marker="o", color=tableau10[3], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="24I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p005_jacks.cv, O1p_corrected_24I_ml0p005_jacks.cv, xerr=epi2_corrected_24I_ml0p005_jacks.std, yerr=O1p_corrected_24I_ml0p005_jacks.std, marker="o", color=tableau10[3], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p008_jacks.cv, O1p_corrected_32I_ml0p008_jacks.cv, xerr=epi2_corrected_32I_ml0p008_jacks.std, yerr=O1p_corrected_32I_ml0p008_jacks.std, marker="s", color=tableau10[3], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p006_jacks.cv, O1p_corrected_32I_ml0p006_jacks.cv, xerr=epi2_corrected_32I_ml0p006_jacks.std, yerr=O1p_corrected_32I_ml0p006_jacks.std, marker="s", color=tableau10[3], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p004_jacks.cv, O1p_corrected_32I_ml0p004_jacks.cv, xerr=epi2_corrected_32I_ml0p004_jacks.std, yerr=O1p_corrected_32I_ml0p004_jacks.std, marker="s", color=tableau10[3], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="32I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_phys.cv, O1p_phys_jacks.cv, xerr=epi2_phys.std, yerr=O1p_phys_jacks.std, marker="D", color="k", ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="Physical pt.")
  for cap in caps:
    cap.set_markeredgewidth(2)
  plt.fill_between(epi2_xx[1:], y1p-dy1p, y1p+dy1p, color=tableau10[3], alpha=0.5)
  plt.xlabel("$\\epsilon_{\\pi}^{2} = m_{\\pi}^{2} / 8 \\pi^{2} f_{\\pi}^{2}$", fontsize=18)
  plt.ylabel("$\\langle \\pi \\vert \\mathscr{O}'_{1+}^{++} \\vert \\pi \\rangle$ (GeV$^{4}$)", fontsize=18)
  plt.legend(loc="upper left", numpoints=1, frameon=True)
  plt.xlim(0.0,0.16)
  plt.ylim(-1.2,-0.3)
  plt.savefig("./plots/O1p_extrap.pdf", bbox_inches="tight")
  plt.clf()

#####################################################################################################

# Plot O2p

if fit_O2p:

  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p01_jacks.cv, O2p_corrected_24I_ml0p01_jacks.cv, xerr=epi2_corrected_24I_ml0p01_jacks.std, yerr=O2p_corrected_24I_ml0p01_jacks.std, marker="o", color=tableau10[4], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="24I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_24I_ml0p005_jacks.cv, O2p_corrected_24I_ml0p005_jacks.cv, xerr=epi2_corrected_24I_ml0p005_jacks.std, yerr=O2p_corrected_24I_ml0p005_jacks.std, marker="o", color=tableau10[4], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p008_jacks.cv, O2p_corrected_32I_ml0p008_jacks.cv, xerr=epi2_corrected_32I_ml0p008_jacks.std, yerr=O2p_corrected_32I_ml0p008_jacks.std, marker="s", color=tableau10[4], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p006_jacks.cv, O2p_corrected_32I_ml0p006_jacks.cv, xerr=epi2_corrected_32I_ml0p006_jacks.std, yerr=O2p_corrected_32I_ml0p006_jacks.std, marker="s", color=tableau10[4], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5)
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_corrected_32I_ml0p004_jacks.cv, O2p_corrected_32I_ml0p004_jacks.cv, xerr=epi2_corrected_32I_ml0p004_jacks.std, yerr=O2p_corrected_32I_ml0p004_jacks.std, marker="s", color=tableau10[4], ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="32I")
  for cap in caps:
    cap.set_markeredgewidth(2)
  (_, caps, _) = plt.errorbar(epi2_phys.cv, O2p_phys_jacks.cv, xerr=epi2_phys.std, yerr=O2p_phys_jacks.std, marker="D", color="k", ls="none", markersize=6, lw=1.0, markeredgecolor="k", markeredgewidth=0.5, label="Physical pt.")
  for cap in caps:
    cap.set_markeredgewidth(2)
  plt.fill_between(epi2_xx[1:], y2p-dy2p, y2p+dy2p, color=tableau10[4], alpha=0.5)
  plt.xlabel("$\\epsilon_{\\pi}^{2} = m_{\\pi}^{2} / 8 \\pi^{2} f_{\\pi}^{2}$", fontsize=18)
  plt.ylabel("$\\langle \\pi \\vert \\mathscr{O}'_{2+}^{++} \\vert \\pi \\rangle$ (GeV$^{4}$)", fontsize=18)
  plt.legend(loc="lower left", numpoints=1, frameon=True)
  plt.xlim(0.0,0.16)
  plt.ylim(0.04,0.14)
  plt.savefig("./plots/O2p_extrap.pdf", bbox_inches="tight")
  plt.clf()
