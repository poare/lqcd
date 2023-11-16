# This script is meant to help me optimize my code. 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import h5py
import os
import time
import argparse
import itertools
import pandas as pd
import gvar as gv
import lsqfit

# imports from lqcd
import sys
sys.path.append(util_path)
import constants as const
import formattools as ft
import plottools as pt
import suNtools as suN

style = ft.styles['prd_twocol']
pt.set_font()

import rhmc
import rhmc_coeffs as coeffs
from pfapack import pfaffian as pf

##############################################################
######################### Test Force Timings #################
##############################################################

##############################################################
################### Test configuration storage ###############
##############################################################
# it's easier to modularize things as separate files, but want to make sure there's not more overhead 
# if we store them as separate hdf5 files

f = h5py.File('')

