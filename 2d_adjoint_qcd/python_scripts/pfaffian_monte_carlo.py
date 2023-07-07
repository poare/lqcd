################################################################################
# Monte Carlo implementation to see what the Pfaffian of the Dirac operator    #
# D will look like. Note that configurations will be sampled randomly, and     #
# there is a chance that exceptional configurations that make the Pfaffian     #
# appear ill-determined do not really come up in the path integral with any    # 
# non-zero measure.                                                            #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

n_boot = 100
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix
from scipy.linalg import block_diag
import h5py
import os
import itertools
import pandas as pd
import gvar as gv
import lsqfit

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from constants import *
from fittools import *
from formattools import *
import plottools as pt

style = styles['prd_twocol']
pt.set_font()

import rhmc


