import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
# job_num = 16583
# job_num = 16636
# job_num = 16677
# job_num = 16872
# job_num = 16954
job_num = 17951

# Momentum the propagators are run at.
# prop_mom_list = [[0, 0, 0, 0], [2, 2, 2, 2], [-2, -2, -2, -2], [2, 2, -2, -2], [-2, -2, 2, 2]]
# prop_mom_list = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
# prop_mom_list = ['point source at origin']

# Sink momentum that we tie up at. Uncomment when done testing
# mom_tot = []
# for i in range(1, 6):
#     for j in range(1, 6):
#         for k in range(1, 6):
#             for l in range(1, 6):
#                 mom_tot.append([i, j, k, l])
# mom_list = analysis.cylinder(mom_tot, 2)

# for job 17951, delete after testing is done
mom_list = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]
#############################################################################
# print('Propagator momenta: ' + str(prop_mom_list))
print('Sink momenta: ' + str(mom_list))

mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
# analysis.prop_mom_list = prop_mom_list
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list

out_dir = '/Users/theoares/lqcd/npr_momfrac/analysis_output/job' + str(job_num)
data_dir = './output/' + cfgbase + '_' + str(job_num)

# mu, sigma = analysis.run_analysis_single_momenta(data_dir)
mu, sigma, pt_list = analysis.run_analysis_point_sources(data_dir)

os.mkdir(out_dir)
np.save(out_dir + '/mu.npy', mu)
np.save(out_dir + '/sigma.npy', sigma)
np.save(out_dir + '/mom_list.npy', mom_list)
# np.save(out_dir + '/prop_mom_list.npy', prop_mom_list)
np.save(out_dir + '/pt_list.npy', pt_list)
np.save(out_dir + '/cfgnum.npy', analysis.get_num_cfgs())

print('Output saved in directory: ' + out_dir)
