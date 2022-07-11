import numpy as np
import h5py
import os

parent = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/nnpp/'
stem = 'cl3_32_48_b6p1_m0p2450'
job1 = 113071
job2 = 99998
outjob = 99999
f1_dir = parent + stem + '_' + str(job1)
f2_dir = parent + stem + '_' + str(job2)
out_dir = parent + stem + '_' + str(outjob)

# assemble configs, keys, and unique paths for each file
cfg_list = [1010, 1020, 1030, 1110, 1210, 1220, 1230]
key_list = ['moms', 'prop_k1', 'prop_k2', 'prop_q']
for mu in range(4):
    key_list.extend(['GA' + str(mu + 1), 'GV' + str(mu + 1)])
for n in range(16):
    key_list.extend(['Gn' + str(n)])

# write files
# paths1 = ['q' + str(k) + str(k) + '00' for k in range(1, 7)]
# paths2 = ['q' + str(k) + str(k) + '00' for k in range(7, 11)] 
paths1 = ['q' + str(k) + str(k) + '00' for k in [9]]
paths2 = ['q' + str(k) + str(k) + '00' for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]]
for cfg in cfg_list:
    filename = '/cfg' + str(cfg) + '.h5'
    fout = h5py.File(out_dir + filename, 'w')
    f1 = h5py.File(f1_dir + filename, 'r')
    for key in key_list:
        for path in paths1:
            full_key = key + '/' + path
            if key == 'moms':
                data1 = f1[full_key + '/k1'][()]
                fout[full_key + '/k1'] = data1
                data2 = f1[full_key + '/k2'][()]
                fout[full_key + '/k2'] = data2
            else:
                data = f1[full_key][()]
                fout[full_key] = data
    f1.close()
    f2 = h5py.File(f2_dir + filename, 'r')
    for key in key_list:
        for path in paths2:
            full_key = key + '/' + path
            if key == 'moms':
                data1 = f2[full_key + '/k1'][()]
                fout[full_key + '/k1'] = data1
                data2 = f2[full_key + '/k2'][()]
                fout[full_key + '/k2'] = data2
            else:
                data = f2[full_key][()]
                fout[full_key] = data
    f2.close()
    fout.close()
    print('File ' + out_dir + filename + ' written.')
print('Done merging files')
