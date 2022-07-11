################################################################################
# This script takes h5 files in from different ensembles and zips them all     #
# together into a larger file. This will hopefully make the workflow of        #
# workingw with a number of different ensembles easier.                        #
################################################################################

ensembles = [
    'cl21_12_24_b6p1_m0p2800m0p2450-a',                                         # 12^3, heavy
    'cl21_12_24_b6p3_m0p2416m0p2050-b',                                         # 12^3, light
    'cl21_16_16_b6p3_m0p2416m0p2050-a'                                         # 16^3, light
    # 'cl21_48_96_b6p3_m0p2416_m0p2050'                                           # 48^3, light
]
n_ens = len(ensembles)

paths = [
    '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_106539.h5',
    '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_113030.h5',
    '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_113356.h5'
]
# this one has an updated path for 12^3 heavy, and will want to rerun more stats for 16^4 as well
# paths = [
#     '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_114141.h5',
#     '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_113030.h5',
#     '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_113356.h5'
# ]

zip_data = [[] for ens_idx in range(n_ens)]
all_keys = None
for ii, ens_path in enumerate(paths):
    f = h5py.File(ens_path, 'r')
    if ii = 0:
        all_keys = list(f.keys())
    for key in all_keys:
        zip_data[ii].append(f[key][()])
    f.close()

out_file = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_all_ensembles.h5'
f = h5py.File(out_file, 'w')
for ii, ens in enumerate(ensembles):
    for jj, key in enumerate(all_keys):
        f[ens + '/' + 'key'] = zip_data[ii][jj]
f.close()
print('Done zipping h5 files.')
