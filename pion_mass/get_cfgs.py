from os import listdir
from os.path import isfile, join

path = "/Users/poare/lqcd/pion_mass/"
cfgbase = "cl3_16_48_b6p1_m0p2450"
folder = path + cfgbase + "/cfgs/"

print('Getting ids from directory: ' + folder)
filenames = [f for f in listdir(folder) if isfile(join(folder, f))]

cfgIds = []
for file in filenames:
    # assume filename format is '...cfg_idx.lime'
    after_cfg = file.split('cfg_')[1]
    idx = after_cfg.split('.lime')[0]
    cfgIds.append(idx + "\n")

# change to '/home/poare/pion_mass/cfgIds' once on wombat
write_to = '/Users/poare/lqcd/pion_mass/cfgIds/' + cfgbase \
                + '/config_ids.txt'

writer = open(write_to, "w")
writer.writelines(cfgIds)
writer.close()
print('Wrote ids to directory: ' + write_to)
