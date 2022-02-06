#!/bin/bash

# base=/data/d10b/ensembles/isoClover
#base=/data/d10b/ensembles/cl21_48_96_b6p3_m0p2416_m0p2050
#stem=cl3_16_48_b6p1_m0p2450
#stem=cl3_24_24_b6p1_m0p2450
# stem=cl3_32_48_b6p1_m0p2450
#stem=cl21_48_96_b6p3_m0p2416_m0p2050
stem=cl21_12_24_b6p1_m0p2800m0p2450-a
base=/data/d10b/ensembles/isoClover/cl21_12_24_b6p1_m0p2800m0p2450/cl21_12_24_b6p1_m0p2800m0p2450/Lattices
exe=/data/d10b/wombat/users/djmurphy/Software/GLU/bin/GLU
#exe=/home/dpefkou/glu_tests/GLU
#exe=/home/dpefkou/GLU/install/bin/GLU
#writeto=/home/poare/lqcd/gf/testing/${stem}_smeared
#writeto=/data/d10b/users/poare/gf/${stem}_smeared
#writeto=${base}/${stem}/smeared
writeto=/data/d10b/ensembles/isoClover/cl21_12_24_b6p1_m0p2800m0p2450_smeared

export OMP_NUM_THREADS=64

for ((traj=1000; traj<=9900; traj++)); do
  if [ -f ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime ]; then
    echo "--- Smearing ${stem}_cfg_${traj}.lime"
    # $exe -i instout.txt -c ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime
    $exe -i instout.txt -c ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime -o ${writeto}/${stem}_cfg_${traj}.lime
  fi
done
