#!/bin/bash

#base=/data/d10b/ensembles/isoClover
base=/data/d10b/ensembles/cl21_48_96_b6p3_m0p2416_m0p2050
#stem=cl3_16_48_b6p1_m0p2450
#stem=cl3_32_48_b6p1_m0p2450
stem=cl21_48_96_b6p3_m0p2416_m0p2050
#exe=/data/d10b/wombat/users/djmurphy/Software/GLU/bin/GLU
exe=/home/dpefkou/GLU/install/bin/GLU
#writeto=/home/poare/lqcd/gf/testing/${stem}_smeared_gf
#smeared=/data/d10b/users/poare/gf/${stem}_smeared
#writeto=/data/d10b/users/poare/gf/${stem}_smeared_gf
smeared=${base}/${stem}/smeared
writeto=${base}/${stem}/smeared_gf

export OMP_NUM_THREADS=64

for ((traj=1000; traj<=9900; traj++)); do
  if [ -f ${smeared}/${stem}_cfg_${traj}.lime ]; then
    echo "--- Gauge fixing ${stem}_cfg_${traj}.lime"
    # $exe -i inlandau.txt -c ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared_gf/landau/${stem}_cfg_${traj}.lime
    # $exe -i incoulomb.txt -c ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared_gf/coulomb/${stem}_cfg_${traj}.lime
    $exe -i inlandau.txt -c ${smeared}/${stem}_cfg_${traj}.lime -o ${writeto}/landau/${stem}_cfg_${traj}.lime
  fi
done
