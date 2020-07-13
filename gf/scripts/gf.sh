#!/bin/bash

base=/data/d10b/ensembles/isoClover
stem=cl3_16_48_b6p1_m0p2450
exe=/data/d10b/wombat/users/djmurphy/Software/GLU/bin/GLU

export OMP_NUM_THREADS=64

for ((traj=1000; traj<=3220; traj++)); do
  if [ -f ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime ]; then
    echo "--- Gauge fixing ${stem}_cfg_${traj}.lime"
    # $exe -i inlandau.txt -c ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared_gf/landau/${stem}_cfg_${traj}.lime
    $exe -i incoulomb.txt -c ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared_gf/coulomb/${stem}_cfg_${traj}.lime
  fi
done
