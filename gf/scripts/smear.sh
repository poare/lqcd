#!/bin/bash

base=/data/d10b/ensembles/isoClover
stem=cl3_16_48_b6p1_m0p2450
exe=/data/wombat/users/djmurphy/Software/GLU/bin/GLU

export OMP_NUM_THREADS=64

for ((traj=100; traj<=3220; traj++)); do
  if [ -f ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime ]; then
    echo "--- Smearing ${stem}_cfg_${traj}.lime"
    $exe -i instout.txt -c ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime
  fi
done
