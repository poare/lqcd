#!/bin/bash

base=/data/d10b/ensembles/isoClover
stem=cl3_16_48_b6p1_m0p2450
exe=/data/d10b/wombat/users/djmurphy/Software/GLU/bin/GLU
writeto=/home/poare/lqcd/gf/testing/${stem}_smeared

export OMP_NUM_THREADS=64

for ((traj=1000; traj<=1020; traj++)); do
  if [ -f ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime ]; then
    echo "--- Smearing ${stem}_cfg_${traj}.lime"
    # $exe -i instout.txt -c ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime
    $exe -i instout.txt -c ${base}/${stem}/cfgs/${stem}_cfg_${traj}.lime -o ${writeto}/${stem}_cfg_${traj}.lime
  fi
done
