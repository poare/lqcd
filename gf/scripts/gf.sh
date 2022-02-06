#!/bin/bash

#base=/data/d10b/ensembles/cl21_48_96_b6p3_m0p2416_m0p2050
#stem=cl21_48_96_b6p3_m0p2416_m0p2050

base=/work/lqcd/d10b/ensembles/isoClover
stem=cl21_12_24_b6p1_m0p2800m0p2450

#exe=/data/d10b/wombat/users/djmurphy/Software/GLU/bin/GLU
exe=/home/lqcd/dpefkou/GLU/install/bin/GLU
#exe=/home/dpefkou/GLU/install_gx/bin/GLU

#smeared=${base}/${stem}/smeared
#writeto=${base}/${stem}/smeared_gf

smeared=${base}/${stem}_smeared
writeto=${base}/${stem}_smeared_gf

# export OMP_NUM_THREADS=64
export OMP_NUM_THREADS=12

for ((traj=1000; traj<=9900; traj++)); do
  if [ -f ${smeared}/${stem}-a_cfg_${traj}.lime ]; then
    echo "--- Gauge fixing ${stem}-a_cfg_${traj}.lime"
#  if [ -f ${smeared}/${stem}_cfg_${traj}.lime ]; then
#    echo "--- Gauge fixing ${stem}_cfg_${traj}.lime"
    # $exe -i inlandau.txt -c ${base}/${stem}_smeared/${stem}_cfg_${traj}.lime -o ${base}/${stem}_smeared_gf/landau/${stem}_cfg_${traj}.lime
    # $exe -i inlandau.txt -c ${smeared}/${stem}_cfg_${traj}.lime -o ${writeto}/landau/${stem}_cfg_${traj}.lime
    $exe -i inlandau.txt -c ${smeared}/${stem}-a_cfg_${traj}.lime -o ${writeto}/landau/${stem}-a_cfg_${traj}.lime
  fi
done
