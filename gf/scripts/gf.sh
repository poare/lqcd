#!/bin/bash

# base=/work/lqcd/d10b/ensembles/isoClover
# stem=cl21_12_24_b6p1_m0p2800m0p2450

base=/work/lqcd/d10b/ensembles/RBC/RBC_UKQCD_24_64/Sea_m0.01
stem=ckpoint_lat.IEEE64BIG

#exe=/data/d10b/wombat/users/djmurphy/Software/GLU/bin/GLU
exe=/home/lqcd/dpefkou/GLU/install/bin/GLU
#exe=/home/dpefkou/GLU/install_gx/bin/GLU

# smeared=${base}/${stem}_smeared
# writeto=${base}/${stem}_smeared_gf

# TODO may have to smear these first
smeared=${base}/Configs
writeto=${base}/gf

# export OMP_NUM_THREADS=64
export OMP_NUM_THREADS=12

for ((traj=1500; traj<=1700; traj++)); do
  if [ -f ${smeared}/${stem}.${traj} ]; then
    echo "--- Gauge fixing ${stem}.${traj}"
  # if [ -f ${smeared}/${stem}-a_cfg_${traj}.lime ]; then
  #   echo "--- Gauge fixing ${stem}-a_cfg_${traj}.lime"
    # $exe -i inlandau.txt -c ${smeared}/${stem}-a_cfg_${traj}.lime -o ${writeto}/landau/${stem}-a_cfg_${traj}.lime
    $exe -i incoulomb.txt -c ${smeared}/${stem}.${traj} -o ${writeto}/coulomb/${stem}.${traj}
  fi
done
