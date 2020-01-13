#!/bin/bash

#ulimit -c unlimited

#SBATCH --partition=prod
#SBATCH --job-name=0p113
#SBATCH --nodes=1

set -x

cd /data/wombat/users/mlwagman/mosaic


module load openmpi/3.1.5-gcc.7.3.0
module load qlua/wd

EXE=qlua

for cfg in `seq 100 10 110`
do

    mpirun $EXE -e "cfgnum = $cfg" /data/d10a/projects/playingwithpions/test_pion.qlua

done
