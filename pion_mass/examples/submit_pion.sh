#!/bin/bash

#ulimit -c unlimited

#SBATCH --partition=prod
#SBATCH --job-name=0p113
#SBATCH --nodes=1

set -x

cd /data/wombat/users/mlwagman/mosaic


module load openmpi/3.1.5-gcc.7.3.0
module load qlua/wd

mkdir /home/poare/lqcd/pion_mass/logs/playing_with_pions_p_5

EXE=qlua

for cfg in `seq 100 10 650`
do

    mpirun $EXE -e "cfgnum = $cfg" /home/poare/lqcd/pion_mass/examples/test_pion.qlua > /home/poare/lqcd/pion_mass/logs/playing_with_pions_p_5/cfg${cfg}.txt

done
