#!/bin/bash

##ulimit -c unlimited

#SBATCH --job-name=gq_mixing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100l
#SBATCH -t 24:00:00

set -x

cd /home/lqcd/poare/lqcd/mosaic

bash

# For p-nodes on platypus
# module load openmpi/4.1.1
# module load slurm/19.05.4.1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/software/openmpi-4.1.1/lib/
# EXE=/opt/software/qlua-20200107-gpu1080/qlua/bin/qlua

# For a-nodes on wombat
module load PrgEnv-Wombat
module load PrgEnv-Qlua
module load slurm/19.05.4.1
EXE=qlua


# cfgpath="/work/lqcd/d20b/ensembles/cl21_48_96_b6p3_m0p2416_m0p2050/"
# cfgbase="cl21_48_96_b6p3_m0p2416_m0p2050"

# Test on 16 by 48
cfgpath="/work/lqcd/d20b/ensembles/isoClover/cl3_16_48_b6p1_m0p2450/"
cfgbase="cl3_16_48_b6p1_m0p2450"

# gpu=true

home=/home/lqcd/poare/lqcd/gq_mixing
logs=${home}/logs/girs_${cfgbase}_${SLURM_JOB_ID}
output=/work/lqcd/d20b/users/poare/gq_mixing/girs/${cfgbase}_${SLURM_JOB_ID}

mkdir ${logs}
mkdir ${logs}/no_output
mkdir ${output}

parameters="jobid = ${SLURM_JOB_ID} cfgpath = '${cfgpath}' cfgbase = '${cfgbase}'"
export OMP_NUM_THREADS=6

# for cfg in `seq 1600 10 2200`
# do
cfg=1600
	# mpirun -np 4 $EXE -e "$parameters cfgnum = ${cfg}" ${home}/girs.qlua > ${logs}/cfg${cfg}.txt
    srun --mpi=pmix --ntasks=4 $EXE -e "$parameters cfgnum = ${cfg}" ${home}/girs.qlua > ${logs}/cfg${cfg}.txt
# done
