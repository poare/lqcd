#!/bin/bash

#ulimit -c unlimited

#PBS -N 0vbb
#PBS -l nodes=1:ppn=4:gpus=4
#PBS -l walltime=24:00:00
#PBS -l cput=24:00:00
#PBS -j oe

echo $(date)

set -x

PBS_O_WORKDIR=/data/wombat/users/djmurphy/Software/cps/src/measurement_package/djm
exe=/data/wombat/users/djmurphy/Software/cps/src/measurement_package/djm/binaries/NOARCH.x

export LD_LIBRARY_PATH=/opt/software/openmpi-2.1.1/lib:/opt/software/cuda-9.0/lib64:$LD_LIBRARY_PATH
export QUDA_RESOURCE_PATH=$PBS_O_WORKDIR/work
export OMP_NUM_THREADS=6
export QUDA_ENABLE_P2P=1

cd $PBS_O_WORKDIR/scripts
now=$(date '+%Y%m%d%H%M%S')
log_file=$PBS_O_WORKDIR/scripts/logs/log-${now}.out
/opt/software/openmpi-2.1.1/bin/mpirun -np 4 $exe ../vmls quda_arg.vml -qmp-geom 1 1 1 4 | tee $log_file

echo $(date)
