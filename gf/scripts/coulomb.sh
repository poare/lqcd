#! /bin/bash

#ulimit -c unlimited

#PBS -N GLU
#PBS -l nodes=1:ppn=1
#PBS -l naccesspolicy=SINGLEJOB -n
#PBS -l walltime=576:00:00
#PBS -l cput=576:00:00
#PBS -j oe

echo $(date)

set -x

export LD_LIBRARY_PATH=/data/wombat/users/djmurphy/Software/cps/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

PBS_O_WORKDIR=/data/wombat/users/djmurphy/Software/GLU/logs
now=$(date '+%Y%m%d%H%M%S')
log_file=${PBS_O_WORKDIR}/log-${now}.out

cfg=5000

## write out stdout file
exec > ${PBS_O_WORKDIR}/${PBS_JOBNAME}.${cfg}.out 2>&1

echo "job commencing"

cd ${PBS_O_WORKDIR}

## function for existance
function exists {
    if [ test -a $@ ]; then
        echo "---->" $@ found "<----"
    else
        echo "---->" $@ not found "<----"
        exit 1
    fi
}

## Binary file we are using - Use the openMP'd fftw one!
exe=/data/wombat/users/djmurphy/Software/GLU/bin/GLU

## All important input_file 
input=/data/wombat/users/djmurphy/Software/GLU/scripts/incoulomb.txt

## Configuration file ...
cfg=/data/d10a/old-dxx/d01/isoClover/cl3_32_48_b6p1_m0p2450/cfgs/cl3_32_48_b6p1_m0p2450_cfg_${cfg}.lime

## Output file 
gfcfg=/data/wombat/users/djmurphy/Lattices/GaugeFixed/cl3_32_48_b6p1_m0p2450_cfg_${cfg}.gauge_fixed_2.lime

## check the input and configuration files exist ...
exists $exe
exists $input
exists $cfg

## want as much threading as possible
export OMP_NUM_THREADS=24

echo "executing with $OMP_NUM_THREADS threads"

## execution command
$exe -i $input -c $cfg -o $gfcfg

echo "Job end"
