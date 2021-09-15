#!/bin/bash
#----------------------------------------------------
# Example Slurm job script
# for TACC Stampede2 KNL nodes
#
#   *** Hybrid Job on Normal Queue ***
#
#       This sample script specifies:
#         10 nodes (capital N)
#         40 total MPI tasks (lower case n); this is 4 tasks/node
#         16 OpenMP threads per MPI task (64 threads per node)
#
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch knl.hybrid.slurm" on Stampede2 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do not use mpirun or mpiexec.
#
#   -- In most cases it's best to specify no more
#      than 64-68 MPI ranks or independent processes
#      per node, and 1-2 threads/core.
#
#   -- If you're running out of memory, try running
#      fewer tasks and/or threads per node to give each
#      process access to more memory.
#
#   -- IMPI and MVAPICH2 both do sensible process pinning by default.
#
#----------------------------------------------------

#SBATCH -J gauge_fix                      # Job name
#SBATCH -o gauge_fix.o%j                  # Name of stdout output file
#SBATCH -e gauge_fix.e%j                  # Name of stderr error file
#SBATCH -p prod                      # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of mpi tasks
#SBATCH --exclusive
#SBATCH -t 48:00:00                  # Run time (hh:mm:ss)
###SBATCH --mail-user=poare@mit.edu
###SBATCH --mail-type=all            # Send email at begin and end of job

module load openmpi/4.0.2
echo $(date)

set -x
export I_MPI_SHM_LMT=shm
export LD_LIBRARY_PATH=/data/d10b/wombat/users/djmurphy/Software/cps4/lib:/data/d10b/wombat/users/djmurphy/Software/grid4/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=64

# base_dir=/data/d10b/wombat/users/djmurphy/Software/GLU
base_dir=/home/poare/lqcd/gf
run_dir=${base_dir}/scripts
now=$(date '+%Y%m%d%H%M%S')
log_file=${base_dir}/logs/log-${now}.out
run_file=gf.sh
#run_file=smear.sh

cd ${run_dir}
${run_dir}/${run_file} | tee ${log_file}
