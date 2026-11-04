#!/bin/bash
##############################################################################
# The A^{-k} ("ak") scan (stage 2g of run_hmc_multigrid_quenched_frontier.job)
# for one MG parameter set and one group of runs per call.  Every run: Mode
# refresh, one pass per MD step, field held fixed (TrajL 2e-4, eps 1e-5), outer
# cap 240 iterations (OUTERMAXIT is in restarts of Outer.Nstep, so it depends on
# the set).
#
# A round is unpreconditioned GCR(10,10) with REFRESHMAXOUTER restarts of 10
# steps, to relative residual REFRESHTOL.  In 5843569 (30 restarts, 5e-4) the
# tolerance never bound: every round ran all 300 steps and ended at residual
# 0.80-0.89.
#
#   base    300 steps at 5e-4 (the setup's rounds) from Gaussian noise
#           (NoiseBasis 1): MD step n is k = n, 0..20
#   short   the same, 3 MD steps at the same eps: k = 0..3 only, about 30 min
#           (k = 3 checks against 5843569's 59 at p0)
#   budget  from noise, 10, 30, 100 steps per round at 5e-4 (the cap binds)
#   tol     from noise, 300 steps at tolerance 0.5, 0.1, 1e-2.  In the setup
#           round 1 reached 2e-3 and round 2 0.29 in 300 steps, so a looser
#           tolerance mostly shortens the first rounds
#
# MG sets (see the job script): p0 (every run so far) or dense3 (the defaults
# of Example_pvdagm_mrhs_3level_dense.cc).
#
# Usage (on a Frontier login node, from this directory):
#   ./submit_ak_scan.sh p0|dense3 base|short|budget|tol
# Each submission is appended to ak_scan_jobs.txt as "jobid label settings".
# Rough wall time at p0: 3.3 h at 300 steps per round, under 1.5 h at 10-30.
##############################################################################
set -e
cd "$(dirname "$0")"

MGSET=$1
case $MGSET in
  p0)     OUTERMAXIT=20 ;;   # 20 x 12 = 240
  dense3) OUTERMAXIT=30 ;;   # 30 x 8  = 240
  *)      echo "usage: $0 p0|dense3 base|short|budget|tol"; exit 1 ;;
esac
COMMON="ALL,MGSET=$MGSET,MODE=refresh,REFRESHPASSES=1,OUTERMAXIT=$OUTERMAXIT"

submit () {   # label noisebasis maxouter tol [mdsteps], eps 1e-5 throughout
  label="ak-$MGSET-$1"
  md=${5:-20}
  trajl=$(awk "BEGIN{print $md*1e-5}")
  id=$(sbatch --parsable -J "$label" --export=$COMMON,NOISEBASIS=$2,REFRESHMAXOUTER=$3,REFRESHTOL=$4,MDSTEPS=$md,TRAJL=$trajl run_hmc_multigrid_quenched_frontier.job)
  echo "$id $label MGSET=$MGSET NOISEBASIS=$2 REFRESHMAXOUTER=$3 REFRESHTOL=$4 MDSTEPS=$md TRAJL=$trajl" | tee -a ak_scan_jobs.txt
}

case "$2" in
  base)   submit noise-s300-t5e-4  1 30 5e-4 ;;
  short)  submit noise-s300-t5e-4-k0to3 1 30 5e-4 3 ;;
  budget) submit noise-s10-t5e-4   1 1  5e-4
          submit noise-s30-t5e-4   1 3  5e-4
          submit noise-s100-t5e-4  1 10 5e-4 ;;
  tol)    submit noise-s300-t0.5   1 30 0.5
          submit noise-s300-t0.1   1 30 0.1
          submit noise-s300-t1e-2  1 30 1e-2 ;;
  *)      echo "usage: $0 p0|dense3 base|short|budget|tol"; exit 1 ;;
esac
