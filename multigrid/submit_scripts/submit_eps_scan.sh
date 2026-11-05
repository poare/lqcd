#!/bin/bash
##############################################################################
# One-MD-step epsilon scan (stage 4a of run_hmc_multigrid_quenched_frontier.job):
# does the perturbative update survive a larger delta A?  Each job loads the
# saved basis at U_0, takes ONE MD step of size eps (TrajL eps, MDsteps 1; the
# same seeds give the same momenta at every eps, so delta U ~ eps P), updates
# the basis by the chosen method, rebuilds and solves once.  eps 0.01 is step 1
# of the existing runs (5801401 frozen 64, 5843573 perturbative 59, 5819892
# refresh 60), so it is not resubmitted.
#
#   frozen   Mode recoarsen
#   pert     Mode perturbative, CorrTol 1e-2, batches of 6, CorrMaxRestarts 10
#            (120 GCR steps per batch; 5843573 needed 27 at step 1)
#   refresh  Mode refresh, 3 passes of 10 steps (as 5819892)
#
# No RBCG control: this scan measures staleness after one step, and the
# control's cost barely moves along the trajectory (5833068).
#
# Outer cap 240 (OUTERMAXIT 20) so that counts past 156 show.  Three jobs, one
# srun per (method, eps) inside each (SCAN in the job script), each run from
# the saved basis with the same seeds and its own log
# MG_QHMC_<jobid>_<mode>_eps<eps>.log:
#   eps-frozen-refresh  frozen then refresh at all 5 eps, ~10 min each, ~1.8 h
#   eps-pert-a          perturbative at 0.02, 0.05, 0.1   (worst case ~1 h each)
#   eps-pert-b          perturbative at 0.2, 0.5
# TIMEOUT is per run.
#
# Usage (on a Frontier login node, from this directory):
#   ./submit_eps_scan.sh frozen-refresh|pert-a|pert-b|all|pert-old|pert-recoarsen
# Each submission is appended to eps_scan_jobs.txt as "jobid label settings".
##############################################################################
set -e
cd "$(dirname "$0")"

EPS="0.02 0.05 0.1 0.2 0.5"
COMMON="ALL,MDSTEPS=1,OUTERMAXIT=20"
REFRESH="REFRESHPASSES=3,REFRESHMAXOUTER=1,REFRESHTOL=0.1"
PERT="CORRTOL=1e-2,CORRNRHS=6,CORRMAXRESTARTS=10"

entries () {   # mode eps...  ->  mode:e1/mode:e2/...
  m=$1; shift; s=""
  for e in "$@"; do s="$s${s:+/}$m:$e"; done; echo "$s"
}

submit () {   # label time timeout scan exports
  id=$(sbatch --parsable --time=$2 -J "$1" --export=$COMMON,TIMEOUT=$3,SCAN=$4,$5 run_hmc_multigrid_quenched_frontier.job)
  echo "$id $1 SCAN=$4 $5" | tee -a eps_scan_jobs.txt
}

case "$1" in
  frozen-refresh) submit eps-frozen-refresh 3:00:00 30m "$(entries recoarsen $EPS)/$(entries refresh $EPS)" $REFRESH ;;
  pert-a)         submit eps-pert-a 4:00:00 75m "$(entries perturbative 0.02 0.05 0.1)" $PERT ;;
  pert-b)         submit eps-pert-b 2:30:00 70m "$(entries perturbative 0.2 0.5)" $PERT ;;
  all)            me=./$(basename "$0"); $me frozen-refresh; $me pert-a; $me pert-b ;;
  # 2026-11-05: why the correction stalls at 0.05 and 0.1 (5853853).  pert-old puts
  # A at the last update on the left (what the basis and the V-cycle were built for);
  # pert-recoarsen keeps A(U) and rebuilds the coarse operators at U first.
  pert-old)       submit eps-pert-old 3:00:00 75m "$(entries perturbative 0.05 0.1)" $PERT,CORROPERATOR=old ;;
  pert-recoarsen) submit eps-pert-recoarsen 3:00:00 75m "$(entries perturbative 0.05 0.1)" $PERT,CORRRECOARSEN=1 ;;
  *)              echo "usage: $0 frozen-refresh|pert-a|pert-b|all|pert-old|pert-recoarsen"; exit 1 ;;
esac
