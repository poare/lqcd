#!/usr/bin/env python3
"""Compact log of an Example_hmc_multigrid_quenched_frontier run.

A full QHMC log is 0.4-1.1 GB, almost all of it per-iteration solver output.
This keeps what the analysis uses, as "key value" lines in the log's own
"Grid : Message : <t> s : TAG ..." format, so the same readers parse both:

  verbatim   every QHMC-* line (SOLVE, MEM, CORRECT, STEP, TRAJ, SETUP,
             SUBSPACE, MATVECS, CONTROL, ...), the job script's STEP / END STEP
             markers, FINAL true-residual checks, GALERKIN certificates, setup
             "intermediate[" lines, Grid Finalize, and failure lines (hipMalloc,
             WATCHDOG, faults, Grid errors; at most MAX_ERRORS)
  QHMC-OUTER one per outer MG solve (Fouter): iters (last outer step), converged
             (1 if it reported convergence), resid and target of the last step
             (target = tol |b|), vcycles (coarse solves seen during it) and
             coarse_steps (mean coarse GCR steps per V-cycle), seconds
  QHMC-ROUNDS one per block of inverse-iteration rounds (refresh passes, setup
             rounds): n rounds, GCR(10,10) steps and final residual, mean / min / max
  QHMC-GENROUND before each setup "intermediate[" line: the final GCR residual
             of the round it closes

Usage: qhmc_compact.py LOG [LOG ...] [--outdir DIR]
Writes DIR/<log stem>_compact.log, DIR defaulting to <log dir>/mghmc_compact.
Python 3.6 and the standard library only, so it runs on Frontier as is.
"""
import os
import re
import sys

VERSION = 1
MAX_ERRORS = 200
KEEP = ("QHMC-", "===== STEP", "===== END STEP", "FINAL Nrhs", "GALERKIN", "intermediate[", "Grid Finalize")
FAIL = ("hipMalloc failed", "WATCHDOG", "Bus error", "Memory access fault", "NO_TRANSLATION", "CANCELLED",
        "Grid : Error", "Assertion", "GRID_ASSERT", "Segmentation fault", "Aborted")

R_TIME = re.compile(r"Message : ([\d.]+) s : ")
R_FOUTER = re.compile(r"Fouter MrhsPGCR step\[(\d+)\]\s+resid (\S+) target (\S+)")
R_FCONV = re.compile(r"Fouter MrhsPGCR: Converged on iteration (\d+)")
R_COUTER = re.compile(r"Couter PGCR step\[(\d+)\]")
R_ROUND = re.compile(r"PGCR\(10,10\) (\d+) steps cp = (\S+)")
R_INTER = re.compile(r"intermediate\[(\d+)\]")


def _stats(xs):
    return sum(xs) / len(xs), min(xs), max(xs)


class Compactor:
    def __init__(self, out):
        self.out = out
        self.t = 0.0
        self.n_outer = 0
        self.outer = None        # the outer solve in progress
        self.coarse = []         # coarse solves since the last outer/correction step: steps each
        self.cur_coarse = None
        self.rounds = []         # (steps, resid) since the last ROUNDS line
        self.round_cur = None
        self.last_round_resid = float("nan")
        self.n_err = 0

    def emit(self, body):
        self.out.write("Grid : Message : %.6f s : %s\n" % (self.t, body))

    def end_coarse(self):
        if self.cur_coarse is not None:
            self.coarse.append(self.cur_coarse)
            self.cur_coarse = None

    def flush_outer(self, converged=None):
        o = self.outer
        if o is None:
            return
        cs = o["coarse"]
        self.emit("QHMC-OUTER n %d iters %d converged %d resid %.6g target %.6g vcycles %d coarse_steps %.4g seconds %.6g"
                  % (o["n"], o["step"], 1 if converged else 0, o["resid"], o["target"], len(cs),
                     sum(cs) / len(cs) if cs else float("nan"), self.t - o["t0"]))
        self.outer = None

    def flush_rounds(self):
        if self.round_cur is not None:      # a round that never reported its end
            self.rounds.append(self.round_cur)
            self.round_cur = None
        if not self.rounds:
            return
        s = _stats([r[0] for r in self.rounds])
        r = _stats([r[1] for r in self.rounds])
        self.emit("QHMC-ROUNDS n %d steps_mean %.6g steps_min %d steps_max %d resid_mean %.6g resid_min %.6g resid_max %.6g"
                  % (len(self.rounds), s[0], s[1], s[2], r[0], r[1], r[2]))
        self.rounds = []

    def line(self, line):
        m = R_TIME.search(line)
        if m:
            self.t = float(m.group(1))
        # coarse solves (one per V-cycle), attributed to the next outer step
        if "Couter PGCR nStep(" in line:
            self.end_coarse()
            self.cur_coarse = 0
            return
        m = R_COUTER.search(line)
        if m:
            self.cur_coarse = int(m.group(1))
            return
        m = R_FOUTER.search(line)
        if m:
            self.end_coarse()
            step = int(m.group(1))
            if step == 1:
                self.flush_outer()
                self.flush_rounds()
                self.n_outer += 1
                self.outer = dict(n=self.n_outer, t0=self.t, coarse=[])
            if self.outer is not None:
                self.outer.update(step=step, resid=float(m.group(2)), target=float(m.group(3)))
                self.outer["coarse"] += self.coarse
            self.coarse = []
            return
        if "Fcorrect MrhsPGCR step[" in line:   # V-cycles of the perturbative correction: not the outer solve's
            self.flush_outer()                  # a capped solve ends without a convergence line
            self.end_coarse()
            self.coarse = []
            return
        m = R_FCONV.search(line)
        if m and self.outer is not None:
            self.outer["step"] = int(m.group(1))
            self.flush_outer(converged=True)
            return
        # inverse-iteration rounds: the last "PGCR(10,10) N steps cp" before the round ends
        m = R_ROUND.search(line)
        if m:
            self.flush_outer()
            self.round_cur = (int(m.group(1)), float(m.group(2)))
            self.last_round_resid = float(m.group(2))
            return
        if "Level 1" in line and ("did not converge" in line or "PGCR: Converged" in line):
            if self.round_cur is not None:
                self.rounds.append(self.round_cur)
                self.round_cur = None
            return
        if any(k in line for k in KEEP):
            if "QHMC-SOLVE" in line or "QHMC-CONTROL" in line or "QHMC-TRAJ" in line or "===== END STEP" in line:
                self.flush_outer()          # a capped solve ends without a convergence line
            m = R_INTER.search(line)
            if m:
                self.emit("QHMC-GENROUND round %s gcr_resid %.6g" % (m.group(1), self.last_round_resid))
            self.out.write(line)
            return
        if any(k in line for k in FAIL) and self.n_err < MAX_ERRORS:
            self.n_err += 1
            self.out.write(line)

    def finish(self):
        self.flush_outer()
        self.flush_rounds()


def compact_path(log, outdir=None):
    log = os.path.abspath(log)
    d = outdir or os.path.join(os.path.dirname(log), "mghmc_compact")
    stem = os.path.basename(log)
    if stem.endswith(".log"):
        stem = stem[:-4]
    return os.path.join(d, stem + "_compact.log")


def compact(log, outdir=None):
    dest = compact_path(log, outdir)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    n = 0
    with open(log, errors="replace") as f, open(tmp, "w") as out:
        out.write("# qhmc_compact v%d source %s\n" % (VERSION, os.path.basename(log)))
        c = Compactor(out)
        for line in f:
            n += 1
            c.line(line)
        c.finish()
        out.write("# end source_lines %d source_bytes %d\n" % (n, os.path.getsize(log)))
    os.replace(tmp, dest)
    return dest


def main(argv):
    outdir, logs, i = None, [], 0
    while i < len(argv):
        if argv[i] == "--outdir":
            outdir = argv[i + 1]
            i += 2
        else:
            logs.append(argv[i])
            i += 1
    if not logs:
        sys.exit(__doc__)
    for log in logs:
        dest = compact(log, outdir)
        print("%s -> %s (%d bytes)" % (log, dest, os.path.getsize(dest)))


if __name__ == "__main__":
    main(sys.argv[1:])
