#!/usr/bin/env python3
"""Compact log of an Example_hmc_multigrid_quenched_frontier run.

A full QHMC log is 0.4-1.1 GB, almost all of it per-iteration solver output.
This keeps what the analysis uses, as lines in the log's own
"Grid : Message : <t> s : TAG key value ..." format, so the same readers parse both.

Kept verbatim:
  - every QHMC-* line (SOLVE, MEM, CORRECT, STEP, TRAJ, SETUP, SUBSPACE,
    MATVECS, CONTROL, ...) and the job script's STEP / END STEP markers
  - checks and certificates: FINAL true residuals, GALERKIN, GUARD
    orthonormality, dense-bottom IMPORT CERTIFICATE and VERIFY, L3 dense
  - rebuild and setup timings: CoarsenOperator (incl. the Fourier condition
    number), DenseCoarseMatrix (not the per-apply apply6D lines),
    BlockCyclicSchurInverse, resident device memory, PVdagMMultiGridSolver totals
  - the integrator: every "Grid : Integrator" line (dt, force average/max,
    Fdt average/max) and the P-update time
  - basis generation: noise[i], filtered[i] and intermediate[i] lines
  - the RBCG control's ConjugateGradient lines, the GPU used/free summary of
    each memory-manager dump, Grid Finalize
  - failures: hipMalloc, WATCHDOG, faults, Grid errors, failed checksums
    (at most MAX_ERRORS lines)

Synthesized:
  QHMC-OUTER      one per outer MG solve (Fouter): n, iters (last outer step),
                  converged (1 if it reported convergence), resid and target of
                  the last step, vcycles and coarse_steps (mean coarse GCR steps
                  per coarse solve, i.e. per V-cycle), seconds
  QHMC-HIST-OUTER the same solve's residual at every outer step, comma-separated
  QHMC-HIST-CORR  one per perturbative-correction batch (Fcorrect): its target
                  (CorrTol |b|) and residual at every GCR step, just before that batch's QHMC-CORRECT line
  QHMC-ROUNDS     one per refresh pass over the basis (a pass starts at vector
                  0's refine-in), and one for the setup's rounds: n rounds, GCR(10,10) steps and final residual,
                  and |<v|Op|v>| before (vopv_in) and after (vopv_out) each round
                  over the vectors, as mean / min / max
  QHMC-TESTROUNDS the same, for each round of the diagnostics' test vectors
                  (between QHMC-TESTGEN begin and end; v4)
  QHMC-CORRGCR    the same, for the plain GCR solves of one correction batch
                  (CorrPrecon none), just before its QHMC-CORRECT line (v4)
  QHMC-GENROUND   before each setup "intermediate[" line: the final GCR
                  residual of the round it closes
  QHMC-IOCHECK    at the end: checksums verified ok and failed

All resid and target values are as MrhsPGCR prints them (MrhsMultiGrid.h:109-125):
SQUARED norms, resid = |r|^2 and target = tol^2 |b|^2, so the relative residual
is |r|/|b| = tol * sqrt(resid / target), with the solver's tol (Outer.Tol, CorrTol).

Dropped: per-iteration lines of the smoothers, the coarse solver and the round
GCRs (summarized above), performance timings (PaddedCell, apply6D, staples),
the rest of the memory-manager dumps, I/O throughput, solver banners.

Usage: qhmc_compact.py LOG [LOG ...] [--outdir DIR]
Writes DIR/<log stem>_compact.log, DIR defaulting to <log dir>/mghmc_compact.
Python 3.6 and the standard library only, so it runs on Frontier as is.
"""
import os
import re
import sys

VERSION = 4
MAX_ERRORS = 200
KEEP = ("QHMC-", "===== STEP", "===== END STEP", "FINAL Nrhs", "GALERKIN", "intermediate[", "Grid Finalize",
        "GUARD:", "CERTIFICATE", "VERIFY", "L3 dense", "CoarsenOperator", "DenseCoarseMatrix:",
        "BlockCyclicSchurInverse:", "resident device memory", "Grid : Integrator", "P update elapsed",
        "noise   [", "filtered[", "ConjugateGradient", "MemoryManager : GPU used")
KEEP_NOT = ("apply6D",)
FAIL = ("hipMalloc failed", "WATCHDOG", "Bus error", "Memory access fault", "NO_TRANSLATION", "CANCELLED",
        "Grid : Error", "Assertion", "GRID_ASSERT", "Segmentation fault", "Aborted")

R_TIME = re.compile(r"Message : ([\d.]+) s : ")
R_FOUTER = re.compile(r"Fouter MrhsPGCR step\[(\d+)\]\s+resid (\S+) target (\S+)")
R_FCONV = re.compile(r"Fouter MrhsPGCR: Converged on iteration (\d+)")
R_FCORR = re.compile(r"Fcorrect MrhsPGCR step\[(\d+)\]\s+resid (\S+) target (\S+)")
R_COUTER = re.compile(r"Couter PGCR step\[(\d+)\]")
R_ROUND = re.compile(r"PGCR\(10,10\) (\d+) steps cp = (\S+)")
R_REFINE = re.compile(r"refine-(in|out)\s*\[(\d+)\] <v\|Op\|v> \(([^,]+),([^)]+)\)")
R_INTER = re.compile(r"intermediate\[(\d+)\]")
R_CHECKSUM = re.compile(r"scidacChecksumVerify computed (\S+) expected (\S+)")


def _stats(xs):
    return sum(xs) / len(xs), min(xs), max(xs)


def _hist(xs):
    return ",".join("%.4g" % x for x in xs)


class Compactor:
    def __init__(self, out):
        self.out = out
        self.t = 0.0
        self.n_outer = 0
        self.outer = None        # the outer solve in progress
        self.coarse = []         # coarse solves since the last outer/correction step: steps each
        self.cur_coarse = None
        self.rounds = []         # (steps, resid) since the last ROUNDS line
        self.round_tag = "QHMC-ROUNDS"   # QHMC-TESTROUNDS between QHMC-TESTGEN begin and end
        self.round_cur = None
        self.last_round_resid = float("nan")
        self.vopv = {"in": [], "out": []}
        self.n_corr = 0
        self.corr = None         # residual history of the correction batch in progress
        self.corr_target = float("nan")
        self.io_ok = 0
        self.io_bad = 0
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
        self.emit("QHMC-HIST-OUTER n %d resid %s" % (o["n"], _hist(o["hist"])))
        self.outer = None

    def flush_corr(self):
        if self.corr:
            self.n_corr += 1
            self.emit("QHMC-HIST-CORR n %d steps %d target %.6g resid %s"
                      % (self.n_corr, len(self.corr), self.corr_target, _hist(self.corr)))
        self.corr = None

    def flush_rounds(self, tag=None):
        if self.round_cur is not None:      # a round that never reported its end
            self.rounds.append(self.round_cur)
            self.round_cur = None
        if not self.rounds:
            return
        s = _stats([r[0] for r in self.rounds])
        r = _stats([r[1] for r in self.rounds])
        body = ((tag or self.round_tag) + " n %d steps_mean %.6g steps_min %d steps_max %d resid_mean %.6g resid_min %.6g resid_max %.6g"
                % (len(self.rounds), s[0], s[1], s[2], r[0], r[1], r[2]))
        for k in ("in", "out"):
            if self.vopv[k]:
                v = _stats(self.vopv[k])
                body += " vopv_%s_mean %.6g vopv_%s_min %.6g vopv_%s_max %.6g" % (k, v[0], k, v[1], k, v[2])
        self.emit(body)
        self.rounds = []
        self.vopv = {"in": [], "out": []}

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
                self.outer = dict(n=self.n_outer, t0=self.t, coarse=[], hist=[])
            if self.outer is not None:
                self.outer.update(step=step, resid=float(m.group(2)), target=float(m.group(3)))
                self.outer["hist"].append(float(m.group(2)))
                self.outer["coarse"] += self.coarse
            self.coarse = []
            return
        m = R_FCORR.search(line)
        if m:                                   # V-cycles of the perturbative correction: not the outer solve's
            self.flush_outer()                  # a capped solve ends without a convergence line
            self.end_coarse()
            self.coarse = []
            if int(m.group(1)) == 1:
                self.flush_corr()
                self.corr = []
                self.corr_target = float(m.group(3))
            if self.corr is not None:
                self.corr.append(float(m.group(2)))
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
        m = R_REFINE.search(line)
        if m:
            if m.group(1) == "in" and m.group(2) == "0":
                self.flush_rounds()             # vector 0 starts a new pass: one ROUNDS line per pass
            self.vopv[m.group(1)].append(abs(complex(float(m.group(3)), float(m.group(4)))))
            return
        m = R_CHECKSUM.search(line)
        if m:
            if m.group(1) == m.group(2):
                self.io_ok += 1
                return
            self.io_bad += 1
        if any(k in line for k in KEEP) and not any(k in line for k in KEEP_NOT):
            if "QHMC-SOLVE" in line or "QHMC-CONTROL" in line or "QHMC-TRAJ" in line or "===== END STEP" in line:
                self.flush_outer()          # a capped solve ends without a convergence line
            if "QHMC-CORRECT" in line:
                self.flush_corr()
                self.flush_rounds("QHMC-CORRGCR")   # CorrPrecon none: the batch's plain GCR solves
            if "QHMC-TESTGEN begin" in line:
                self.flush_rounds()
                self.round_tag = "QHMC-TESTROUNDS"
            if "QHMC-TESTGEN end" in line:
                self.flush_rounds()
                self.round_tag = "QHMC-ROUNDS"
            m = R_INTER.search(line)
            if m:
                self.emit("QHMC-GENROUND round %s gcr_resid %.6g" % (m.group(1), self.last_round_resid))
            self.out.write(line)
            return
        if (any(k in line for k in FAIL) or R_CHECKSUM.search(line)) and self.n_err < MAX_ERRORS:
            self.n_err += 1
            self.out.write(line)

    def finish(self):
        self.flush_outer()
        self.flush_corr()
        self.flush_rounds()
        self.emit("QHMC-IOCHECK checksums_ok %d checksums_failed %d" % (self.io_ok, self.io_bad))


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
