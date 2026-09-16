# Guard against the traceless-source trap.
#
# emt_npr.qlua has two sequential sources: get_sequential_source (plain) at
# line 127 and get_sequential_source_full (trace-subtracted) at line 115. The
# production data used the PLAIN one -- proved from the data itself, since
# sum_mu O_mumu is O(1) rather than zero.
#
# If the port accidentally subtracted the trace it would still pass every Pi
# and Z comparison downstream, and fail the raw O* comparison for a reason
# that looks like a J_mu bug. This names it outright.
#
#   python3 check_trace.py cfg1600_chroma.txt
import sys, numpy as np

got = {}
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) != 11: continue
    tag = p[0]; k = tuple(int(x) for x in p[1:5])
    s0,s1,c0,c1 = (int(x) for x in p[5:9])
    got.setdefault((tag,k), np.zeros((4,4,3,3), dtype=complex))[s0,s1,c0,c1] = float(p[9])+1j*float(p[10])

ks = sorted({k for (t,k) in got if t == "O11"})
assert ks, "no operator entries found"
for k in ks:
    tot = sum(got[(f"O{m}{m}",k)] for m in range(1,5))
    mx  = max(np.abs(got[(f"O{m}{m}",k)]).max() for m in range(1,5))
    r = np.abs(tot).max()/mx
    print(f"k={k}: |sum_mu O_mumu| / max|O_mumu| = {r:.3f}")
    assert r > 0.1, ("sum_mu O_mumu vanishes - the TRACELESS source was used. "
                     "Switch to the plain get_sequential_source equivalent; "
                     "every raw O* will disagree with the reference until you do.")
print("PASS: operators are the plain (non-traceless) ones")
