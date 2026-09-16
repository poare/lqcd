# Compare the Chroma port's text output against the QLUA reference HDF5.
#
#   python3 compare_cfg1600.py <chroma_txt> <ref_h5> <y_group> [tag,tag,...]
#
# The reference layout is <tag>/<y_group>/p<k0><k1><k2><k3>/cfg<n>, a scalar
# dataset of compound type (4,4) of (3,3) complex -- spin outer, colour inner,
# the same order the port writes.
#
# Momentum keys are bare %d concatenation, so p(-1,-1,-1,-1) is "p-1-1-1-1".
# Ambiguous to read, but deterministic to write, and it is what QLUA wrote.
import sys, numpy as np, h5py

chroma_txt, ref_h5, ygrp = sys.argv[1], sys.argv[2], sys.argv[3]
tags = (sys.argv[4].split(",") if len(sys.argv) > 4 else ["prop"])

got = {}
for line in open(chroma_txt):
    p = line.split()
    if len(p) != 11: continue
    tag = p[0]; k = tuple(int(x) for x in p[1:5])
    s0,s1,c0,c1 = (int(x) for x in p[5:9])
    got.setdefault((tag,k), np.zeros((4,4,3,3), dtype=complex))[s0,s1,c0,c1] = float(p[9])+1j*float(p[10])

def kstr(k): return "p" + "".join(str(x) for x in k)

TOL = 1e-10

f = h5py.File(ref_h5, "r")
worst = 0.0; failed = False; n = 0
for tag in tags:
    for (t,k) in sorted(got):
        if t != tag: continue
        grp = f[tag][ygrp][kstr(k)]
        cfg = list(grp.keys())[0]
        ref = np.array(grp[cfg][()].tolist())          # (4,4,3,3)
        a = np.abs(got[(t,k)] - ref).max()
        r = a / max(np.abs(ref).max(), 1e-300)
        worst = max(worst, r); n += 1
        flag = "" if r < TOL else "   <-- MISMATCH"
        print(f"{tag:5s} {kstr(k):12s} abs {a:.3e}  rel {r:.3e}{flag}")
        if r >= TOL: failed = True

print(f"\n{n} entries compared; worst relative deviation: {worst:.3e}")
sys.exit(1 if failed else 0)
