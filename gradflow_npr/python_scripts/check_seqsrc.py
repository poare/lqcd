# Free-field test of seqSource.
#
# On a unit gauge field b_mu(x) = gamma_mu [S(x+mu) - S(x-mu)].  Feeding the
# conjugate plane wave S(x) = exp(-i q.x) * 1 gives
#   b_mu(x) = -2i sin(q_mu) gamma_mu S(x),
# so the projection at q -- whose own phase is e^{+i q.x} -- must be exactly
# -2i sin(q_mu) * V * gamma_mu.
#
# This tests the shift DIRECTIONS, the gamma PLACEMENT (left-multiplication)
# and the relative sign all at once. A swapped FORWARD/BACKWARD negates the
# answer; a missing gamma_mu changes the spin structure entirely.
#
# b = 0 here, deliberately. A twisted plane wave is antiperiodic in time while
# QDP++'s shift is periodic, so with b_3 = 1/2 the reference formula is wrong
# on the boundary time slices and mu=3 misses by exactly 1/4 (mu=0,1,2 still
# agree to 1e-16). That is the reference being inapplicable, not seqSource
# being wrong -- and it confirms the design's claim that the derivative uses
# raw links and a periodic shift. The twist is tested in check_project.py.
#
#   ./npr_momfrac -i test_seqsrc.ini.xml -o seqsrc.out.xml | python3 check_seqsrc.py
import sys, numpy as np

L = [4,4,4,8]; V = 4*4*4*8
k = [1,1,1,2]; b = [0,0,0,0]
q = [2*np.pi*(k[m]+b[m])/L[m] for m in range(4)]

g = np.zeros((4,4,4), dtype=complex)
g[0] = [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]
g[1] = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
g[2] = [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]
g[3] = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]

got = np.zeros((4,4,4), dtype=complex)
seen = 0
for line in sys.stdin:
    if not line.startswith("SEQ "):
        continue
    _, mu, r, c, re, im = line.split()
    got[int(mu), int(r), int(c)] = float(re) + 1j*float(im)
    seen += 1
assert seen == 4*4*4, f"expected 256 SEQ lines, got {seen}"

ok = True
for mu in range(4):
    want = -2j*np.sin(q[mu]) * V * g[mu]
    d = np.abs(got[mu] - want).max()
    scale = np.abs(want).max()
    print(f"mu={mu}: max dev {d:.3e}  (scale {scale:.3e})")
    if d > 1e-8*max(scale, 1.0):
        ok = False
        # a pure sign error is the most likely failure - name it explicitly
        if np.abs(got[mu] + want).max() < 1e-8*max(scale, 1.0):
            print(f"  -> mu={mu} is exactly NEGATED: FORWARD/BACKWARD are swapped")
assert ok, "sequential source mismatch"
print("PASS: sequential source")
