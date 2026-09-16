# Exact orthogonality test for projectMomentum.
#
# The port feeds F(x) = exp(-i q.(x-y)) * 1 with q the twisted momentum for
# k0 = (1,1,1,2), then projects at every k in a small box. The projector's own
# phase is e^{+i(k+b).(x-y)}, QLUA's sign (emt_npr.qlua:303-307), so field and
# projector cancel at k = k0 giving exactly V, and are orthogonal elsewhere
# giving exactly 0. A sign error in the projector matches at -k0, outside the
# scanned box, so the diagonal vanishes; a dropped twist does the same; a
# dropped y offset keeps the magnitude but rotates the phase.
#
#   ./npr_momfrac -i test_project.ini.xml -o project.out.xml | python3 check_project.py
import sys

V = 4*4*4*8

vals = {}
for line in sys.stdin:
    if not line.startswith("PROJ "):
        continue
    _, k, re, im = line.split()
    vals[k] = float(re) + 1j*float(im)

assert "1112" in vals, f"missing the diagonal entry; got {sorted(vals)}"
diag = vals.pop("1112")
print(f"diagonal  k=(1,1,1,2): {diag:.6e}   (expect {V}+0j)")
assert abs(diag - V) < 1e-8 * V, "wrong normalisation or wrong phase sign"

worst_k, worst = max(vals.items(), key=lambda kv: abs(kv[1]), default=("-", 0.0))
print(f"worst off-diagonal k={worst_k}: {abs(worst):.3e}   (expect ~0, over {len(vals)} momenta)")
assert abs(worst) < 1e-8 * V, "off-diagonal leakage - twist, sign or y-offset is wrong"
print("PASS: momentum projection")
