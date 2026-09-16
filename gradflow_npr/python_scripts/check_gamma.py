# Compare Chroma's Gamma(1<<mu) against the gamma basis the 2020 analysis
# assumes (analysis.py:19-22).  Reads the port's stdout on stdin.
#
#   ./npr_momfrac -i test_gamma.ini.xml -o gamma.out.xml | python3 check_gamma.py
import sys, numpy as np

# analysis.py's basis, the one QLUA wrote and the 2020 analysis assumes
g = np.zeros((4,4,4), dtype=complex)
g[0] = [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]
g[1] = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
g[2] = [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]
g[3] = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]

got = np.zeros((4,4,4), dtype=complex)
seen = 0
for line in sys.stdin:
    if not line.startswith("GAMMA "):
        continue
    _, mu, r, c, re, im = line.split()
    got[int(mu), int(r), int(c)] = float(re) + 1j*float(im)
    seen += 1

assert seen == 4*4*4, f"expected 256 GAMMA lines, got {seen}"
d = np.abs(got - g).max()
print(f"max |Chroma Gamma(1<<mu) - analysis.py gamma[mu]| = {d:.3e}")
if d >= 1e-12:
    for mu in range(4):
        dm = np.abs(got[mu] - g[mu]).max()
        print(f"  mu={mu}: max diff {dm:.3e}")
        if dm >= 1e-12:
            print("   Chroma:\n", got[mu])
            print("   analysis.py:\n", g[mu])
    raise AssertionError("GAMMA BASIS MISMATCH - a rotation is needed; stop and redesign")
print("PASS: gamma bases agree")
