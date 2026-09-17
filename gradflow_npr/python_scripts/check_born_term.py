# Free-field Born-term cross-check.
#
#   python3 check_born_term.py <chroma_txt> [born_term.h5]
#
# Two independent comparisons of the same Chroma output:
#
#   (1) RAW, against the 2020 reference born_term.h5. That file was written by
#       testing/free_field/zero_field_npr.qlua's compute_npr, which uses a
#       MOMENTUM WALL source and projects through the sink. This port uses a
#       POINT source. In the free field S(x,y) depends only on x - y, so the
#       two differ by exactly V in every entry and for any source point:
#
#           born_term.h5  ==  V * chroma,     V = prod(L_mu)
#
#       No fit, no tolerance on the factor -- V is put in by hand and the
#       residual has to vanish.
#
#   (2) AMPUTATED, against the analytic tree-level vertex. The operator
#       insertion is a pure point-split gamma_mu term with no Wilson piece, so
#       S~(p) cancels and the vertex does not depend on kappa, csw or the
#       Wilson term:
#
#           Gamma_mu(p) = -2i sin(p_mu) gamma_mu,  p_mu = 2 pi (k_mu + b_mu)/L_mu
#
#       FULL angle sin(p_mu), not the half-angle 2 sin(p_mu / 2) that
#       npr_momfrac/python_scripts/analysis.py:244 uses in its analytic
#       born_term (left commented out there in favour of born_term_numerical).
#
#       The time direction carries a boundary correction. Antiperiodicity is
#       implemented by flipping U_3 on the last time slice inside the Dirac
#       operator while seqSource uses the RAW link, so one slice per term
#       enters with the wrong sign. Each slice carries weight 1 / L_t:
#
#           Gamma_3(p) = -2i sin(p_3) (1 - 2 / L_t) gamma_3
#
#       This is a convention, not a bug -- it cancels in Z as long as the Born
#       term is generated at the SAME L_t as the data. It does NOT cancel if a
#       Born term from one geometry is reused on another.
#
# h5py note: QLUA writes a compound type, (4,4) of (3,3) complex, that this
# h5py build has no conversion path for -- d[()] and read_direct both raise
# "no appropriate function for conversion path". The datasets are contiguous,
# so they are read at their file offsets instead and byte-swapped by numpy.
import sys, numpy as np, h5py

chroma_txt = sys.argv[1]
ref_h5 = (sys.argv[2] if len(sys.argv) > 2
          else "/home/poare/lqcd/npr_momfrac/testing/free_field/born_term.h5")

LL   = [16, 16, 16, 48]
BVEC = [0.0, 0.0, 0.0, 0.5]
V    = LL[0] * LL[1] * LL[2] * LL[3]
TAGS = ["prop", "O11", "O22", "O33", "O44"]
TOL  = 1e-10

# DeGrand-Rossi, identical to utilities/pytools.py:47-50.
gamma = np.zeros((4, 4, 4), dtype=complex)
gamma[0] = [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]
gamma[1] = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
gamma[2] = [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]
gamma[3] = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]

# ---------------------------------------------------------------- chroma text
got = {}
for line in open(chroma_txt):
    f = line.split()
    if len(f) != 11:
        continue
    tag = f[0]
    k = tuple(int(x) for x in f[1:5])
    s0, s1, c0, c1 = (int(x) for x in f[5:9])
    got.setdefault((tag, k), np.zeros((4,4,3,3), dtype=complex))[s0,s1,c0,c1] \
        = float(f[9]) + 1j*float(f[10])

moms = sorted({k for (_, k) in got})
print(f"read {len(got)} objects at {len(moms)} momenta from {chroma_txt}")

# ------------------------------------------------------------- reference h5
# born_term.h5 layout is <tag>/p<k0><k1><k2><k3>/cfg<n>: one level less than
# cfg1600.h5, which interposes the source-point group.
def kstr(k):
    return "p" + "".join(str(x) for x in k)

h = h5py.File(ref_h5, "r")
loc = {}
missing = []
for tag in TAGS:
    for k in moms:
        path = f"{tag}/{kstr(k)}"
        if path not in h:
            missing.append(path); continue
        d = h[path][list(h[path].keys())[0]]
        loc[(tag, k)] = (d.id.get_offset(), d.id.get_storage_size())
h.close()
if missing:
    print(f"  !! {len(missing)} paths absent from reference, e.g. {missing[:3]}")

fh = open(ref_h5, "rb")
def ref(tag, k):
    off, sz = loc[(tag, k)]
    fh.seek(off)
    return np.frombuffer(fh.read(sz), dtype=">c16").reshape(4,4,3,3).astype(complex)

# -------------------------------------------------- (1) raw, against 2020
print(f"\n(1) RAW  vs born_term.h5, with the exact factor V = {V}")
worst_raw, n_raw, failed = 0.0, 0, False
# Normalize per tag, by that tag's largest reference entry over all momenta,
# rather than per momentum. The vertex vanishes wherever sin(p_mu) = 0 -- O11
# at every k_0 = 0, and so on -- so a per-momentum denominator divides by an
# exact zero and turns 1e-15 of roundoff into a 1e290 "mismatch".
for tag in TAGS:
    kk = [k for k in moms if (tag, k) in loc]
    if not kk:
        continue
    refs = {k: ref(tag, k) for k in kk}
    scale = max(np.abs(refs[k]).max() for k in kk)
    w, w_at = 0.0, None
    nz = 0
    for k in kk:
        a = got[(tag, k)] * V
        r = np.abs(a - refs[k]).max() / scale
        if np.abs(refs[k]).max() > 1e-12 * scale:
            nz += 1
        if r > w:
            w, w_at = r, k
        worst_raw = max(worst_raw, r); n_raw += 1
        if r >= TOL:
            failed = True
    verdict = "OK" if w < TOL else "FAIL"
    print(f"    {tag:5s} worst {w:.3e} at {kstr(w_at):12s} "
          f"(scale {scale:.4e}, {nz}/{len(kk)} momenta with a nonvanishing "
          f"vertex)  {verdict}")
print(f"    {n_raw} objects compared; worst {worst_raw:.3e}"
      f"   {'OK' if worst_raw < TOL else 'FAIL'}")

# ------------------------------------ (2) amputated, against analytic vertex
def mat(v):                      # [i,j,a,b] -> 12x12 ordered (a,i),(b,j)
    return np.einsum("ijab->aibj", v).reshape(12, 12)

# No correction is imposed here. The ratio to the clean -2i sin(p_mu) is
# measured and then confronted with the two hypotheses, because which one
# holds depends on the SOURCE TYPE, not on the port:
#
#   1              a point source deep in the time bulk: the raw-link boundary
#                  defect is exponentially suppressed in the distance from the
#                  source to t = L_t - 1
#   1 - 2 / L_t    a momentum wall (what born_term.h5 used): the y-average
#                  turns the defect into a uniform 1 / L_t weight per slice
#
# Anything else means the source sits close enough to the time boundary for
# the defect to survive, and the vertex is then not proportional to gamma_mu
# at all -- the |imag| and spread columns are what reveal that.
WALL = 1 - 2/LL[3]
print("\n(2) AMPUTATED  vs -2i sin(p_mu) gamma_mu")
print(f"    hypotheses for the ratio: point source in bulk 1.0, "
      f"momentum wall 1 - 2/L_t = {WALL:.12f}")
ratios = {mu: [] for mu in range(4)}
resid  = {mu: 0.0 for mu in range(4)}
n_amp = 0
for k in moms:
    if ("prop", k) not in got:
        continue
    Sinv = np.linalg.inv(mat(got[("prop", k)]))
    p = [2*np.pi*(k[mu] + BVEC[mu])/LL[mu] for mu in range(4)]
    for mu in range(4):
        tag = f"O{mu+1}{mu+1}"
        if (tag, k) not in got:
            continue
        G = mat(got[(tag, k)])
        Gam = np.einsum("aiaj->ij", (Sinv @ G @ Sinv).reshape(3,4,3,4)) / 3.0
        ref_mag = 2*abs(np.sin(p[mu]))
        if ref_mag < 1e-14:                  # sin(p_mu) = 0: vertex vanishes
            continue
        c = np.trace(gamma[mu].conj().T @ Gam) / 4
        r = c / ((-2j) * np.sin(p[mu]))
        ratios[mu].append(r)
        # how much of the vertex is NOT along gamma_mu
        resid[mu] = max(resid[mu],
                        np.abs(Gam - c*gamma[mu]).max() / ref_mag)
        n_amp += 1

worst_amp = 0.0
for mu in range(4):
    a = np.array(ratios[mu])
    if len(a) == 0:
        print(f"    mu={mu}  no momenta with sin(p_mu) != 0")
        continue
    m = a.real.mean()
    d1, dw = abs(m - 1.0), abs(m - WALL)
    which = ("point-source/clean" if d1 < dw else "momentum-wall")
    dev = min(d1, dw)
    spread = a.real.max() - a.real.min()
    offg = resid[mu]
    # A direction only counts as clean if the ratio sits on one hypothesis AND
    # the vertex really is proportional to gamma_mu across all momenta.
    bad = max(dev, spread, offg, np.abs(a.imag).max())
    worst_amp = max(worst_amp, bad)
    print(f"    mu={mu}  n={len(a):4d}  ratio {m:.15f}  "
          f"|imag| {np.abs(a.imag).max():.2e}  spread {spread:.3e}  "
          f"off-gamma {offg:.3e}")
    print(f"           -> {which}: |ratio - {1.0 if d1 < dw else WALL:.12f}| "
          f"= {dev:.3e}   {'OK' if bad < TOL else 'DEVIATES'}")
if worst_amp >= TOL:
    failed = True
print(f"    {n_amp} vertices compared; worst {worst_amp:.3e}"
      f"   {'OK' if worst_amp < TOL else 'FAIL'}")
fh.close()

print(f"\n{'PASS' if not failed else 'FAIL'}"
      f"  (raw {worst_raw:.3e}, amputated {worst_amp:.3e}, tol {TOL:.0e})")
sys.exit(1 if failed else 0)
