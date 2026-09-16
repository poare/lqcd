# Free-field check of the full measurement.
#
# On a unit gauge field the clover term vanishes and the action is free Wilson,
# so the amputated vertex Gamma_mu = S^-1 G_mu S^-1 has the analytic tree form.
# The check is structural and normalisation-independent, which is the honest
# scope here: it verifies that Gamma_mu, colour-traced, is PROPORTIONAL to
# gamma_mu, with a coefficient whose ratio to the analytic momentum factor is
# one constant common to all mu and all k. That catches any residual error in
# gamma placement, shift direction, phase or twist.
#
# In the event the constant is not merely common but exactly -2i, so this
# checks the normalisation too: Gamma_mu = -2i sin(p_mu) gamma_mu exactly.
# Note sin(p_mu), NOT the Wilson p~ = 2 sin(p_mu/2) -- a symmetric difference
# gives the full-angle sine, and the two differ by cos(p_mu/2), which is not
# constant across k. Both ratios are printed so the distinction stays visible.
#
# The test runs with PERIODIC time and b = 0, deliberately. With antiperiodic
# time the propagator is antiperiodic while seqSource's shift is periodic, so
# the tree-level reference fails at the boundary slices and mu=3 misses by a
# few percent. That boundary treatment is correct -- it is what QLUA did, and
# it is where any O_44 discrepancy is predicted to come from -- but it makes
# the free-field formula inapplicable, so it is tested against real reference
# data in Task 6/7 rather than against an analytic form that does not hold.
#
#   python3 check_freefield.py freefield.txt
import sys, numpy as np

L = [4,4,4,8]; b = [0,0,0,0]

g = np.zeros((4,4,4), dtype=complex)
g[0] = [[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]]
g[1] = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
g[2] = [[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]]
g[3] = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]

def read(path):
    d = {}
    for line in open(path):
        p = line.split()
        if len(p) != 11: continue
        tag = p[0]; k = tuple(int(x) for x in p[1:5])
        s0,s1,c0,c1 = (int(x) for x in p[5:9])
        d.setdefault((tag,k), np.zeros((4,4,3,3), dtype=complex))[s0,s1,c0,c1] = float(p[9])+1j*float(p[10])
    return d

d = read(sys.argv[1])
ks = sorted({k for (t,k) in d if t == "prop"})
assert ks, "no prop entries in output"

# Two candidate momentum factors. The symmetric difference in seqSource gives a
# vertex going as sin(p_mu); the Wilson p~ = 2 sin(p_mu/2) is the other natural
# guess. Report both and assert on whichever the structure actually follows.
rows = []
for k in ks:
    S = d[("prop",k)].transpose(2,0,3,1).reshape(12,12)
    Sinv = np.linalg.inv(S)
    p    = [2*np.pi*(k[m]+b[m])/L[m] for m in range(4)]
    sinp = [np.sin(pm)     for pm in p]           # symmetric difference
    ptil = [2*np.sin(pm/2) for pm in p]           # Wilson p-tilde
    for mu in range(4):
        G = d[(f"O{mu+1}{mu+1}",k)].transpose(2,0,3,1).reshape(12,12)
        Gam = (Sinv @ G @ Sinv).reshape(3,4,3,4)
        Gam = np.einsum('aiaj->ij', Gam)/3.0          # colour trace
        scale = np.abs(Gam).max()
        if scale < 1e-10:
            print(f"k={k} mu={mu}: vertex vanishes (|Gam| {scale:.2e})")
            continue
        # projection onto gamma_mu, and the residual orthogonal to it
        c = np.vdot(g[mu], Gam)/np.vdot(g[mu], g[mu])
        resid = np.abs(Gam - c*g[mu]).max()/scale
        assert resid < 1e-6, \
            f"k={k} mu={mu}: vertex NOT proportional to gamma_mu (resid {resid:.2e})"
        rows.append((k, mu, c, sinp[mu], ptil[mu]))

print(f"\nvertex is proportional to gamma_mu for all {len(rows)} (k,mu) pairs\n")
print(f"{'k':14s} {'mu':>3s} {'c':>26s} {'c/sin(p)':>26s} {'c/ptilde':>26s}")
for (k, mu, c, s, t) in rows:
    rs = c/s if abs(s) > 1e-12 else float('nan')
    rt = c/t if abs(t) > 1e-12 else float('nan')
    print(f"{str(k):14s} {mu:3d} {c:26.6f} {rs:26.6f} {rt:26.6f}")

def spread(vals):
    a = np.array(vals); return np.abs(a - a.mean()).max(), a.mean()

sp_s, mean_s = spread([c/s for (_,_,c,s,_) in rows if abs(s) > 1e-12])
sp_t, mean_t = spread([c/t for (_,_,c,_,t) in rows if abs(t) > 1e-12])
print(f"\nc/sin(p) : mean {mean_s:.6f}  spread {sp_s:.3e}")
print(f"c/ptilde : mean {mean_t:.6f}  spread {sp_t:.3e}")

tol_s = 1e-6*max(abs(mean_s), 1.0)
assert sp_s < tol_s, (
    "c/sin(p) is not common to all mu and k -- a convention differs per "
    f"direction (spread {sp_s:.3e} > {tol_s:.3e})")

want = -2j
assert abs(mean_s - want) < 1e-9, (
    f"constant is {mean_s:.9f}, expected exactly {want} "
    "-- the vertex normalisation is off")
print("\nPASS: free-field vertex is exactly -2i sin(p_mu) gamma_mu")
