
################################################################################
# TODO description
################################################################################

from utils import *

# TODO make a function which runs given a set of parameters, and do a parameter scan to see if
# anything comes close to the original
def main():
    Ntau = 8
    Nomega = 100
    G = simulate_G(Ntau)
    lam0, mu0, mup0 = 1., 1., 1.
    max_iters, eps = 10000, 1e-8
    x0 = np.zeros((Nomega), dtype = np.float64)
    zp0, up0 = np.zeros((Nomega), dtype = np.float64), np.zeros((Nomega), dtype = np.float64)
    z0, u0 = np.zeros((Nomega), dtype = np.float64), np.zeros((Nomega), dtype = np.float64)
    params = ADMMParams(lam0, mu0, mup0, max_iters, eps, x0, zp0, up0, z0, u0, (Ntau, Nomega))
    admm()

if __name__ == '__main__':
    main()
