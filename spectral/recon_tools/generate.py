"""
Generates a spectral function to simulate and saves it to an hdf5 file.

Arguments
---------
string : path to save the spectral function at.
"""

import sys
import numpy as np
import gmpy2 as gmp

sys.path.append('/Users/theoares/inverse_problems/inverse_problems')
import fileio
import scipy.interpolate as interpolate
import pylab as plt
# from nevanlinna import *

# sys.path.append('/Users/theoares/lqcd/spectral/python_scripts')
# import poare_utils

# Set precision for gmpy2 and initialize complex numbers
prec = 128
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
I = gmp.mpc(0, 1)

def main():

    fname = str(sys.argv[1])
    print('Writing data to: ' + fname)

    # Construct desired spectral function
    beta = 100
    freqs = matsubara(beta, boson=True)

    # ngs = np.zeros(len(freqs), dtype=object)
    # for mm in np.linspace(0.2, 0.8, num=4):
    # for mm in [gmp.mpfr("0.05"), gmp.mpfr("0.1")]:
    #     ngs = ngs + analytic_ft(freqs, mm, beta)
    # ngs = ngs + kinematic_feature(freqs)
    # ngs = ngs + gaussian(freqs)
    # ngs = ngs / np.absolute(np.sum(ngs))

    ngs = gaussian_density(beta, sigma=0.1, factor=10)
    freqs, ngs = freqs[2:], ngs[2:]

    # print('Freqs: ')
    # print(freqs)

    # print('Greens function data: ')
    # print(ngs)

    # phis = poare_utils.construct_phis(freqs, poare_utils.h(ngs))

    # print('Phis: ')
    # print(phis)

    # Save spectral function
    fileio.write_gmp_input_h5(fname, beta, freqs[:20], ngs[:20])
    print('Green\'s function data written to: ' + fname)

def build_bspline(x, y):
    t, c, k = interpolate.splrep(x, y, s=0, k=4)
    bspline = interpolate.BSpline(t, c, k, extrapolate=True)
    return bspline

def matsubara(beta, boson = False):
    rng = np.arange(beta)
    if boson:
        return np.array([2*gmp.const_pi()*I*n/beta for n in rng])
    return np.array([(2*n + 1)*gmp.const_pi()*I/beta for n in rng])

def pole(m, z):
    return 1 / (m - z)

def analytic_dft(m, z, beta):
    prefactor = -1*(1 + np.exp(-m*beta))
    pole1 = 1/(np.exp(z-m) - 1)
    pole2 = 1/(np.exp(z+m) - 1)
    return prefactor * (pole1 + pole2)

def analytic_ft(z, m, beta):
    prefactor = -1*(1 + gmp.exp(-m*beta))
    pole1 = 1/(z - m)
    pole2 = 1/(z + m)
    return prefactor * (pole1 + pole2)

def unstable_pole(z, m, gamma):
    mpi = 0.140
    return 0.25*(1 - 4*mpi**2/z**2)**0.5/(z - (m - 1j*gamma))

def kinematic_feature(z):
    z = z + 0*1j
    return -1*(gmp.mpfr("0.1") - z)**0.5

def gaussian(z, sigma=None):
    if sigma is None:
        sigma = gmp.mpfr("0.5")
    pi = gmp.acos(gmp.mpfr("-1.0"))
    arg = -z**2/sigma**2
    tmp = []
    for arg_i in arg:
        tmp.append(
            gmp.exp(arg_i)/ sigma / gmp.sqrt(pi*gmp.mpfr("2.0"))
        )
    tmp = np.array(tmp) / np.sum(tmp)
    print("Norm is", np.sum(tmp))
    return tmp

def kernel(omega, beta, tau):
    return (np.exp(-omega*tau) + np.exp(-omega*(beta-tau)))/(1 + np.exp(-beta*tau))

def gaussian_density(beta, sigma, factor):

    # Input spectral density is a Gaussian
    x = np.linspace(0, 10*sigma, num=10000)
    rho = np.exp(-0.5*x**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)
    rho = 2*rho  # Unit normalized on [0, infinity)

    print("Normalization of spectral density", np.trapz(x=x, y=rho))

    # Euclidean-time correlator = Laplace transform of the spectral density
    tau = np.arange(beta)  # Evaluate on the lattice grid [0, ..., beta-1]
    corr = np.array([np.trapz(x=x, y=rho*kernel(x, beta, tau_i)) for tau_i in tau])

    # corr = [np.trapz(x=x, y=rho*np.exp(-x*tau_i)/(2.*np.pi)) for tau_i in tau]

    # Refine the lattice correlator on a finer grid
    corr_spline = build_bspline(np.arange(1, beta), corr[1:])
    tau_fine = np.linspace(0, beta, num=factor*beta)
    corr_fine = corr_spline(tau_fine) / corr_spline(0)
    print("C(t=0)", corr[0], "(direct)")
    print("C(t=0)", corr_fine[0], "(spline)")

    # fig, ax = plt.subplots(1)
    # ax.errorbar(x=tau, y=corr, label='direct')
    # ax.errorbar(x=tau_fine, y=corr_fine, label='spline')
    # ax.legend()
    # fig.savefig("tmp.png")

    # Compute the Fourier coefficients
    corr_ft = np.zeros(beta, dtype=complex)
    omega = (2*np.arange(beta) + 1) * np.pi/beta
    for l, omega_l in enumerate(omega):
        phase = np.exp(1j*omega_l*tau_fine)
        re = np.trapz(x=tau_fine, y=(corr_fine*phase).real)
        im = np.trapz(x=tau_fine, y=(corr_fine*phase).imag)
        corr_ft[l] = re + 1j*im
    corr_ft = corr_ft / (2*np.pi)
    return corr_ft

if __name__ == '__main__':
    main()