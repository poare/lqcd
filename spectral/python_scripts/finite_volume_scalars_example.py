"""
Minimal working example for constructing smeared spectral functions in the
toy model of interacting scalars used in:

Hansen, Meyer, and Robaina
Phys.Rev.D 96 (2017) 9, 094513
1704.08993 [hep-lat]

Hansen, Lupo, and Tantalo
Phys.Rev.D 99 (2019) 9, 094508
1903.06476 [hep-lat]

To avoid computing slow sums, the script reads in pre-tabulated values for the
locations of finite-volume poles and their associated spectral weights.
The folder containing these "weights" is specified at run time.
"""
import sys
import os
import numpy as _np
import seaborn as sns
import pylab as plt
import pandas as pd
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np


def main():

    if len(sys.argv) != 2:
        print(f"Usage: $ python {sys.argv[0]} <path/to/weights>")
        exit(1)
    path = sys.argv[1]

    poisson_kernel_fast = jax.vmap(poisson_kernel, in_axes=(0, None, None), out_axes=0)

    @jax.jit
    def build_fv_density_fast(L, mpi, mk, mphi, E, epsilon, n2, nu):
        """
        Computes the smeared finite-volume spectral density.
        """
        p2 = (2*np.pi/L)**2 * n2
        Ek = 2 * np.sqrt(mk**2 + p2)
        rho = np.zeros(len(E))

        weight = (2*np.pi / 2) * (mphi**2 * mpi) / (mpi*L)**3
        weight = weight * nu / Ek**2

        pos = poisson_kernel_fast(E, +Ek, epsilon) * weight
        # And include the poles on the negative axis
        neg = poisson_kernel_fast(E, -Ek, epsilon) * weight
        rho = np.sum(pos, axis=1) + np.sum(neg, axis=1)
        return rho

    # Define particle masses - all masses in lattice units
    mpi = 0.066
    mk = 3.55*mpi
    mphi = 7.30*mpi
    sigma = 0.25

    # Compute infinite-volume spectral density
    Emin = 0
    Emax = 3.0
    num = 1000
    E = np.linspace(Emin, Emax, num=num)
    rho = np.zeros(len(E))
    threshold = np.heaviside(E - 2*mk, 0.5)
    rho = rho + threshold * np.sqrt(1 - (2*mk/E)**2) / (16*np.pi**2) * (mphi**2/mpi**2)
    rho = rho.at[E < (2*mk)].set(0)
    rho_infty = rho

    # Compute smeared FV spectral densities
    sigmas = np.arange(6)* 0.1 + 0.05
    rho_fv = []
    for L in [48, 96, 128]:#, 256, 512]:
        print("L=",L)
        for sigma in sigmas:
            print("-->", sigma)
            fname = os.path.join(path, f"weights_{L}.txt")
            # f"/Users/willjay/GitHub/inverse_problems/inverse_problems/weights_{L}.txt"
            n2, nu = read_weights(fname)
            tmp = build_fv_density_fast(L, mpi, mk, mphi, E, epsilon=sigma, n2=n2, nu=nu)
            rho_fv.append({
                'L': L,
                'sigma': sigma,
                'rho': tmp,
            })
    rho_fv = pd.DataFrame(rho_fv)

    fig, axarr = plt.subplots(ncols=3, nrows=2, sharey=True, sharex=True)

    # Plot finite-volume smeared spectral functions
    colors = {L: color for L, color in zip(rho_fv['L'].unique(), sns.color_palette())}
    for L, subdf in rho_fv.groupby("L"):
        print(L)
        for (_, row), ax in zip(subdf.iterrows(), _np.ravel(axarr)):
            x = np.array(E)
            y = np.array(row['rho'])
            print(len(x), len(y))
            ax.errorbar(x, y, color=colors[L])
            ax.set_title(r"$\sigma=$"+f"{row['sigma']:.2f}")
    ax.set_ylim(top=0.3)

    # Plot infinite-volume spectral function
    for ax in _np.ravel(axarr):
        x = np.array(E)
        y = np.array(rho_infty) / np.sqrt(2*np.pi)  # Kludge for normalization?
        ax.errorbar(x, y, color='k')

    plt.show()


def poisson_kernel(x, x0, eps):
    """
    The Poisson kernel, i.e.,
    1/pi * Im[1/(x0 - z)], where z = x + i*eta
    Args:
        x: independent variable, the location on the real line
        x0: pole location
        eta: regulator/smearing width (location above the real axis)
    """
    return eps/(eps**2 + x**2 - 2*x0*x + x0**2)/np.pi


def read_weights(fname):
    """
    Reads the weights from file.
    """
    n2, nu = [], []
    with open(fname) as ifile:
        for line in ifile:
            tmp1, tmp2 = line.split(", ")
            n2.append(tmp1)
            nu.append(tmp2)
    return np.array(n2, dtype=int), np.array(nu, dtype=int)


# def build_fv_density(L, mpi, mk, mphi, E, epsilon):
#     """
#     Computes the smeared finite-volume spectral density.
#     """
#     base = "/Users/willjay/GitHub/inverse_problems/inverse_problems"
#     n2, nu = read_weights(f"{base}/weights_{L}.txt")
#     p2 = (2*np.pi/L)**2 * n2
#     Ek = 2 * np.sqrt(mk**2 + p2)
#     rho = np.zeros(len(E))
#     for Ei, nu_i in zip(Ek, nu):
#         weight = (2*np.pi / 2)
#         weight *= mphi**2 * mpi
#         weight /= (mpi*L)**3
#         weight *= nu_i/Ei**2
#         rho = rho + weight*poisson_kernel(E, Ei, epsilon)
#         # And include the poles on the negative axis
#         rho = rho + weight*poisson_kernel(E, -Ei, epsilon)
#     return rho


if __name__ == '__main__':
    main()