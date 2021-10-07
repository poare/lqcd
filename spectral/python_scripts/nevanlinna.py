
################################################################################
# This script reads in Npts values of the Matsubara Green's function at        #
# frequencies i\omega and performs an analytic continuation of these points to #
# the upper half plane by exploiting properties of Nevanlinna / contractive    #
# functions. This is an implementation following the paper:                    #
#       Fei, J., Yeh, C.N. & Gull, E. Nevanlinna Analytical Continuation.      #
#                       Phys Rev Lett 126, 056402 (2021).                      #
# A note about the implementation: the notation used in this script is the     #
# same as that used in the paper, for clarity. Namely, the Matsubara freqs are #
# stored in a numpy array Y, and the Nevanlinna function values NG are stored  #
# in a numpy array C.
################################################################################

from utils import *

def main():
    # Choose interpolant function theta_{M + 1}, and the grid to evaluate on. These parameters can all be varied.
    Nreal = 6000
    omega_bounds = [-10, 10]
    eta = 1e-3
    theta_mp1 = lambda z : 0            # change this based on priors

    # Data input with frequencies and values of G
    data_path = '/Users/theoares/lqcd/spectral/hardy_optim_clean_submission/test/freqs_1.txt'
    Y, C, lam, Npts = read_txt_input(data_path)
    print('Read in ' + str(Npts) + ' Matsubara modes at frequencies ~ ' + str(["{0:1.8f}".format(x) for x in Y]) + '.')
    for z in C:
        assert z.imag > 0.0, 'Negative of input function is not Nevanlinna.'
    Pick = construct_Pick(Y, lam)
    print('Pick matrix: ')
    print(Pick)
    # TODO implement check for positive semidefinite-ness

    phi = construct_phis(Y, lam)
    print('Phi[k] is: ' + str(phi))
    zmesh = np.linspace(omega_bounds[0], omega_bounds[1], num = Nreal)
    zspace = np.array([gmp.mpc(z, eta) for z in zmesh])
    NGreal = analytic_continuation(Y, phi, zspace, theta_mp1)

    # confirm this is a valid analytic continuation-- it extends the input points
    test_cont = analytic_continuation(Y, phi, Y, theta_mp1)
    for k in range(Npts):
        assert abs(test_cont[k] - C[k]) < gmp.mpfr(1e-15)

    out_path = '/Users/theoares/Dropbox (MIT)/research/spectral/testing/output_1.txt'
    write_txt_output(out_path, NGreal, Nreal, omega_bounds, eta)

if __name__ == '__main__':
    main()
