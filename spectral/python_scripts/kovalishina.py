################################################################################
# Implementation of the solution to the Hamburger moment problem from          #
# Kovalishina's paper.
################################################################################

# from utils import *
import utils
import numpy as np
import mpmath as mp

import sys
sys.path.append('./will_scripts')
import hamburger_mp as ham              # Will code base

def gaussian(mu, sigma):
    # def f(x):
    #     return 1/(mp.sqrt(2*mp.pi)*sigma) * mp.exp(-(x - mu)**2 / (mp.mpf("2")*(sigma**2)))
    # return f
    return lambda x : 1/(mp.sqrt(2*mp.pi)*sigma) * mp.exp(-(x - mu)**2 / (mp.mpf("2")*(sigma**2)))

def test_basics():
    """Tests a basic 2 model with Gaussians"""
    k = 2
    # rho = lambda x : mp.matrix([
    rho = np.array([
        [gaussian(0, 1), gaussian(1, 1)],
        [gaussian(1, 1), gaussian(0, 2)],
    ])
    N = 4           # compute moments 0, 1, ..., 2N = 0, 1, ..., 8
    print(rho[0, 0](0))
    moments = []
    for n in range(0, 2*N+1):           # moment to compute
        kmoment = []
        for i in range(k):
            row = []
            for j in range(k):
                moment_functional = lambda x : (x**n) * rho[i, j](x)
                row.append(
                    mp.quad(moment_functional, [-mp.inf, mp.inf])
                )
            kmoment.append(row)
        # moments.append(
        #     mp.quad(moment_functional, [-mp.inf, mp.inf])
        # )
        moments.append(
            mp.matrix(kmoment)
        )
    print(moments)
    print(len(moments))
    print((len(moments) - 1) // 2)
    # VERIFY: pth moment of N(0, sigma) is \sigma^p (p-1)!!



    return

def main():

    # TODO method stub
    # test_basics()
    corr = ham.ToyModel(nt=30, nstates=32).correlator
    print(corr)
    print(corr.shape)
    
    
    return

if __name__ == '__main__':
    main()
