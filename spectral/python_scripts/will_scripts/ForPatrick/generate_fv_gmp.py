import numpy as np
import scipy.interpolate as interpolate
import gmpy2 as gmp
import fileio
import pylab as plt

# Set precision for gmpy2 and initialize complex numbers
prec = 256
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
I = gmp.mpc(0, 1)
PI = gmp.const_pi()

def main():

    # Define particle masses -- all in lattice units
    mpi = gmp.mpfr("0.066")
    mk = gmp.mpfr("3.55")*mpi
    mphi = gmp.mpfr("7.30")*mpi

    #for L in np.array([24, 32, 40, 48, 96, 128, 256, 512], dtype=gmp.mpfr):
    for L in np.array([48, 96, 128, 256], dtype=gmp.mpfr):
        print(f"Running L={L}")
        beta = 2*L
        freqs, ngs, norm = compute_ng(L, mpi, mk, mphi, beta)
        freqs, ngs = freqs[0:min(96, len(freqs))], ngs[0:min(96, len(freqs))]
        # idxs = get_input_idxs(beta, target=min(100, beta-1))
        # print(f"Used a total of {len(idxs)} values")
        # idxs = generate_uniform_idxs(beta, 90)
        # idxs = range(len(ngs)-1)
        fname = f"data_fv_{L}.h5"
        fileio.write_gmp_input_h5(fname, beta, freqs, ngs, start=0, stop=1, num=1000, normalization=norm)
        # fileio.write_gmp_input_h5(fname, beta, freqs[idxs], ngs[idxs], norm)

# ----- end main ----- #

def generate_uniform_idxs(beta, num):
    idxs = np.arange(0, beta, beta//num, dtype=int)
    return idxs[:num]

def generate_log_idxs(beta, num):
    x = np.linspace(0.01, 1, num=num)
    x = np.log(x)
    x = -x/(max(x) - min(x))
    x = x[::-1]
    x = np.array(x * (beta-1), dtype=int)
    x = np.unique(x)
    return x


def get_input_idxs(beta, target):
    for num in range(target, beta):
        idxs = generate_log_idxs(beta, num)
        if len(idxs) >= target:
            return idxs
    return idxs


def matsubara(beta, boson = False):
    n = np.arange(beta, dtype=gmp.mpfr)
    if boson:
        return 2*n*PI*I/beta
    return (2*n+1)*PI*I/beta


def read(fname):
    """
    Reads the weights from file.
    """
    n2, nu = [], []
    with open(fname) as ifile:
        for line in ifile:
            tmp1, tmp2 = line.rstrip().split(", ")
            n2.append(gmp.mpfr(tmp1))
            nu.append(gmp.mpfr(tmp2))
    return np.array(n2), np.array(nu)


def compute_ng(L, mpi, mk, mphi, beta):
    """
    Computes the finite-volume Euclidean correlation functions.
    """
    n2, nu = read(f"weights_{L}.txt")
    L = gmp.mpfr(L)

    p2 = (2*PI/L)**2 * n2
    Ek = np.array([2 * gmp.sqrt(mk**2 + p2_i) for p2_i in p2])
    # Evaluate the retarded Green function at z = Matsubara frequencies
    z = matsubara(beta, boson=False)
    weight = (2*np.pi / 2) * (mphi**2 * mpi) / (mpi*L)**3
    weight = weight * nu / Ek**2
    # Normalization of the spectral function
    # This doesn't involve the factor of -1/pi from converting the imaginary
    # part back to the coefficient of the pole
    norm = gmp.fsum(weight)
    weight = -weight/PI
    ng = []
    for zi in z:
        ng_i = gmp.mpc(0, 0)
        for wj, Ej in zip(weight, Ek):
            pos = gmp.mpfr("1") / gmp.sub(zi, Ej)  # Poles on positive real axis
            neg = gmp.mpfr("1") / gmp.add(zi, Ej)  # Poles on negative real axis
            ng_i = gmp.add(ng_i,
                           gmp.mul(wj, gmp.add(pos, neg)))
        ng.append(ng_i)
    ng = np.array(ng)
    ng = ng / norm  # Remove overall normalization, to be reintroduced later
    ng = ng / gmp.mpfr("2")  # KLUDGE
    fig, axarr = plt.subplots(ncols=2)
    ax1, ax2 = axarr
    x = [zi.imag for zi in z]
    y = [ngi.real for ngi in ng]
    ax1.errorbar(x, y)
    y = [ngi.imag for ngi in ng]
    ax2.errorbar(x, y)
    fig.savefig(f"tmp{L}.png")
    return z, ng, float(norm)


if __name__ == '__main__':
    main()