import numpy as np
import gmpy2 as gmp
import io
import h5py

# Set precision for gmpy2 and initialize complex numbers
prec = 128
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
ONE = gmp.mpc(1, 0)
I = gmp.mpc(0, 1)

def write_gmp_input_h5(fname, beta, freqs, ng, start, stop, num):
    """
    Writes spectral function data to the appropriate file.

    freqs : list of integers
    ng : list of gmp.mpc
    """
    freqs_imag = [str(p.imag) for p in freqs]
    ng_real = [str(p.real) for p in ng]
    ng_imag = [str(p.imag) for p in ng]
    f = h5py.File(fname, 'w')
    f['beta'] = beta
    f['freqs/imag'] = freqs_imag
    f['ng/real'] = ng_real
    f['ng/imag'] = ng_imag
    f['start'] = start
    f['stop'] = stop
    f['num'] = num
    f.close()
    return fname

def read_gmp_input_h5(fname):
    """
    Reads input simulation data from the appropriate file. Note that the HDF5 file is 
    assumed to contain the extended precision numbers as strings. 
    """
    f = h5py.File(fname, 'r')
    beta = f['beta'][()]
    freqs_imag = f['freqs/imag'][()]
    ng_real = f['ng/real'][()]        # encoded as byte strings
    ng_imag = f['ng/imag'][()]
    freqs = np.zeros((beta), dtype = object)
    ng = np.zeros((beta), dtype = object)
    for ii in range(beta):
        tmp_real = gmp.mpfr(ng_real[ii].decode('utf-8'))
        tmp_imag = gmp.mpfr(ng_imag[ii].decode('utf-8'))
        freqs[ii] = I * gmp.mpfr(freqs_imag[ii].decode('utf-8'))
        ng[ii] = ONE * tmp_real + I * tmp_imag
    f.close()
    return beta, freqs, ng


def read_gmp_output_h5(fname):
    """
    Reads input simulation data from the appropriate file. Note that the HDF5 file is 
    assumed to contain the extended precision numbers as strings. 
    """
    f = h5py.File(fname, 'r')
    beta = f['beta'][(0)]
    start = f['start'][(0)]
    stop = f['stop'][(0)]
    num = f['num'][(0)]
    eta = gmp.mpfr(f['eta'][(0)].decode('utf-8'))

    data_keys = ['freqs', 'ng', 'phis', 'recon', 'a_vec', 'b_vec', 'c_vec', 'd_vec']
    data = [np.zeros((beta), dtype = object), np.zeros((beta), dtype = object), np.zeros((beta), dtype = object), np.zeros((num), dtype = object), \
        np.zeros((num), dtype = object), np.zeros((num), dtype = object), np.zeros((num), dtype = object), np.zeros((num), dtype = object)]
    for k, key in enumerate(data_keys):
        re = f[key + '_real'][()]
        im = f[key + '_imag'][()]
        for ii in range(len(re)):
            tmp_real = gmp.mpfr(re[ii].decode('utf-8'))
            tmp_imag = gmp.mpfr(im[ii].decode('utf-8'))
            data[k][ii] = ONE * tmp_real + I * tmp_imag
    [freqs, ng, phis, recon, avec, bvec, cvec, dvec] = data
    abcd = [[avec, bvec], [cvec, dvec]]
    return beta, start, stop, num, eta, freqs, ng, phis, recon, abcd

def read_gmp_output_disk_h5(fname):
    """
    Reads input simulation data from the appropriate file. Note that the HDF5 file is 
    assumed to contain the extended precision numbers as strings. 
    """
    f = h5py.File(fname, 'r')
    beta = f['beta'][(0)]
    start = f['start'][(0)]
    stop = f['stop'][(0)]
    num = f['num'][(0)]
    eta = gmp.mpfr(f['eta'][(0)].decode('utf-8'))

    data_keys = ['freqs', 'zeta_list', 'ng', 'w_list', 'phi', 'recon', 'P', 'Q', 'R', 'S', 'eig']
    print(f['S_real'][()])
    data = [np.zeros((beta), dtype = object), np.zeros((beta), dtype = object), np.zeros((beta), dtype = object), np.zeros((beta), dtype = object), \
        np.zeros((num), dtype = object), np.zeros((num), dtype = object), np.zeros((num), dtype = object), np.zeros((num), dtype = object), \
        np.zeros((num), dtype = object), np.zeros((num), dtype = object)]
    for k, key in enumerate(data_keys):
        re = f[key + '_real'][()]
        im = f[key + '_imag'][()]
        for ii in range(len(re)):
            tmp_real = gmp.mpfr(re[ii].decode('utf-8'))
            tmp_imag = gmp.mpfr(im[ii].decode('utf-8'))
            data[k][ii] = ONE * tmp_real + I * tmp_imag
    [freqs, zetas, ng, w, phis, recon, P, Q, R, S, eig] = data
    nev_coeffs = [[P, Q], [R, S]]
    return beta, start, stop, num, eta, freqs, zetas, ng, w, phis, recon, nev_coeffs, eig