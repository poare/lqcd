#include <iostream>
#include <omp.h>
#include <chrono>

#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

#include "nevanlinna.hpp"

int main(int argc, char const *argv[]) {

    // Initialize precision
    mpf_set_default_prec(PRECISION);
    std::cout.precision(DIGITS);
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(DIGITS));

    std::cout << std::endl << "Running Nevanlinna." << std::endl;

    assertm (argc == 4, "Requires 3 arguments: input file name; output file name; eta value.");
    std::string in_name = argv[1];
    std::string out_name = argv[2];
    std::string eta_str = argv[3];
    std::cout << "Reading input from: " << in_name << std::endl;
    std::cout << "Writing output to: " << out_name << std::endl;
    std::cout << "Running reconstruction with eta = " << eta_str << std::endl;

    // Read data from input file
    H5Reader<mpfr::mpreal> reader (in_name);
    Prec<mpfr::mpreal>::NVector freqs = reader.get_freqs();         // Matsubara frequencies i\omega_n
    Prec<mpfr::mpreal>::NVector ng = reader.get_ng();               // Green's function data -G(i\omega_n)
    // int beta = freqs.size();
    int beta = reader.get_beta();
    std::cout << "Number of measured Matsubara frequencies: " << freqs.size() << std::endl;
    std::cout << "Beta is: " << beta << std::endl;

    std::cout << std::endl << "Matsubara frequencies:" << std::endl;
    print_vector<mpfr::mpreal>(freqs);
    std::cout << std::endl << "Green's function at Matsubara frequencies:" << std::endl;
    print_vector<mpfr::mpreal>(ng);
    
    Prec<mpfr::mpreal>::NReal eta (eta_str);
    Nevanlinna<mpfr::mpreal> nevanlinna (freqs, ng);
    
    
    std::cout << std::endl << "Phi vals: " << std::endl;
    print_vector<mpfr::mpreal>(nevanlinna.get_schur().get_phi());
    
    // Read evaluation axis parameters
    double start = reader.get_start();
    double stop = reader.get_stop();
    int num = reader.get_num();             // Number of points for recon
    std::cout << "Evaluation axis runs from " << start << " to " << stop << " with " << num << " evaluation points." << std::endl;

    RealDomainData<mpfr::mpreal> omegas;
    Prec<mpfr::mpreal>::NVector rho_recon_disk;
    Prec<mpfr::mpreal>::NVector rho_recon;
    std::tie(omegas, rho_recon_disk) = nevanlinna.evaluate(start, stop, num, eta);
    rho_recon = Nevanlinna<mpfr::mpreal>::inv_cayley(rho_recon_disk);

    std::cout << "Spectral function reconstructed." << std::endl;

    // Write to output file
    H5Writer<mpfr::mpreal> fout (out_name, beta, start, stop, num, eta, freqs, ng, rho_recon, nevanlinna);
    fout.write();
    std::cout << std::endl << "Recon written to: " << out_name << std::endl;

    return 0;
}