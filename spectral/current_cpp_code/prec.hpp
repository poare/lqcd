/**
 * @file prec.hpp
 * @author Patrick Oare
 * @brief 
 * @version 0.1
 * @date 2022-11-10
 * 
 * @copyright Copyright (c) 2022
 * 
 * Template class for precision numbers.
 */

#ifndef PRECISION_H
#define PRECISION_H

#include <iostream>
#include <complex>
#include <vector>
#include <time.h>
#include <cmath>
// #include <mpfr.h>
#include <eigen3/unsupported/Eigen/MPRealSupport>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <mpreal.h>
#include <gmpxx.h>

// ************************************************************ //
// ************************** Classes ************************* //
// ************************************************************ //

// const mpfr_prec_t PRECISION    = mpfr_prec_t(1028);
// const int DIGITS               = 128;
const mpfr_prec_t PRECISION    = mpfr_prec_t(256);
const int DIGITS               = 64;
// const mpfr_prec_t PRECISION    = mpfr_prec_t(128);
// const int DIGITS               = 31;
const mpfr_rnd_t RRND          = MPFR_RNDN;
const std::string MACHINE_EPS  = "1e-16";

// mpf_set_default_prec(PRECISION);
// std::cout.precision(DIGITS);
// mpfr::mpreal::set_default_prec(mpfr::digits2bits(DIGITS));

/**
 * @brief Template class for precision numbers. The abstract class T should be a 
 * number type, generally either a double or mpfr::mpreal.
 * Note that templates should all be defined in .h files: 
 *          https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl
 * 
 * @tparam T Number type to use (double, mpfr::mpreal)
 */
template <class T>
class Prec {
    public:
        // Precision types
        using NReal = T;
        using NComplex = std::complex<T>;
        using NVector = std::vector<NComplex>;
        using NMatrix = Eigen::Matrix<NComplex, Eigen::Dynamic, Eigen::Dynamic>;
        using NVectorArray = std::vector<NVector>;
        using NMatrixArray = std::vector<NMatrix>;

        // Constants
        static const NComplex ZERO;
        static const NComplex ONE;
        static const NComplex I;
        static const NComplex TWO;
        static const NComplex SQRT2;
        static const NComplex PI;

        static const NReal DEFAULT_ETA;
        static const NReal EPSILON;         // For comparison to zero

        // Getters
        static NComplex get_zero()           {return ZERO;}
        static NComplex get_one()            {return ONE;}
        static NComplex get_i()              {return I;}
        static NComplex get_two()            {return TWO;}
        static NComplex get_sqrt2()          {return SQRT2;}
        static NComplex get_pi()             {return PI;}
        static NReal    get_default_eta()    {return DEFAULT_ETA;}
        static NReal    get_epsilon()        {return EPSILON;}

        static NVector row_to_vector(NMatrix rowvec);
        static NVector col_to_vector(NMatrix colvec);
        static NVector mat_to_vec(const NMatrix mat);
        static NMatrix vec_to_mat(const NVector vec);
        static NMatrix vecarray_to_mat(const NVectorArray va);
};

// DEFINE CONSTANTS
template <> const typename Prec<double>::NComplex Prec<double>::                ZERO(Prec<double>::NComplex{0.0, 0.0});
template <> const typename Prec<mpfr::mpreal>::NComplex Prec<mpfr::mpreal>::    ZERO(Prec<mpfr::mpreal>::NComplex{mpfr::mpreal("0.0"), mpfr::mpreal("0.0")});

template <> const typename Prec<double>::NComplex Prec<double>::                ONE(Prec<double>::NComplex{1.0, 0.0});
template <> const typename Prec<mpfr::mpreal>::NComplex Prec<mpfr::mpreal>::    ONE(Prec<mpfr::mpreal>::NComplex{mpfr::mpreal("1.0"), mpfr::mpreal("0.0")});

template <> const typename Prec<double>::NComplex Prec<double>::                I(Prec<double>::NComplex{0.0, 1.0});
template <> const typename Prec<mpfr::mpreal>::NComplex Prec<mpfr::mpreal>::    I(Prec<mpfr::mpreal>::NComplex{mpfr::mpreal("0.0"), mpfr::mpreal("1.0")});

template <> const typename Prec<double>::NComplex Prec<double>::                TWO(Prec<double>::NComplex{2.0, 0.0});
template <> const typename Prec<mpfr::mpreal>::NComplex Prec<mpfr::mpreal>::    TWO(Prec<mpfr::mpreal>::NComplex{mpfr::mpreal("2.0"), mpfr::mpreal("0.0")});

template <> const typename Prec<double>::NComplex Prec<double>::                PI(Prec<double>::NComplex{M_PI, 0.0});
template <> const typename Prec<mpfr::mpreal>::NComplex Prec<mpfr::mpreal>::    PI(Prec<mpfr::mpreal>::NComplex{mpfr::const_pi(mpfr::digits2bits(DIGITS)), mpfr::mpreal("0.0")});

template <> const typename Prec<double>::NComplex Prec<double>::                SQRT2(Prec<double>::NComplex{sqrt(2.0), 0.0});
template <> const typename Prec<mpfr::mpreal>::NComplex Prec<mpfr::mpreal>::    SQRT2(Prec<mpfr::mpreal>::NComplex{sqrt(mpfr::mpreal("2.0", mpfr::digits2bits(DIGITS))), mpfr::mpreal("0.0")});

template <> const typename Prec<double>::NReal Prec<double>::                   DEFAULT_ETA(1e-5);
template <> const typename Prec<mpfr::mpreal>::NReal Prec<mpfr::mpreal>::       DEFAULT_ETA(mpfr::mpreal("1e-5", mpfr::digits2bits(DIGITS)));

// template <> const typename Prec<double>::NReal Prec<double>::                   EPSILON(1e-10);
// template <> const typename Prec<mpfr::mpreal>::NReal Prec<mpfr::mpreal>::       EPSILON(mpfr::mpreal("1e-10", mpfr::digits2bits(DIGITS)));
template <> const typename Prec<double>::NReal Prec<double>::                   EPSILON(1e-48);
template <> const typename Prec<mpfr::mpreal>::NReal Prec<mpfr::mpreal>::       EPSILON(mpfr::mpreal("1e-48", mpfr::digits2bits(DIGITS)));

// ************************************************************ //
// ********************* Utility functions ******************** //
// ************************************************************ //

/**
 * @brief Converts an Prec<T>::NMatrix row vector to a Prec<T>::NVector. Intended 
 * to be used in conjunction with .row, i.e. if m is a Prec<T>::NMatrix and one wants 
 * to access its k'th row, use row_to_vector(m.row(k))
 * 
 * @tparam T Base type.
 * @param rowvec Row vector to convert.
 * @return Prec<T>::NVector Converted row vector.
 */
template <class T>
typename Prec<T>::NVector Prec<T>::row_to_vector(Prec<T>::NMatrix rowvec) {
    int m = rowvec.cols();
    Prec<T>::NVector outvec(m);
    for (int i = 0; i < m; i++) {
        // outvec[i] = rowvec(1, i);             // this errors
        outvec[i] = rowvec(i);
    }
    return outvec;
}

/**
 * @brief Converts an Prec<T>::NMatrix column vector to a Prec<T>::NVector. Intended 
 * to be used in conjunction with .cols, i.e. if m is a Prec<T>::NMatrix and one wants 
 * to access its k'th column, use col_to_vector(m.col(k))
 * 
 * @tparam T Base type.
 * @param colvec Column vector to convert.
 * @return Prec<T>::NVector Converted column vector.
 */
template <class T>
typename Prec<T>::NVector Prec<T>::col_to_vector(Prec<T>::NMatrix colvec) {
    int n = colvec.rows();
    Prec<T>::NVector outvec(n);
    for (int i = 0; i < n; i++) {
        outvec[i] = colvec(i, 1);
    }
    return outvec;
}

/**
 * @brief Prints a complex number of base type T.
 * 
 * @tparam T Base type of complex number.
 * @param c Complex number to print.
 */
template <class T>
void print_complex(const typename Prec<T>::NComplex c) {
    std::cout << c.real() << " + " << c.imag() << "i";
}

/**
 * @brief Prints a complex vector of base type T.
 * 
 * @tparam T Base type of vector.
 * @param v Vector to print.
 */
template <class T>
void print_vector(const typename Prec<T>::NVector v) {
    std::cout << "[";
    for (int i = 0; i < v.size(); i++) {
        print_complex<T>(v[i]);
        if (i < v.size() - 1) {
            std::cout << ", ";
        } else {
            std::cout << "]" << std::endl;
        }   
    }
}

/**
 * @brief Prints an array of complex vectors of base type T.
 * 
 * @tparam T Base type of vector array.
 * @param v Vector array to print.
 */
template <class T>
void print_vector_array(const typename Prec<T>::NVectorArray va) {

    std::cout << "[" << std::endl;
    for (int i = 0; i < va.size(); i++) {
        print_vector<T>(va[i]);
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Prints a complex matrix of base type T.
 * 
 * @tparam T Base type of matrix.
 * @param m Matrix to print.
 */
template <class T>
void print_matrix(const typename Prec<T>::NMatrix m) {
    std::cout << "[" << std::endl;
    for (int i = 0; i < m.rows(); i++) {
        print_vector<T>(Prec<T>::row_to_vector(m.row(i)));
    }
    std::cout << "]" << std::endl;
}

template <class T>
bool is_zero(const typename Prec<T>::NReal r) {
    return abs(r) < Prec<T>::EPSILON;
}

template <class T>
bool is_zero(const typename Prec<T>::NComplex c) {
    return is_zero<T>(c.real()) && is_zero<T>(c.imag());
}

/**
 * @brief Returns the real part of a NVector as a string.
 * 
 * @tparam T 
 * @return std::vector<std::string> 
 */
template <class T>
std::vector<std::string> vec_to_rstring(const typename Prec<T>::NVector vec) {
    int n = vec.size();
    std::vector<std::string> out_str (n);
    for (int i = 0; i < n; i++) {
        out_str[i] = vec[i].real().toString(DIGITS);
    }
    return out_str;
}

/**
 * @brief Returns the imaginary part of a NVector as a string.
 * 
 * @tparam T 
 * @return std::vector<std::string> 
 */
template <class T>
std::vector<std::string> vec_to_istring(const typename Prec<T>::NVector vec) {
    int n = vec.size();
    std::vector<std::string> out_str (n);
    for (int i = 0; i < n; i++) {
        out_str[i] = vec[i].imag().toString(DIGITS);
    }
    return out_str;
}

template <class T>
typename Prec<T>::NVector Prec<T>::mat_to_vec(const typename Prec<T>::NMatrix mat) {
    int len = mat.size();
    Prec<T>::NVector out (len);
    for (int i = 0; i < len; i++) {
        out[i] = mat(i);
    }
    return out;
}

template <class T>
typename Prec<T>::NMatrix Prec<T>::vec_to_mat(const typename Prec<T>::NVector vec) {
    int len = vec.size();
    Prec<T>::NMatrix out (len, 1);
    for (int i = 0; i < len; i++) {
        out(i) = vec[i];
    }
    return out;
}

template <class T>
typename Prec<T>::NMatrix Prec<T>::vecarray_to_mat(const typename Prec<T>::NVectorArray va) {
    int n = va.size();
    int m = va[0].size();
    Prec<T>::NMatrix out (n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out(i, j) = va[i][j];
        }
    }
    return out;
}

#endif