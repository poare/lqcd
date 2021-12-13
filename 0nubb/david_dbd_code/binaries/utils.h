#ifndef __UTILS_H_INCLUDED__
#define __UTILS_H_INCLUDED__

// C++
#include <limits>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <vector>

// CPS
#include <alg/wilson_matrix.h>

#include "color_matrix.h"

CPS_START_NAMESPACE

// Some simple utility functions used elsewhere

static void compute_coord(int x[4], const int size[4], int i)
{
	for(int j=0; j<4; ++j){
		x[j] = i % size[j];
		i /= size[j];
	}
}

static void compute_spatial_coord(int x[3], const int size[3], int i)
{
	for(int j=0; j<3; ++j){
		x[j] = i % size[j];
		i /= size[j];
	}
}

static void compute_coord_ap(int x[4], const int size[4], int i, int glb_t)
{
	for(int j=0; j<4; ++j){
		x[j] = i % size[j];
		i /= size[j];
	}
	if(i != 0){ x[3] += glb_t; }
}

static void compute_coord(int x[4], const int delta[4], const int low[4], int i)
{
  for(int j=0; j<4; ++j){
    x[j] = i % delta[j] + low[j];
    i /= delta[j];
  }
}

// Convert 4D spacetime index (x) to vector index [i]
static int compute_vector_idx(const int x[4], const int size[4])
{
	int idx(0);
	for(int j=3; j>=0; --j){ idx = idx*size[j] + x[j]; }
	return idx;
}

static int compute_id(const int x[4], const int size[4])
{
  int ret(0);
  for(int j=3; j>=0; j--){ ret = ret * size[j] + x[j]; }
  return ret;
}

// Propagator types
enum PROP_TYPE {
	PROP_P,		// Periodic (P)
	PROP_A,		// Antiperiodic (A)
	PROP_PA		// P+A
};

template<typename type>
std::string toString(type v)
{
	std::stringstream ss;
	ss << v;
	return ss.str();
}

// Supported gamma matrix insertions at source and sink
enum Operator {
	GAMMA_0, GAMMA_1, GAMMA_2, GAMMA_3, GAMMA_5, ID, 
	GAMMA_05, GAMMA_15, GAMMA_25, GAMMA_35,
	GAMMA_50, GAMMA_51, GAMMA_52, GAMMA_53
};

template<class T>
static bool isZero(T& x)
{
	return (x < std::numeric_limits<T>::epsilon() && x > -std::numeric_limits<T>::epsilon()) ? true : false;
}


class ChromaGammaBasis
{
  private:
    const char* cname = "ChromaGammaBasis";
    std::vector<WilsonMatrix> gamma;

  public:
    ChromaGammaBasis()
    {
      const char* fname = "ChromaGammaBasis()";

      gamma.resize(16);

      WilsonMatrix id(0), gx(0), gy(0), gz(0), gt(0);
      for(int s=0; s<4; s++){
      for(int c=0; c<3; c++){
        Rcomplex tmp(1.0,0.0);
        id.Element(s,c,s,c,tmp);
      }}
      gx = id.glV(0);
      gy = id.glV(1);
      gz = id.glV(2);
      gt = id.glV(3);

      gamma[0]  = id;
      gamma[1]  = gx;
      gamma[2]  = gy;
      gamma[3]  = gx * gy;
      gamma[4]  = gz;
      gamma[5]  = gx * gz;
      gamma[6]  = gy * gz;
      gamma[7]  = gx * gy * gz;
      gamma[8]  = gt;
      gamma[9]  = gx * gt;
      gamma[10] = gy * gt;
      gamma[11] = gx * gy * gt;
      gamma[12] = gz * gt;
      gamma[13] = gx * gz * gt;
      gamma[14] = gy * gz * gt;
      gamma[15] = gx * gy * gz * gt;
    }

    const WilsonMatrix& operator()(int i) const { return gamma[i]; }
};

inline WilsonMatrix hconj(const WilsonMatrix& M)
{
  WilsonMatrix Mdag(M);
  Mdag.hconj();
  return Mdag;
}

static SpinMatrix ExtractColor(const WilsonMatrix& S, int c1, int c2)
{
  SpinMatrix ret(0.0);
  for(int sp1=0; sp1<4; sp1++){
  for(int sp2=0; sp2<4; sp2++){
    ret(sp1, sp2) = S(sp1, c1, sp2, c2);
  }}
  return ret;
}

static ColorMatrix ExtractSpin(const WilsonMatrix& S, int sp1, int sp2)
{
  ColorMatrix ret(0.0);
  for(int c1=0; c1<3; c1++){
  for(int c2=0; c2<3; c2++){
    ret(c1, c2) = S(sp1, c1, sp2, c2);
  }}
  return ret;
}

inline Rcomplex Trace(const SpinMatrix& S){ return S.Tr(); }

inline bool file_exists(const std::string& fname) 
{
  struct stat buffer;
  return ( stat(fname.c_str(), &buffer) == 0 );
}

CPS_END_NAMESPACE

#endif
