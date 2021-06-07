// -*- C++ -*-
/*! \file
 *  \brief 0nubb measurement header
 */

#ifndef __zero_nubb_h__
#define __zero_nubb_h__

namespace Chroma {

//! Computes four-point function for use in 0nubb renormalization calculation
/* This routine should be copied into the /data/d10b/users/poare/software/src-new/chroma/lib/meas/hadron directory
 * before the chroma source is remade.
 *
 * Construct RI/sMOM 4-point function for 0nubb renormalization and writes in COMPLEX
 *
 * Three propagators should be passed in at k1, k2, and q, subject to q = k2 - k1
 *
 * \param quark_prop_1  first quark propagator, computed with wall source of momentum k1 ( Read )
 * \param quark_prop_2  second (anti-) quark propagator, computed with wall source of momentum k2 ( Read )
 * \param quark_prop_3  third quark propagator, computed with wall source of momentum q
 * \param k 						momentum index, k1 = (-k, 0, k, 0), k2 = (0, k, k, 0), k3 = (k, k, 0, 0)
 * \param xml           xml file object ( Write )
 * \param xml_group     std::string used for writing xml data ( Read )
 */

void zero_nubb(const LatticePropagator& quark_prop_k1,
	     const LatticePropagator& quark_prop_k2,
			 const LatticePropagator& quark_prop_q,
			 int k,
			 bool ferm_bc,
	     XMLWriter& xml,
	     const std::string& xml_group) ;

}  // end namespace Chroma

#endif
