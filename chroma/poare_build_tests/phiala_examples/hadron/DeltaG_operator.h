// -*- C++ -*-
// $Id: weinberg_operator_w.h,v 3.0 2006/04/03 04:58:59 edwards Exp $
/*! \file
 *  \brief New measurement
 */

#ifndef __DeltaG_operator_h__
#define __DeltaG_operator_h__

namespace Chroma {

  //! Some measurement
  /*!
   * \ingroup hadron
   *
   * This routine is specific to Wilson fermions! 
   *
   * Do something
   *
   */


  void measureDeltaGn2Test( const multi1d<LatticeColorMatrix>& u, const SftMom& phases,
			    XMLWriter& xml,
			    const std::string& xml_out, int tube, multi1d<int> tsrc);
  void measureFFtilUncontracted( const multi1d<LatticeColorMatrix>& u, const SftMom& phases,
				 XMLWriter& xml,
				 const std::string& xml_out, int tube, multi1d<int> tsrc);
    
  void measureF2( const multi1d<LatticeColorMatrix>& u,
		  const SftMom& phases,
		  XMLWriter& xml,
		  const std::string& xml_out);

  void measureF2threeindex( const multi1d<LatticeColorMatrix>& u,
			    const SftMom& phases,
			    XMLWriter& xml,
			    const std::string& xml_out);

  void measureF2threeindexNEW( const multi1d<LatticeColorMatrix>& u,
			       const SftMom& phases,
			       XMLWriter& xml,
			       const std::string& xml_out);
    
  void measureF2fourindex( const multi1d<LatticeColorMatrix>& u,
			   const SftMom& phases,
			   XMLWriter& xml,
			   const std::string& xml_out);

  void measureFFtil( const multi1d<LatticeColorMatrix>& u,
		     const SftMom& phases,
		     XMLWriter& xml,
		     const std::string& xml_out);
    
  void measureDeltaGn3( const multi1d<LatticeColorMatrix>& u,
			const SftMom& phases,
			XMLWriter& xml,
			const std::string& xml_out);
    
  void measureGlueball( const multi1d<LatticeColorMatrix>& u,
			const SftMom& phases,
			XMLWriter& xml,
			const std::string& xml_out);
    
  void measuregluon( const multi1d<LatticeColorMatrix>& u,
		     const SftMom& phases,
		     XMLWriter& xml,
		     const std::string& xml_out);
    
  void measuregluon3( const multi1d<LatticeColorMatrix>& u,
		      const SftMom& phases,
		      XMLWriter& xml,
		      const std::string& xml_out);

  void measuregluonBALL( const multi1d<LatticeColorMatrix>& u,
                         const SftMom& phases,
                         XMLWriter& xml,
			 int ball_size, multi1d<int> tsrc,
                         const std::string& xml_out);
    
  multi1d<LatticeComplex> constructGlueballs(const multi1d<LatticeColorMatrix> & u);
  LatticeComplex constructF2Test(const multi1d<LatticeColorMatrix> & u);
  LatticeComplex constructF2TestChroma(const multi1d<LatticeColorMatrix> & u);
  multi3d<LatticeComplex> GetCloverFmunu(const multi1d<LatticeColorMatrix> & u, int clover_loops);
  multi3d<LatticeComplex> CalculateDumbFmunu(const multi1d<LatticeColorMatrix> & u);
  multi2d<LatticeColorMatrix> CalculateCloverTerm(const multi1d<LatticeColorMatrix> & u, int musize, int nusize);
  const LatticeColorMatrix multishift(const LatticeColorMatrix & uu, int sign, int dir, int mag);
    
    
  multi1d<LatticeComplex> constructDeltaGn2Test(const multi1d<LatticeColorMatrix> & u);

  multi1d<LatticeComplex> constructFFtilUncontracted(const multi1d<LatticeColorMatrix> & u);
    
  multi1d<LatticeComplex> constructDeltaGn3(const multi1d<LatticeColorMatrix> & u);
    
  multi3d<LatticeComplex> GetChromaFmunu(const multi1d<LatticeColorMatrix> & u);
  
  multi1d<LatticeComplex> constructF2threeindex(const multi1d<LatticeColorMatrix> & u);

  multi1d<LatticeComplex> constructF2fourindex(const multi1d<LatticeColorMatrix> & u);

  multi1d<LatticeComplex> constructF2threeindexNEW(const multi1d<LatticeColorMatrix> & u);

  multi1d<LatticeComplex> constructF2(const multi1d<LatticeColorMatrix> & u);
    
  multi1d<LatticeComplex> constructFFtil(const multi1d<LatticeColorMatrix> & u);
    
  multi2d<LatticeComplex> ConstructGluon(const multi1d<LatticeColorMatrix> & u);
    
    
  double LeviCivita4D(int ii, int jj, int kk, int ll);
    
  void constructSU3generators(multi1d < ColorMatrix > & tSU3);

  Real fabc(int a, int b, int c);
  Real dabc(int a, int b, int c);

}  // end namespace Chroma


#endif
