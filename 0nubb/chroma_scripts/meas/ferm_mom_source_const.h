// -*- C++ -*-
/*! \file
 *  \brief Fixed momentum (wall) source construction with fermionic boundary conditions
 */

#ifndef __ferm_mom_source_const_h__
#define __ferm_mom_source_const_h__

#include "meas/sources/source_construction.h"

namespace Chroma
{

  //! Name and registration
  /*! @ingroup sources */
  namespace FermMomWallQuarkSourceConstEnv
  {
    bool registerAll();

    //! Return the name
    std::string getName();

    //! MomWall source parameters
    /*! @ingroup sources */
    struct Params
    {
      Params();
      Params(XMLReader& in, const std::string& path);
      void writeXML(XMLWriter& in, const std::string& path) const;

      multi1d<int>     mom    ;              /*<! prototype momentum */
      int              j_decay;              /*<! time direction */
      multi1d<int>     t_srce ;              /*<! the origin for the FT */
      bool             av_mom ;              /*<! average equivalent momenta */
      bool            ferm_bc;
    };


    //! MomWall source construction
    /*! @ingroup sources
     *
     * Create a momentum wall propagator source
     */
    template<typename T>
    class SourceConst : public QuarkSourceConstruction<T>
    {
    public:
      //! Full constructor
      SourceConst(const Params& p) : params(p) {}

      //! Construct the source
      T operator()(const multi1d<LatticeColorMatrix>& u) const;

    private:
      //! Hide partial constructor
      SourceConst() {}

    private:
      Params  params;   /*!< source params */
    };

  }  // end namespace


  //! Reader
  /*! @ingroup sources */
  void read(XMLReader& xml, const std::string& path, FermMomWallQuarkSourceConstEnv::Params& param);

  //! Writer
  /*! @ingroup sources */
  void write(XMLWriter& xml, const std::string& path, const FermMomWallQuarkSourceConstEnv::Params& param);

}  // end namespace Chroma


#endif
