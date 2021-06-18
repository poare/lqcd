/*! \file
 *  \brief Momentum (wall) source construction
 */

#include "chromabase.h"

#include "meas/sources/source_const_factory.h"
#include "meas/sources/ferm_mom_source_const.h"
#include "util/ft/sftmom.h"
#include "util/ferm/transf.h"

namespace Chroma
{
  // Read parameters
  void read(XMLReader& xml, const std::string& path, FermMomWallQuarkSourceConstEnv::Params& param)
  {
    FermMomWallQuarkSourceConstEnv::Params tmp(xml, path);
    param = tmp;
  }

  // Writer
  void write(XMLWriter& xml, const std::string& path, const FermMomWallQuarkSourceConstEnv::Params& param)
  {
    param.writeXML(xml, path);
  }

  // boxfil was defined here, if the build doesn't work then we may need to import mom_source_const.cc as well

  //! Fill a specific color and spin index with 1.0 within a volume
  /*! \ingroup sources */
  void boxfil2(LatticeFermion& a, int color_index, int spin_index)
  {
    START_CODE();

    if (color_index >= Nc || color_index < 0)
      QDP_error_exit("invalid color index", color_index);

    if (spin_index >= Ns || spin_index < 0)
      QDP_error_exit("invalid spin index", spin_index);

    // Write ONE to all field
    Real one = 1;
    Complex sitecomp = cmplx(one,0);
    ColorVector sitecolor = zero;
    Fermion sitefield = zero;

    pokeSpin(sitefield,
	     pokeColor(sitecolor,sitecomp,color_index),
	     spin_index);

    // Broadcast to all sites
    a = sitefield;  // QDP (not installed version) now supports   construct OLattice = OScalar

    END_CODE();
  }


  //! Hooks to register the class
  namespace FermMomWallQuarkSourceConstEnv
  {
    namespace
    {
      //! Callback function
      QuarkSourceConstruction<LatticePropagator>* createProp(XMLReader& xml_in,
							     const std::string& path)
      {
	return new SourceConst<LatticePropagator>(Params(xml_in, path));
      }

      //! Name to be used
      const std::string name("FERMION_MOMENTUM_VOLUME_SOURCE");

      //! Local registration flag
      bool registered = false;
    }

    //! Return the name
    std::string getName() {return name;}

    //! Register all the factories
    bool registerAll()
    {
      bool success = true;
      if (! registered)
      {
	success &= Chroma::ThePropSourceConstructionFactory::Instance().registerObject(name, createProp);
	registered = true;
      }
      return success;
    }


    //! Initialize
    Params::Params()
    {
      j_decay = -1;
      av_mom = false ;
    }


    //! Read parameters
    Params::Params(XMLReader& xml, const std::string& path)
    {
      XMLReader paramtop(xml, path);

      int version;
      read(paramtop, "version", version);

      switch (version)
      {
      case 1:
	break;

      default:
	QDPIO::cerr << __func__ << ": parameter version " << version
		    << " unsupported." << std::endl;
	QDP_abort(1);
      }

      read(paramtop, "j_decay", j_decay);
      read(paramtop, "t_srce", t_srce);
      read(paramtop, "av_mom", av_mom) ;
      read(paramtop, "mom", mom);

      if (mom.size() != Nd)
      {
	QDPIO::cerr << name << ": wrong size of mom array: expected length=" << Nd << std::endl;
	QDP_abort(1);
      }
    }


    // Writer
    void Params::writeXML(XMLWriter& xml, const std::string& path) const
    {
      push(xml, path);

      int version = 1;
      write(xml, "version", version);

      write(xml, "mom", mom);
      write(xml, "av_mom", av_mom) ;
      write(xml, "j_decay", j_decay);
      write(xml, "t_srce", t_srce);

      pop(xml);
    }


    //! Construct the source
    template<>
    LatticePropagator
    SourceConst<LatticePropagator>::operator()(const multi1d<LatticeColorMatrix>& u) const
    {
      QDPIO::cout << "Volume Momentum Source with Fermionic BCs" << std::endl;

      LatticeComplex phase ;
      // Initialize the slow Fourier transform phases
      if(params.av_mom){ // TODO implement later
	multi1d<int> mom3(Nd-1);
	for(int mu=0,j=0; mu < Nd; ++mu){
	  if (mu != params.j_decay)
	    mom3[j++] = params.mom[mu];
	}
	//just get one momentum. the one we want!
	SftMom phases(0, params.t_srce, mom3, params.av_mom, params.j_decay);
	mom3 = phases.canonicalOrder(mom3);
	Real fact = twopi * Real(params.mom[params.j_decay]) / Real(Layout::lattSize()[params.j_decay]);
	phase = cos( fact * QDP::Layout::latticeCoordinate(params.j_decay) );
	/**
	for (int sink_mom_num=0; sink_mom_num < phases.numMom(); ++sink_mom_num){
	  multi1d<int> mom = phases.canonicalOrder(phases.numToMom(sink_mom_num));
	  if (mom == mom3)
	    phase *= phases[sink_mom_num];
        }**/
	phase *= phases[0];
	multi1d<int> mom = phases.canonicalOrder(phases.numToMom(0));
	QDPIO::cout<<"Source momentum (averaged over equivalent momenta): " ;
	QDPIO::cout<<mom[0]<<mom[1]<<mom[2]<<std::endl;
      }
      else{ // do not use momentum averaged sources
	// SftMom phases(0, params.t_srce, params.mom);
	// phase = phases[0] ;

  multi1d<int> k = params.mom;
  multi1d<double> bvec;
  bvec.resize(4);
  bvec[0] = 0.0;
  bvec[1] = 0.0;
  bvec[2] = 0.0;
  // bvec[3] = 0.5;    // just to see if it matches the QLUA output
  // TODO TOMORROW: Recompile with bvec set to 0, see if this does anything
  bvec[3] = 0.0;    // Compare to usual momentum source, should be equal
  LatticeReal phase_arg = zero;
  for (int mu = 0; mu < 4; mu++) {
    double comp = (double) k[mu];
    phase_arg += Layout::latticeCoordinate(mu) * LatticeReal(comp + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
    // phase_arg -= Layout::latticeCoordinate(mu) * LatticeReal(comp + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
  }
  LatticeComplex phase = cmplx(cos(phase_arg), sin(phase_arg));

	QDPIO::cout<<"Source momentum: " ;
	QDPIO::cout<<k[0]<<k[1]<<k[2]<<k[3]<<std::endl;
      }

      // Create the quark source
      LatticePropagator quark_source;
      for(int color_source = 0; color_source < Nc; ++color_source)
      {
	for(int spin_source = 0; spin_source < Ns; ++spin_source)
	{
	  // MomWall fill a fermion source. Insert it into the propagator source
	  LatticeFermion chi;
	  boxfil2(chi, color_source, spin_source);
	  // Multiply in the time direction phases (not handled in sftmom)
	  chi *= phase;
	  FermToProp(chi, quark_source, color_source, spin_source);
	}
      }

      return quark_source;
    }

  }

}
