// -*- C++ -*-
/*! \file
 * \brief Inline NPR momentum-fraction measurement for the quark EMT
 *
 * RI/MOM vertex functions for the quark energy-momentum tensor: one point
 * source, four sequential solves through O_{mu mu}, then momentum projection
 * by FFT (or by an explicit phase sum). Reproduces compute_npr_through_op
 * from the QLUA npr_momfrac script used for the <x> renormalization in
 * Detmold et al., arXiv:2009.05522, Appendix A.
 *
 * The fermion action is built from Chroma's own factory, so retargeting the
 * measurement to a different action -- the exponential clover in particular --
 * is an edit to <FermionAction> in the input XML and nothing else.
 */

#ifndef __inline_npr_momfrac_h__
#define __inline_npr_momfrac_h__

#include "chromabase.h"
#include "meas/inline/abs_inline_measurement.h"
#include "io/xml_group_reader.h"
#include "handle.h"

#include <vector>

namespace Chroma
{
  /*! \ingroup inlinehadron */
  namespace InlineNprMomfracEnv
  {
    extern const std::string name;
    bool registerAll();
  }

  //! Parameter structure
  /*! \ingroup inlinehadron */
  struct InlineNprMomfracParams
  {
    InlineNprMomfracParams();
    InlineNprMomfracParams(XMLReader& xml_in, const std::string& path);
    void write(XMLWriter& xml_out, const std::string& path);

    unsigned long frequency;

    bool          dump_gamma;   // Task 2: dump Gamma(1<<mu) and exit the measurement
    bool          test_project; // Task 3: self-test the momentum projection
    bool          test_seqsrc;  // Task 4: self-test the sequential source
    bool          test_fft;     // self-test: FFT against phase sum, random field
    bool          have_tsrc;    // false until drawn, if absent from the XML
    bool          have_tsrc_from_xml;
    multi1d<int>  tsrc;         // source point y
    multi1d<Real> bvec;         // twist, (0,0,0,1/2) for antiperiodic time

    // Momenta: exactly one of mom_list or ksq_cut. h_cut and all_pos refine
    // ksq_cut only; an explicit list is taken exactly as written.
    bool                  have_mom_list;
    multi1d<multi1d<int> > mom_list;
    bool                  have_ksq_cut;  // the ball sum_mu k_mu^2 <= ksq_cut
    int                   ksq_cut;
    double                h_cut;         // keep h(k) <= h_cut; default 1 = no cut
    bool                  all_pos;       // keep k_mu >= 0 only; default true

    std::string projection;     // FFT (default) or PHASE_SUM
    bool        check_fft;      // FFT only: also phase-sum, log the difference

    std::string output_type;    // TEXT (HDF5 not implemented yet)
    std::string output_file;

    GroupXML_t fermact;         // THE extension point -- swap the action here
    GroupXML_t invParam;
    struct NamedObject_t
    {
      std::string gauge_id;
    } named_obj;
  };

  //! Inline measurement of NPR momentum-fraction vertex functions
  /*! \ingroup inlinehadron */
  class InlineNprMomfrac : public AbsInlineMeasurement
  {
  public:
    ~InlineNprMomfrac() {}
    InlineNprMomfrac(const InlineNprMomfracParams& p) : params(p) {}
    InlineNprMomfrac(const InlineNprMomfrac& p) : params(p.params) {}

    unsigned long getFrequency(void) const {return params.frequency;}

    //! Do the measurement
    void operator()(const unsigned long update_no,
		    XMLWriter& xml_out);

  protected:
    //! Do the measurement
    void func(const unsigned long update_no,
	      XMLWriter& xml_out);

  private:
    //! Draw tsrc uniformly on the lattice from the current RNG state
    void drawSourcePoint();

    //! Project each field at every momentum, by the method <projection> names
    multi2d<DPropagator> project(const std::vector<const LatticePropagator*>& Fs,
                                 const multi1d<multi1d<int> >& moms,
                                 bool check) const;

    //! The measurement proper
    void measure(XMLWriter& xml_out);

    //! Self-tests, each run instead of the measurement
    void dumpGamma() const;
    void testProject() const;
    void testSeqSource() const;
    void testFft() const;

    InlineNprMomfracParams params;
  };

}

#endif
