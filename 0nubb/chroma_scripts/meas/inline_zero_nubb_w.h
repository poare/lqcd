// -*- C++ -*-
/*! \file
 * \brief Inline zero nubb calculations
 *
 * Zero nubb renormalization calculations
 */

#ifndef __inline_zero_nubb_h__
#define __inline_zero_nubb_h__

#include "chromabase.h"
#include "meas/inline/abs_inline_measurement.h"

namespace Chroma
{
  /*! \ingroup inlinehadron */
  namespace InlineZeroNubbEnv
  {
    extern const std::string name;
    bool registerAll();
  }

  //! Parameter structure
  /*! \ingroup inlinehadron */
  struct InlineZeroNubbParams
  {
    InlineZeroNubbParams();
    InlineZeroNubbParams(XMLReader& xml_in, const std::string& path);
    void write(XMLWriter& xml_out, const std::string& path);

    unsigned long frequency;

    struct Param_t
    {
      // bool MesonP;             // Meson spectroscopy
      // bool CurrentP;           // Meson currents
      // bool BaryonP;            // Baryons spectroscopy

      // bool time_rev;           // Use time reversal in baryon spectroscopy

      int mom_idx;                // k1 = (-n, 0, n, 0), k2 = (0, n, n, 0), q = (n, n, 0, 0)
      bool ferm_bc;               // true if fermionic boundary conditions
      // int mom2_max;            // (mom)^2 <= mom2_max. mom2_max=7 in szin.
      // bool avg_equiv_mom;      // average over equivalent momenta
    } param;

    struct NamedObject_t
    {
      std::string  gauge_id;           /*!< Input gauge field */

      struct Props_t
      {
	       std::string  first_id;
         std::string  second_id;
         std::string  third_id;
      };

      multi1d<Props_t> sink_pairs;
    } named_obj;

    std::string xml_file;  // Alternate XML file pattern
  };


  //! Inline zero nubb measurement
  /*! \ingroup inlinehadron */
  class InlineZeroNubb : public AbsInlineMeasurement
  {
  public:
    ~InlineZeroNubb() {}
    InlineZeroNubb(const InlineZeroNubbParams& p) : params(p) {}
    InlineZeroNubb(const InlineZeroNubb& p) : params(p.params) {}

    unsigned long getFrequency(void) const {return params.frequency;}

    //! Do the measurement
    void operator()(const unsigned long update_no,
		    XMLWriter& xml_out);

  protected:
    //! Do the measurement
    void func(const unsigned long update_no,
	      XMLWriter& xml_out);

  private:
    InlineZeroNubbParams params;
  };

}

#endif
