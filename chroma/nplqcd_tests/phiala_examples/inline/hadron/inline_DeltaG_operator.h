#ifndef __inline_DeltaG_operator_h__
#define __inline_DeltaG_operator_h__

#include "chromabase.h"
#include "meas/inline/abs_inline_measurement.h"
#include "meas/hadron/DeltaG_operator.h"

namespace Chroma 
{
    
    
  /*! \ingroup inlinehadron */
  /*  namespace InlineDeltaGEnv
  {
    extern const std::string name;
    bool registerAll();

  //! Parameter structure
  /*! \ingroup inlinehadron */
  /*  struct Params
  {
    Params();
    Params(XMLReader& xml_in, const std::string& path);
    void write(XMLWriter& xml_out, const std::string& path);

    unsigned long frequency;

    
      // I need the ID of the named object for the prop
      std::string gauge_id;
      std::string prop_id;
      
      std::string xml_file; // Support output to own private XML File
  }; // struct
}; // namespace InlineDeltaGEnv

  */

   
    namespace InlineDeltaGEnv
    {
      extern const std::string name;
      bool registerAll();
    }
    
    //! Parameter structure
    /*! \ingroup inlinehadron */
    struct InlineDeltaGParams
    {
      InlineDeltaGParams();
      InlineDeltaGParams(XMLReader& xml_in, const std::string& path);
      void write(XMLWriter& xml_out, const std::string& path);
        
      unsigned long frequency;
        
        struct Param_t
        {
	  /*int mom2_max;  */          // (mom)^2 <= mom2_max. mom2_max=7 in szin.
            
	  multi1d<int> tsrc;
	  int tube;
	  multi1d<int> nrow;
            
        } param;
        
        struct NamedObject_t
        {
	  std::string gauge_id;
            
        } named_obj;
        
      std::string xml_file;  // Alternate XML file pattern
    };
  
    
    //! Inline measurement of Wilson loops
	/*! \ingroup inlinehadron */
    class InlineDeltaG : public Chroma::AbsInlineMeasurement
    {
    public:
      // Constructor -- default -- do nothing
					~InlineDeltaG() {}
    
					// Constructor -- from param struct -- copy param struct inside me
					     InlineDeltaG(const InlineDeltaGParams& p) : params(p) {}
    
					// Constructor -- copy constructor -- copy param struct from argument
					     InlineDeltaG(const InlineDeltaG& p) : params(p.params) {}

					unsigned long getFrequency(void) const {return params.frequency;}

					//! Do the measurement
					    void operator()(const unsigned long update_no,
							    XMLWriter& xml_out); 

    protected:
					//! Do the measurement
					  void func(const unsigned long update_no,
						    XMLWriter& xml_out); 

    private:
					InlineDeltaGParams params;
    };
    

};

#endif
