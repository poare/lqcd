#include "meas/inline/abs_inline_measurement_factory.h"
#include "util/ft/sftmom.h"
#include "util/info/proginfo.h"
#include "chroma.h"

#include "meas/inline/hadron/inline_DeltaG_operator.h"
#include "meas/hadron/DeltaG_operator.h"

#include "meas/inline/io/named_objmap.h"

#include "meas/glue/gluecor.h"

namespace Chroma 
{

    namespace InlineDeltaGEnv
    {
        namespace
        {
	  AbsInlineMeasurement* createMeasurement(XMLReader& xml_in,
						  const std::string& path)
	  {
            return new InlineDeltaG(InlineDeltaGParams(xml_in, path));
	  }
        
	  bool registered = false;
        }
        
        
      const std::string name = "DELTA_G";
        
    
      bool registerAll()
      {
	bool success = true;
	if (! registered) {
	  success &= TheInlineMeasurementFactory::Instance().registerObject(name, createMeasurement);
	  registered = true;
	}
	return success;
      }
        
        
      /*
        // Param stuff
        Params::Params() { frequency = 0; gauge_id=""; xml_file="";}
        
        Params::Params(XMLReader& xml_in, const std::string& path)
        {
            try
            {
                XMLReader paramtop(xml_in, path);
                
                if (paramtop.count("Frequency") == 1)
                    read(paramtop, "Frequency", frequency);
                else
                    frequency = 1;
                
                // Get either the gauge_id in the XML or the ID of the default
                // field if no explicit ID exists.
                read(paramtop, "NamedObject/gauge_id", gauge_id);
                
                if( paramtop.count("xml_file") != 0 ) {
                    read(paramtop, "xml_file", xml_file);
                }
                else {
                    xml_file == "";
                }
                
            }
            catch(const std::string& e)
            {
	    QDPIO::cerr << "Caught Exception reading XML: " << e << std::endl;
                QDP_abort(1);
            }
            
        };
        
        
        void Params::write(XMLWriter& xml_out, const std::string& path)
        {
            push(xml_out, path);
            
            QDP::write(xml_out, "gauge_id", gauge_id);
            
            if( xml_file != "" ){ 
                QDP::write(xml_out, "xml_file", xml_file);
            }
            
            pop(xml_out);
        }
        
      */
    };
    
    
    
    
  //! Reader for parameters
  void read(XMLReader& xml, const std::string& path, InlineDeltaGParams::Param_t& param)
  {
    XMLReader paramtop(xml, path);
        
    QDPIO::cerr << "Reading input" << std::endl;
     
    /*read(paramtop, "mom2_max", param.mom2_max);*/
    /*read(paramtop, "avg_equiv_mom", param.avg_equiv_mom);*/
        
    read(paramtop, "tube", param.tube);
        
    switch (param.tube)
      {
      case 0:
	param.tsrc.resize(4);
	for (int ii=0; ii < 4; ii++)
	  {
	    param.tsrc[ii] = 0;
	  }
	break;
      case 1:
	read(paramtop, "tsrc", param.tsrc);
	break;
                
      default:
	QDPIO::cerr << "Tube switch value " << param.tube << " unsupported." << std::endl;
	QDP_abort(1);
      }
        
  }
    
    
  //! Writer for parameters
  void write(XMLWriter& xml, const std::string& path, const InlineDeltaGParams::Param_t& param)
  {
    push(xml, path);
        
    write(xml, "mom2", 0);
    /*write(xml, "avg_equiv_mom", param.avg_equiv_mom);*/
        
    write(xml, "tsrc", param.tsrc);
    write(xml, "tube", param.tube);
        
    write(xml, "nrow", Layout::lattSize());
        
    pop(xml);
  }
    
    
  //! Propagator input
  void read(XMLReader& xml, const std::string& path, InlineDeltaGParams::NamedObject_t& input)
  {
    XMLReader inputtop(xml, path);
        
    read(inputtop, "gauge_id", input.gauge_id);
  }
    
  //! Propagator output
  void write(XMLWriter& xml, const std::string& path, const InlineDeltaGParams::NamedObject_t& input)
  {
    push(xml, path);
        
    write(xml, "gauge_id", input.gauge_id);
        
    pop(xml);
  }
    
    
  // Param stuff
  InlineDeltaGParams::InlineDeltaGParams() { frequency = 0; }
    
  InlineDeltaGParams::InlineDeltaGParams(XMLReader& xml_in, const std::string& path)
  {
        try
	  {
            XMLReader paramtop(xml_in, path);
            
            if (paramtop.count("Frequency") == 1)
	      read(paramtop, "Frequency", frequency);
            else
	      frequency = 1;
            
            // Parameters for source construction
            read(paramtop, "Param", param);
            
            // Read in the output propagator/source configuration info
            read(paramtop, "NamedObject", named_obj);
            
            // Possible alternate XML file pattern
            if (paramtop.count("xml_file") != 0) 
	      {
                read(paramtop, "xml_file", xml_file);
	      }
	  }
        catch(const std::string& e) 
	  {
	    QDPIO::cerr << __func__ << ": Caught Exception reading XML: " << e << std::endl;
            QDP_abort(1);
	  }
  }
    
    
    void
    InlineDeltaGParams::write(XMLWriter& xml_out, const std::string& path)
    {
      push(xml_out, path);
        
      Chroma::write(xml_out, "Param", param);
      Chroma::write(xml_out, "NamedObject", named_obj);
      QDP::write(xml_out, "xml_file", xml_file);
        
      pop(xml_out);
    }

    
    
    

  // Function call
  void InlineDeltaG::operator()(long unsigned int update_no,
				XMLWriter& xml_out)
  {
        
    // If xml file not empty, then use alternate
    if (params.xml_file != "")
      {
	std::string xml_file = makeXMLFileName(params.xml_file, update_no);
            
	push(xml_out, "DeltaG_operator");
	write(xml_out, "update_no", update_no);
	write(xml_out, "xml_file", xml_file);
	pop(xml_out);
            
	XMLFileWriter xml(xml_file);
	func(update_no, xml);
      }
    else
      {
	func(update_no, xml_out);
      }
  }
    
    

    
    
    
  void InlineDeltaG::func(unsigned long int update_no, XMLWriter& xml_out)
  {
    START_CODE();
        
    StopWatch measure_time;
    measure_time.reset();
    measure_time.start();
        
        
    // Test that the gauge configuration exists in the map.
    XMLBufferWriter gauge_xml;
        try
	  {
            // Try and get at the gauge field if it doesn't exist
            // an exception will be thrown.
	    TheNamedObjMap::Instance().getData< multi1d<LatticeColorMatrix> >(params.named_obj.gauge_id);
	    TheNamedObjMap::Instance().get(params.named_obj.gauge_id).getRecordXML(gauge_xml);
            
	  }
        catch( std::bad_cast )
	  {
            
            // If the object exists and is of the wrong type we will
            // land in this catch.
	    QDPIO::cerr << InlineDeltaGEnv::name << ": caught dynamic cast error"
			<< std::endl;
            QDP_abort(1);
	  }
        catch (const std::string& e)
	  {
            // If the object is not in the map we will land in this
            // catch
	    QDPIO::cerr << InlineDeltaGEnv::name << ": map call failed: " << e
			<< std::endl;
            QDP_abort(1);
	  }
        
        // If we got here, then the gauge field is in
        // the map. Its XML will have been captured.
        // Let us bind the references to a local name so
        // we don't have to type the long lookup string every time.
        //
        // Note const here means we can't change the field
        const multi1d<LatticeColorMatrix>& u =
	  TheNamedObjMap::Instance().getData< multi1d<LatticeColorMatrix> >(params.named_obj.gauge_id);
        
        
	QDPIO::cout << InlineDeltaGEnv::name <<": Beginning" << std::endl;
        
        // Boilerplate stuff to the output XML
        push(xml_out, "deltaG_measurement");
        write(xml_out, "update_no", update_no);
        
        // Write info about the program
        proginfo(xml_out);
        
        // Write out the input
        params.write(xml_out, "Input");
        
        // Write out the config info
        write(xml_out, "Config_info", gauge_xml);
        
        
        // Initialize the slow Fourier transform phases
        /*SftMom phases(params.param.mom2_max, params.param.avg_equiv_mom, Nd-1);*/
        
        // Keep a copy of the phases with NO momenta
        SftMom phases_nomom(0, false, Nd-1);
        
        // Keep a copy of the phases with momenta
	// setting j_decay to -1 gives the 4D FT
        SftMom phases4D(36, false, -1);
        
        SftMom phases3D(36, false, Nd-1);
        
        SftMom phases_ninemom(18, false, Nd-1);
        
        multi1d<int> offset;
        offset.resize(4);
        offset[0] = 1;
        offset[1] = 2;
        offset[2] = 3;
        offset[3] = 2;
        
        //SftMom phases_ninemom_testoriginoffset(9, offset, false, Nd-1);
        

        //Measure delta G, for now just n=2
       
	measureDeltaGn2Test(u, phases_ninemom, xml_out, "DeltaGn2", params.param.tube, params.param.tsrc);
        
        //measureFFtilUncontracted(u, phases_ninemom, xml_out, "FFtilUncon", params.param.tube, params.param.tsrc);
        
	measureF2(u, phases_ninemom, xml_out, "F2");
        
        measureFFtil(u, phases_ninemom, xml_out, "FFtil");
        
	//measureF2threeindexNEW(u, phases_ninemom, xml_out, "F2threeindex");

	//measureF2fourindex(u, phases_ninemom, xml_out, "F2fourindex");

	/*        
		  measuregluon(u, phases4D, xml_out, "2pts4D");
		  measuregluon3(u, phases3D, xml_out, "2pts3D");

		  measuregluonBALL(u, phases4D, xml_out, 5, params.param.tsrc, "2pts4DBALL5");
		  measuregluonBALL(u, phases4D, xml_out, 10, params.param.tsrc, "2pts4DBALL10");
		  measuregluonBALL(u, phases4D, xml_out, 15, params.param.tsrc, "2pts4DBALL15");

	*/

        // End of  your code.
        pop(xml_out);
        
        
        measure_time.stop();
	QDPIO::cout << InlineDeltaGEnv::name << ": total time = "
		    << measure_time.getTimeInSeconds() 
		    << " secs" << std::endl;
        
	QDPIO::cout << InlineDeltaGEnv::name << ": ran successfully" << std::endl;
        END_CODE();
  };

    
    
    
    

}  // end namespace Chroma
