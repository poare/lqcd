#include "chromabase.h"
#include "util/ft/sftmom.h"
#include "util/info/proginfo.h"
#include "chroma.h"

#include "meas/glue/mesfield.h"

#include "meas/hadron/DeltaG_operator.h"


namespace Chroma
{

    
  void measureDeltaGn2Test( const multi1d<LatticeColorMatrix>& u,
			    const SftMom& phases,
			    XMLWriter& xml,
			    const std::string& xml_out, int tube, multi1d<int> tsrc)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
        
    /*Get lattice size*/
    multi1d<int> nrow = Layout::lattSize();
    multi1d<int> coord = nrow;

    Complex myzero=cmplx(Real(0.),Real(0.));

    /*LatticeComplex DeltaGn2 = constructDeltaGn2Test(u,1);
        
        //Do fourier transform...
        
        multi2d<DComplex> DeltaGn2OperatorFT;
        
        DeltaGn2OperatorFT = phases.sft(DeltaGn2);
        
        push(xml, xml_out);
        write(xml, xml_out, DeltaGn2OperatorFT[0]);
        pop(xml);*/
        
        
    multi1d<LatticeComplex> DeltaGn2;
    multi2d<DComplex> DeltaGn2OperatorFT;
        
    DeltaGn2 = constructDeltaGn2Test(u);
        
    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    push(xml_sink_mom, "momenta");
        
    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      {
            
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
        
	for (int this_ii=0; this_ii < 14; ++this_ii )
	  {

	    switch ( tube )
	      {
	      case 0:
		break;
	      case 1:

		for (int ix=0; ix < nrow(0); ix++)
		  {
		    coord[0] = ix;
		    for(int iy=0; iy < nrow(1); iy++)
		      {
			coord[1] = iy;
			for(int iz=0; iz < nrow(2); iz++)
			  {
			    coord[2] = iz;
			    for(int it = 0; it < nrow(3); it++)
			      {
				coord[3] = it;

				int xdist = std::min(abs(ix - tsrc[0]),nrow[0]-abs(ix - tsrc[0]));

				int ydist = std::min(abs(iy - tsrc[1]),nrow[1]-abs(iy - tsrc[1]));

				int zdist = std::min(abs(iz - tsrc[2]),nrow[2]-abs(iz - tsrc[2]));

				if (xdist > 2) {
				  pokeSite(DeltaGn2[this_ii],myzero,coord);
				};

				if (ydist > 2) {
				  pokeSite(DeltaGn2[this_ii],myzero,coord);
				};

				if (zdist > 2) {
				  pokeSite(DeltaGn2[this_ii],myzero,coord);
				};
			      };
			  };
		      };
		  };


		break;
	      default:
		QDPIO::cerr << "Tube switch value " << tube << " unsupported." << std::endl;
		QDP_abort(1);
	      }


	    DeltaGn2OperatorFT = phases.sft(DeltaGn2[this_ii]);
            
	    write(xml_sink_mom, xml_out, DeltaGn2OperatorFT[mom_num]);

	  }
        
	pop(xml_sink_mom);
      }

        
        
        
        
  }; // end measureDeltaGn2
    
  void measureFFtilUncontracted( const multi1d<LatticeColorMatrix>& u,
				 const SftMom& phases,
				 XMLWriter& xml,
				 const std::string& xml_out, int tube, multi1d<int> tsrc)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
         */
        
        
    /*Get lattice size*/
    multi1d<int> nrow = Layout::lattSize();
    multi1d<int> coord = nrow;
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
    /*LatticeComplex DeltaGn2 = constructDeltaGn2Test(u,1);
         
         //Do fourier transform...
         
         multi2d<DComplex> DeltaGn2OperatorFT;
         
         DeltaGn2OperatorFT = phases.sft(DeltaGn2);
         
         push(xml, xml_out);
         write(xml, xml_out, DeltaGn2OperatorFT[0]);
         pop(xml);*/
        
        
    multi1d<LatticeComplex> DeltaGn2;
    multi2d<DComplex> DeltaGn2OperatorFT;
        
    DeltaGn2 = constructFFtilUncontracted(u);
        
    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    push(xml_sink_mom, "momenta");
        
    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      {
            
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
            
	for (int this_ii=0; this_ii < 14; ++this_ii )
	  {
                
	    switch ( tube )
	      {
	      case 0:
		break;
	      case 1:
                        
		for (int ix=0; ix < nrow(0); ix++)
		  {
		    coord[0] = ix;
		    for(int iy=0; iy < nrow(1); iy++)
		      {
			coord[1] = iy;
			for(int iz=0; iz < nrow(2); iz++)
			  {
			    coord[2] = iz;
			    for(int it = 0; it < nrow(3); it++)
			      {
				coord[3] = it;
                                        
				int xdist = std::min(abs(ix - tsrc[0]),nrow[0]-abs(ix - tsrc[0]));
                                        
				int ydist = std::min(abs(iy - tsrc[1]),nrow[1]-abs(iy - tsrc[1]));
                                        
				int zdist = std::min(abs(iz - tsrc[2]),nrow[2]-abs(iz - tsrc[2]));
                                        
				if (xdist > 2) {
				  pokeSite(DeltaGn2[this_ii],myzero,coord);
				};
                                        
				if (ydist > 2) {
				  pokeSite(DeltaGn2[this_ii],myzero,coord);
				};
                                       
				if (zdist > 2) {
				  pokeSite(DeltaGn2[this_ii],myzero,coord);
				};
			      };
			  };
		      };
		  };
                        
                        
		break;
	      default:
		QDPIO::cerr << "Tube switch value " << tube << " unsupported." << std::endl;
		QDP_abort(1);
	      }
                
                
	    DeltaGn2OperatorFT = phases.sft(DeltaGn2[this_ii]);
                
	    write(xml_sink_mom, xml_out, DeltaGn2OperatorFT[mom_num]);
                
	  }
            
	pop(xml_sink_mom);
      }
        
        
        
        
        
  }; // end measureDeltaGn2
    
    
  void measureDeltaGn3( const multi1d<LatticeColorMatrix>& u,
			const SftMom& phases,
			XMLWriter& xml,
			const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi1d<LatticeComplex> DeltaGn3;
    multi2d<DComplex> DeltaGn3OperatorFT;
        
    DeltaGn3 = constructDeltaGn3(u);
        
        
    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    push(xml_sink_mom, "momenta");
        
    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      {
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
            
	for (int this_ii=0; this_ii < 4; ++this_ii )
	  {
                
	    DeltaGn3OperatorFT = phases.sft(DeltaGn3[this_ii]);
                
	    write(xml_sink_mom, xml_out, DeltaGn3OperatorFT[mom_num]);
	  }
            
	pop(xml_sink_mom);
      }

        
        
        
  };
    
  /*REMEMBER TO GAUGE FIX BEFORE DOING THIS*/
  void measuregluon( const multi1d<LatticeColorMatrix>& u,
		     const SftMom& phases,
		     XMLWriter& xml,
		     const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi2d<LatticeComplex> LRgluon;
    multi2d<multi2d<DComplex>> gluonFT;
    multi1d<DComplex> result;
        
    multi1d<int> nrow = Layout::lattSize();
        
    Real p_mu_a_ontwo;
    Real p_nu_a_ontwo;
    Complex extraPhase_mu;
    Complex extraPhase_nu;
        
    int Nadj = Nc*Nc - 1;
    const Real pi = 3.141592653589793238462643383279502;
        
    gluonFT.resize(Nd, Nadj);
        
    LRgluon = ConstructGluon(u);
        
        
    push(xml, xml_out);
        
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
                
	    gluonFT[mu][c] = phases.sft(LRgluon[mu][c]);
	  }
      }
        
        
    for(int mu=0; mu < Nd; ++mu)
      {
        
	for(int nu = 0; nu < Nd; ++nu)
	  {
                
	    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
	    push(xml_sink_mom, "momenta");
                    
	    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
	      {
                        
		p_mu_a_ontwo = pi * Real(phases.numToMom(mom_num)[mu]) / Layout::lattSize()[mu];
		p_nu_a_ontwo = pi * Real(phases.numToMom(phases.numMom()-1-mom_num)[nu]) / Layout::lattSize()[nu];
                        
		extraPhase_mu = cmplx(cos(p_mu_a_ontwo), sin(p_mu_a_ontwo));
		extraPhase_nu = cmplx(cos(p_nu_a_ontwo), sin(p_nu_a_ontwo));
                        
		result = gluonFT[mu][0][mom_num]*gluonFT[nu][0][phases.numMom()-1-mom_num] * extraPhase_mu * extraPhase_nu;
                        
		for(int c = 1; c < Nadj; ++c)
		  {

		    result += gluonFT[mu][c][mom_num]*gluonFT[nu][c][phases.numMom()-1-mom_num] * extraPhase_mu * extraPhase_nu;
		  }
                    
		push(xml_sink_mom);
		write(xml_sink_mom, "sink_mom_num", mom_num);
		write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
		write(xml_sink_mom, "my2pt", result);
		pop(xml_sink_mom);
	      }
                
                
	    pop(xml_sink_mom);
                
	  }
            
      }
        
    pop(xml);
        
  };


  /*REMEMBER TO GAUGE FIX BEFORE DOING THIS*/
  void measuregluonBALL( const multi1d<LatticeColorMatrix>& u,
                         const SftMom& phases,
                         XMLWriter& xml,
			 int ball_size, multi1d<int> tsrc,
                         const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi2d<LatticeComplex> LRgluon;
    multi2d<multi2d<DComplex>> gluonFT;
    multi1d<DComplex> result;
        
    multi1d<int> nrow = Layout::lattSize();
    multi1d<int> coord = nrow;

    Complex myzero=cmplx(Real(0.),Real(0.));

    Real p_mu_a_ontwo;
    Real p_nu_a_ontwo;
    Complex extraPhase_mu;
    Complex extraPhase_nu;
        
    int Nadj = Nc*Nc - 1;
    const Real pi = 3.141592653589793238462643383279502;
        
    gluonFT.resize(Nd, Nadj);
        
    LRgluon = ConstructGluon(u);
        
        
    push(xml, xml_out);
        
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
                      
	    for(int ix=0; ix < ball_size; ix++)
	      {
		coord[0] = ix;
		for(int iy=0; iy < ball_size; iy++)
		  {
		    coord[1] = iy;
		    for(int iz=0; iz < ball_size; iz++)
		      {
			coord[2] = iz;
			for(int it=0; it < ball_size; it++)
			  {
			    coord[3] = it;

			    int xdist = std::min(abs(ix - tsrc[0]),nrow[0]-abs(ix - tsrc[0]));                                                                   
			    int ydist = std::min(abs(iy - tsrc[1]),nrow[1]-abs(iy - tsrc[1]));                                                         
			    int zdist = std::min(abs(iz - tsrc[2]),nrow[2]-abs(iz - tsrc[2]));  
			    int tdist = std::min(abs(it - tsrc[2]),nrow[3]-abs(it - tsrc[3]));  
			    
			    double dist = std::sqrt(xdist*xdist+ydist*ydist+zdist*zdist+tdist*tdist);

			    if (dist > ball_size) {                                                                                                                                      
			      pokeSite(LRgluon[mu][c],myzero,coord);                                                                                                           
			    };  

			  }
		      }
		  }
	      }
	  }
      }


    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
                
	    gluonFT[mu][c] = phases.sft(LRgluon[mu][c]);
	  }
      } 
  
    for(int mu=0; mu < Nd; ++mu)
      {
        
	for(int nu = 0; nu < Nd; ++nu)
	  {
                
	    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
	    push(xml_sink_mom, "momenta");
                    
	    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
	      {
                        
		p_mu_a_ontwo = pi * Real(phases.numToMom(mom_num)[mu]) / Layout::lattSize()[mu];
		p_nu_a_ontwo = pi * Real(phases.numToMom(phases.numMom()-1-mom_num)[nu]) / Layout::lattSize()[nu];
                        
		extraPhase_mu = cmplx(cos(p_mu_a_ontwo), sin(p_mu_a_ontwo));
		extraPhase_nu = cmplx(cos(p_nu_a_ontwo), sin(p_nu_a_ontwo));
                        
		result = gluonFT[mu][0][mom_num]*gluonFT[nu][0][phases.numMom()-1-mom_num] * extraPhase_mu * extraPhase_nu;
                        
		for(int c = 1; c < Nadj; ++c)
		  {

		    result += gluonFT[mu][c][mom_num]*gluonFT[nu][c][phases.numMom()-1-mom_num] * extraPhase_mu * extraPhase_nu;
		  }
                    
		push(xml_sink_mom);
		write(xml_sink_mom, "sink_mom_num", mom_num);
		write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
		write(xml_sink_mom, "my2pt", result);
		pop(xml_sink_mom);
	      }
                
                
	    pop(xml_sink_mom);
                
	  }
            
      }
        
    pop(xml);
        
  };

    
    
  /*REMEMBER TO GAUGE FIX BEFORE DOING THIS*/
  void measuregluon3( const multi1d<LatticeColorMatrix>& u,
                      const SftMom& phases,
                      XMLWriter& xml,
                      const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi2d<LatticeComplex> LRgluon;
    multi2d<multi2d<DComplex>> gluonFT;
    multi1d<DComplex> result;
        
    multi1d<int> nrow = Layout::lattSize();
        
    Real p_mu_a_ontwo;
    Real p_nu_a_ontwo;
    Complex extraPhase_mu;
    Complex extraPhase_nu;
        
    int Nadj = Nc*Nc - 1;
    const Real pi = 3.141592653589793238462643383279502;
        
    gluonFT.resize(Nd, Nadj);
        
    LRgluon = ConstructGluon(u);
        
        
    push(xml, xml_out);
        
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
                
	    gluonFT[mu][c] = phases.sft(LRgluon[mu][c]);
	  }
      }
        
        
    for(int mu=0; mu < Nd; ++mu)
      {
            
	for(int nu = 0; nu < Nd; ++nu)
	  {
                
	    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
	    push(xml_sink_mom, "momenta");
                
	    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
	      {
                    
		result = gluonFT[mu][0][mom_num][0]*gluonFT[nu][0][phases.numMom()-1-mom_num];
                    
		for(int c = 1; c < Nadj; ++c)
		  {
                        
		    result += gluonFT[mu][c][mom_num][0]*gluonFT[nu][c][phases.numMom()-1-mom_num];
		  }
                    
		push(xml_sink_mom);
		write(xml_sink_mom, "sink_mom_num", mom_num);
		write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
		write(xml_sink_mom, "my2pt", result);
		pop(xml_sink_mom);
	      }
                
                
	    pop(xml_sink_mom);
                
	  }
            
      }
        
    pop(xml);
        
  };

    
    
  void measureF2( const multi1d<LatticeColorMatrix>& u,
		  const SftMom& phases,
		  XMLWriter& xml,
		  const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi1d<LatticeComplex> F2Results;
    multi2d<DComplex> F2FT;
        
    F2Results = constructF2(u);
        
    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    //XMLArrayWriter xml_sink_mom(xml,phases.numMom());
    push(xml_sink_mom, "momenta");
        
    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      //for(int mom_num = 0; mom_num < phases.numMom(); ++mom_num)
      {
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
        
	for (int this_ii=0; this_ii < 9; ++this_ii )
	  {
            
	    F2FT = phases.sft(F2Results[this_ii]);
            
	    write(xml_sink_mom, xml_out, F2FT[mom_num]);
	  }
            
	pop(xml_sink_mom);
      }

    
  };

  void measureF2threeindex( const multi1d<LatticeColorMatrix>& u,
			    const SftMom& phases,
			    XMLWriter& xml,
			    const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi1d<LatticeComplex> F2Results;
    multi2d<DComplex> F2FT;
        
    F2Results = constructF2threeindex(u);
        
    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    //XMLArrayWriter xml_sink_mom(xml,phases.numMom());
    push(xml_sink_mom, "momenta");
        
    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      //for(int mom_num = 0; mom_num < phases.numMom(); ++mom_num)
      {
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
        
	for (int this_ii=0; this_ii < 8; ++this_ii )
	  {
            
	    F2FT = phases.sft(F2Results[this_ii]);
            
	    write(xml_sink_mom, xml_out, F2FT[mom_num]);
	  }
            
	pop(xml_sink_mom);
      }

    
  };


  void measureF2threeindexNEW( const multi1d<LatticeColorMatrix>& u,
			       const SftMom& phases,
			       XMLWriter& xml,
			       const std::string& xml_out)
  {
    /*!                                                                                                    
     * \param u  gauge field ( Read )                                                                      
     * \param xml_out output ( Write )                                                                     
     */

    multi1d<LatticeComplex> F2Results;
    multi2d<DComplex> F2FT;

    F2Results = constructF2threeindexNEW(u);

    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    //XMLArrayWriter xml_sink_mom(xml,phases.numMom());                                                    
    push(xml_sink_mom, "momenta");

    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      //for(int mom_num = 0; mom_num < phases.numMom(); ++mom_num)                                           
      {
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));

	for (int this_ii=0; this_ii < 12; ++this_ii )
	  {

	    F2FT = phases.sft(F2Results[this_ii]);

	    write(xml_sink_mom, xml_out, F2FT[mom_num]);
	  }

	pop(xml_sink_mom);
      }


  };
  void measureF2fourindex( const multi1d<LatticeColorMatrix>& u,
			   const SftMom& phases,
			   XMLWriter& xml,
			   const std::string& xml_out)
  {
    /*!                                                                                                    
     * \param u  gauge field ( Read )                                                                      
     * \param xml_out output ( Write )                                                                     
     */

    multi1d<LatticeComplex> F2Results;
    multi2d<DComplex> F2FT;

    F2Results = constructF2fourindex(u);

    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    //XMLArrayWriter xml_sink_mom(xml,phases.numMom());                                                    
    push(xml_sink_mom, "momenta");

    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      //for(int mom_num = 0; mom_num < phases.numMom(); ++mom_num)                                           
      {
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", mom_num);
	write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));

	for (int this_ii=0; this_ii < 16; ++this_ii )
	  {

	    F2FT = phases.sft(F2Results[this_ii]);

	    write(xml_sink_mom, xml_out, F2FT[mom_num]);
	  }

	pop(xml_sink_mom);
      }


  };


  void measureFFtil( const multi1d<LatticeColorMatrix>& u,
		     const SftMom& phases,
		     XMLWriter& xml,
		     const std::string& xml_out)
  {

    /*!                                                                                                                                         
     * \param u  gauge field ( Read )                                                                                                           
     * \param xml_out output ( Write )                                                                                                          
     */

    multi1d<LatticeComplex> F2Results;
    multi2d<DComplex> F2FT;

    F2Results = constructFFtil(u);

    XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
    //XMLArrayWriter xml_sink_mom(xml,phases.numMom());                                                                                         
    push(xml_sink_mom, "momenta");

    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
      //for(int mom_num = 0; mom_num < phases.numMom(); ++mom_num)                                                                              
      {
        push(xml_sink_mom);
        write(xml_sink_mom, "sink_mom_num", mom_num);
        write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));

        for (int this_ii=0; this_ii < 9; ++this_ii )
          {

            F2FT = phases.sft(F2Results[this_ii]);

            write(xml_sink_mom, xml_out, F2FT[mom_num]);
          }

        pop(xml_sink_mom);
      } 
  };

    
    
  void measureGlueball( const multi1d<LatticeColorMatrix>& u,
			const SftMom& phases,
			XMLWriter& xml,
			const std::string& xml_out)
  {
    /*!
     * \param u  gauge field ( Read )
     * \param xml_out output ( Write )
     */
        
    multi1d<LatticeComplex> GlueBallResults;
    multi2d<DComplex> GlueballFT;
        
    GlueBallResults = constructGlueballs(u);
        
    for (int this_ii=0; this_ii < 17; ++this_ii )
      {
            
	GlueballFT = phases.sft(GlueBallResults[this_ii]);
            
	push(xml, xml_out);
	write(xml, xml_out, GlueballFT[0]);
	pop(xml);
      }
        
        
  }; // end measureDeltaGn2

    
    
  multi1d<LatticeComplex> constructGlueballs(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the Glueballs
   */
    
  {
        
    multi2d<LatticeComplex> pl;
    pl.resize(Nd,Nd);
        
    multi2d<LatticeComplex> dpl;
    dpl.resize(Nd,Nd);
        
    multi2d<LatticeComplex> dplbak;
    dplbak.resize(Nd,Nd);
        
    multi3d<LatticeComplex> bpl1;
    multi3d<LatticeComplex> bpl2;
    multi3d<LatticeComplex> bpl3;
    multi3d<LatticeComplex> bpl4;
    bpl1.resize(Nd,Nd,Nd);
    bpl2.resize(Nd,Nd,Nd);
    bpl3.resize(Nd,Nd,Nd);
    bpl4.resize(Nd,Nd,Nd);
        
    multi1d<LatticeComplex> bplList;
    bplList.resize(12);
        
    multi1d<LatticeComplex> GlueballRes;
    GlueballRes.resize(17);

      
    /*Just the plaquette, in any direction*/
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {

	    pl[mu][nu]=trace(u[mu]*shift(u[nu],FORWARD,mu)*adj(shift(u[mu], FORWARD, nu))*adj(u[nu]));

	  };
      };
        
    /*The double plaquettes---short,long indices*/
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {

	    dpl[mu][nu]=trace(u[mu]*shift(u[nu],FORWARD,mu)*shift(shift(u[nu],FORWARD,mu),FORWARD,nu)*adj(shift(shift(u[mu], FORWARD, nu),FORWARD,nu))*adj(shift(u[nu],FORWARD,nu))*adj(u[nu]));

	  };
      };
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
                
	    dplbak[mu][nu]=trace(u[mu]*adj(shift(shift(u[nu],FORWARD,mu),BACKWARD,nu))*adj(shift(shift(shift(u[nu],FORWARD,mu),BACKWARD,nu),BACKWARD,nu))*adj(shift(shift(u[mu],BACKWARD,nu),BACKWARD,nu))*shift(shift(u[nu],BACKWARD,nu),BACKWARD,nu)*shift(u[nu],BACKWARD,nu));
                
	  };
      };
        
    /*The bent plaquettes---three indices are first three directions i.e., staple in first two, then staple to third*/
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int gam = 0; gam < Nd; ++gam)
	      {

		bpl1[mu][nu][gam]=trace(u[mu]*shift(u[nu],FORWARD,mu)*adj(shift(u[mu], FORWARD, nu))*adj(shift(shift(u[gam],FORWARD,nu),BACKWARD,gam))*adj(shift(u[nu],BACKWARD,gam))*shift(u[gam],BACKWARD,gam));

	      };
	  };
      };
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int gam = 0; gam < Nd; ++gam)
	      {
                    
		bpl2[mu][nu][gam]=trace(adj(shift(u[mu],BACKWARD,mu))*shift(u[nu],BACKWARD,mu)*shift(shift(u[mu],BACKWARD,mu),FORWARD,nu)*adj(shift(shift(u[gam],FORWARD,nu),BACKWARD,gam))*adj(shift(u[nu],BACKWARD,gam))*shift(u[gam],BACKWARD,gam));
                    
	      };
	  };
      };
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int gam = 0; gam < Nd; ++gam)
	      {
                    
		bpl3[mu][nu][gam]=trace(u[mu]*shift(u[nu],FORWARD,mu)*adj(shift(u[mu], FORWARD, nu))*shift(u[gam],FORWARD,nu)*adj(shift(u[nu],FORWARD,gam))*adj(u[gam]));
                    
	      };
	  };
      };
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int gam = 0; gam < Nd; ++gam)
	      {
                    
		bpl4[mu][nu][gam]=trace(adj(shift(u[mu],BACKWARD,mu))*shift(u[nu],BACKWARD,mu)*shift(shift(u[mu],BACKWARD,mu),FORWARD,nu)*shift(u[gam],FORWARD,nu)*adj(shift(u[nu],FORWARD,gam))*adj(u[gam]));
                    
	      };
	  };
      };
        
        
        
    bplList[0] = bpl1[2][1][0];
    bplList[1] = bpl1[1][0][2];
    bplList[2] = bpl1[0][2][1];
        
    bplList[3] = bpl2[0][1][2];
    bplList[4] = bpl2[2][0][1];
    bplList[5] = bpl2[1][2][0];
        
    bplList[6] = bpl3[0][1][2];
    bplList[7] = bpl3[2][0][1];
    bplList[8] = bpl3[1][2][0];
        
    bplList[9] = bpl4[2][1][0];
    bplList[10] = bpl4[1][0][2];
    bplList[11] = bpl4[0][2][1];
        

        
    GlueballRes[0] = pl[2][0]-pl[1][2];
    GlueballRes[1] = -2.*pl[0][1]+pl[2][0]+pl[1][2];
        
    GlueballRes[2] = dpl[2][0]-dpl[1][2]+dplbak[0][2]-dplbak[2][1];
    GlueballRes[3] = -2.*dpl[0][1]+dpl[2][0]+dpl[1][2]-2.*dplbak[1][0]+dplbak[0][2]+dplbak[2][1];
    GlueballRes[4] = -dpl[2][0]+dpl[1][2]+dplbak[0][2]-dplbak[2][1];
    GlueballRes[5] = -2.*dpl[0][1]+dpl[2][0]+dpl[1][2]+2.*dplbak[1][0]-dplbak[0][2]-dplbak[2][1];
        
    GlueballRes[6] = dpl[0][1]-dplbak[1][0];
    GlueballRes[7] = dpl[2][0]-dplbak[0][2];
    GlueballRes[8] = dpl[1][2]-dplbak[2][1];
        
    GlueballRes[9] = bplList[2]-bplList[5]-bplList[8]+bplList[11];
    GlueballRes[10] = bplList[0]-bplList[3]-bplList[6]+bplList[9];
    GlueballRes[11] = bplList[1]-bplList[4]-bplList[7]+bplList[10];
        
    GlueballRes[12] = bplList[0]+bplList[1]-bplList[3]+bplList[4]+bplList[6]-bplList[7]-bplList[9]-bplList[10];
    GlueballRes[13] = bplList[1]+bplList[2]-bplList[4]+bplList[5]+bplList[7]-bplList[8]-bplList[10]-bplList[11];
    GlueballRes[14] = bplList[0]+bplList[2]+bplList[3]-bplList[5]-bplList[6]+bplList[8]-bplList[9]-bplList[11];
        
    GlueballRes[15] = -bplList[0]+bplList[1]-bplList[3]+bplList[4]-bplList[6]+bplList[7]-bplList[9]+bplList[10];
    GlueballRes[16] = -bplList[0]-bplList[1]+2.*bplList[2]-bplList[3]-bplList[4]+2.*bplList[5]-bplList[6]-bplList[7]+2.*bplList[8]-bplList[9]-bplList[10]+2.*bplList[11];
        
        
    return GlueballRes;
        
  };

    
    
    
  multi1d<LatticeComplex> constructDeltaGn2Test(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
     */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeColorMatrix> pl;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    multi4d<LatticeComplex> Opn2; /*Delta G operator*/
    multi4d<LatticeComplex> Otilde; /*Box-sym operator*/
    multi4d<LatticeComplex> OtildeSYM; /*Symmetrize mu1 mu2 by hand...*/
        
    multi1d<LatticeComplex> DeltaGRes;
    DeltaGRes.resize(14);
        
    int Nadj = Nc*Nc - 1;
    pl.resize(Nd,Nd);
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    Opn2.resize(Nd, Nd, Nd, Nd);
    Otilde.resize(Nd, Nd, Nd, Nd);
    OtildeSYM.resize(Nd, Nd, Nd, Nd);
        
    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));

    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
        
    /*construct Opn2=G_{mu,mu1}G_{nu,mu2}*/
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int mu1 = 0; mu1 < Nd; ++mu1)
	      {
		for(int mu2 = 0; mu2 < Nd; ++mu2)
		  {
		    Opn2[mu][nu][mu1][mu2] = myzero;
                        
		    for(int c = 0; c < Nadj; ++c)
		      {
			Opn2[mu][nu][mu1][mu2] += F[mu][mu1][c]*F[nu][mu2][c];
		      }
		  }
	      }
	  }
      }
        
        
    /* construct box-symmetrized operator */
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int mu1 = 0; mu1 < Nd; ++mu1)
	      {
		for(int mu2 = 0; mu2 < Nd; ++mu2)
		  {
                        
		    Otilde[mu][nu][mu1][mu2] = Opn2[mu][nu][mu1][mu2];
		    Otilde[mu][nu][mu1][mu2] -= Opn2[mu1][nu][mu][mu2];
                        
		    Otilde[mu][nu][mu1][mu2] -= Opn2[mu][mu2][mu1][nu];
		    Otilde[mu][nu][mu1][mu2] += Opn2[mu1][mu2][mu][nu];
                        
		    Otilde[mu][nu][mu1][mu2] += Opn2[nu][mu][mu2][mu1];
		    Otilde[mu][nu][mu1][mu2] -= Opn2[mu2][mu][nu][mu1];
                        
		    Otilde[mu][nu][mu1][mu2] -= Opn2[nu][mu1][mu2][mu];
		    Otilde[mu][nu][mu1][mu2] += Opn2[mu2][mu1][nu][mu];
                        
                        
		  }
	      }
	  }
      }

    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int mu1 = 0; mu1 < Nd; ++mu1)
	      {
		for(int mu2 = 0; mu2 < Nd; ++mu2)
		  {
		    OtildeSYM[mu][nu][mu1][mu2] = 0.5*(Otilde[mu][nu][mu1][mu2]+Otilde[mu][nu][mu2][mu1]);
		  }
	      }
	  }
      }
        
    /* !!!NEED A factor of 8 ??? only for the case we have here where mu=nu and mu1=mu2, otherwise a different factor */
        
        
    /* combination corresponding to tau_1^(2) basis vector of interest */
        
    DeltaGRes[0] = -2.*Otilde[0][0][1][1];
    DeltaGRes[0] += Otilde[0][0][2][2];
    DeltaGRes[0] += Otilde[0][0][3][3];
    DeltaGRes[0] += Otilde[1][1][2][2];
    DeltaGRes[0] += Otilde[1][1][3][3];
    DeltaGRes[0] -= 2.*Otilde[2][2][3][3];
    DeltaGRes[0] = DeltaGRes[0]/(8.*sqrt3);
        
    /*next vector, same tau_1^(2) basis*/
    DeltaGRes[1] = Otilde[0][0][3][3];
    DeltaGRes[1] += Otilde[1][1][2][2];
    DeltaGRes[1] -= Otilde[0][0][2][2];
    DeltaGRes[1] -= Otilde[1][1][3][3];
    DeltaGRes[1] = DeltaGRes[1]/(8.);
        

    /*this is same as DeltaGRes[9] but bonus symmetric*/
    DeltaGRes[2] = OtildeSYM[0][0][1][3];
    DeltaGRes[2] += OtildeSYM[1][2][2][3];
    DeltaGRes[2] *=1./4.;
        
    /*this is same as DeltaGRes[11] but bonus symmetric*/
    DeltaGRes[3] = OtildeSYM[0][1][1][3];
    DeltaGRes[3] -= OtildeSYM[0][2][2][3];
    DeltaGRes[3] *=1./4.;

    /*same as DeltaGRes[12] but bonus symmetric*/
    DeltaGRes[6] = OtildeSYM[0][0][2][3];
    DeltaGRes[6] -= OtildeSYM[1][1][2][3];
    DeltaGRes[6] *=1./4.;

        
    /* now for tau_2^(2) basis */
    DeltaGRes[4] = Otilde[0][2][1][3];
    DeltaGRes[4] += Otilde[0][1][2][3];
    DeltaGRes[4] = DeltaGRes[4]/(4.);
        
    /* next vector, tau_2^(2) basis */
    DeltaGRes[5] = Otilde[0][2][1][3];
    DeltaGRes[5] -= Otilde[0][1][2][3];
    DeltaGRes[5] -= 2.*Otilde[0][1][3][2];
    DeltaGRes[5] = DeltaGRes[5]/(4.*sqrt3);
        
    /*this one should be the same as above when mtm in z-direction*/
    DeltaGRes[7] = Otilde[0][1][2][3];
    DeltaGRes[7] -= 2.*Otilde[0][1][3][2];
    DeltaGRes[7] = DeltaGRes[7]/(4.*sqrt3);
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[8] = Otilde[0][0][1][2];
    DeltaGRes[8] -= Otilde[1][2][3][3];
    DeltaGRes[8] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[9] = Otilde[0][0][1][3];
    DeltaGRes[9] += Otilde[1][2][2][3];
    DeltaGRes[9] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    /*THERE WAS A TYPO IN THIS BASIS VECTOR*/
    DeltaGRes[10] = Otilde[0][1][1][2];
    DeltaGRes[10] += Otilde[0][2][3][3];
    DeltaGRes[10] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[11] = Otilde[0][1][1][3];
    DeltaGRes[11] -= Otilde[0][2][2][3];
    DeltaGRes[11] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[12] = Otilde[0][0][2][3];
    DeltaGRes[12] -= Otilde[1][1][2][3];
    DeltaGRes[12] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[13] = Otilde[0][1][2][2];
    DeltaGRes[13] -= Otilde[0][1][3][3];
    DeltaGRes[13] *=1./4.;
        
        
    return DeltaGRes;
        
  };
    
  multi1d<LatticeComplex> constructFFtilUncontracted(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeColorMatrix> pl;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    multi4d<LatticeComplex> Opn2; /*Delta G operator*/
    multi4d<LatticeComplex> Otilde; /*Box-sym operator*/
        
    multi1d<LatticeComplex> DeltaGRes;
    DeltaGRes.resize(14);
        
    int Nadj = Nc*Nc - 1;
    pl.resize(Nd,Nd);
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    Opn2.resize(Nd, Nd, Nd, Nd);
    Otilde.resize(Nd, Nd, Nd, Nd);
        
    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
        
    /*construct Opn2=G_{mu,mu1}G_{nu,mu2}*/
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int mu1 = 0; mu1 < Nd; ++mu1)
	      {
		for(int mu2 = 0; mu2 < Nd; ++mu2)
		  {
		    Opn2[mu][nu][mu1][mu2] = myzero;
                        
		    for(int alpha = 0; alpha < Nd; ++alpha)
		      {
			for(int beta = 0; beta < Nd; ++beta)
			  {
                        
			    for(int c = 0; c < Nadj; ++c)
			      {
				Opn2[mu][nu][mu1][mu2] += LeviCivita4D(mu,mu1,alpha,beta)*F[alpha][beta][c]*F[nu][mu2][c];
			      }
			  }
		      }
		  }
	      }
	  }
      }
        
        
    /* construct box-symmetrized operator */
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int mu1 = 0; mu1 < Nd; ++mu1)
	      {
		for(int mu2 = 0; mu2 < Nd; ++mu2)
		  {
                        
		    Otilde[mu][nu][mu1][mu2] = Opn2[mu][nu][mu1][mu2];
		    Otilde[mu][nu][mu1][mu2] -= Opn2[mu1][nu][mu][mu2];
                        
		    Otilde[mu][nu][mu1][mu2] -= Opn2[mu][mu2][mu1][nu];
		    Otilde[mu][nu][mu1][mu2] += Opn2[mu1][mu2][mu][nu];
                        
		    Otilde[mu][nu][mu1][mu2] += Opn2[nu][mu][mu2][mu1];
		    Otilde[mu][nu][mu1][mu2] -= Opn2[mu2][mu][nu][mu1];
                        
		    Otilde[mu][nu][mu1][mu2] -= Opn2[nu][mu1][mu2][mu];
		    Otilde[mu][nu][mu1][mu2] += Opn2[mu2][mu1][nu][mu];
                        
                        
		  }
	      }
	  }
      }
        
    /* !!!NEED A factor of 8 ??? only for the case we have here where mu=nu and mu1=mu2, otherwise a different factor */
        
        
    /* combination corresponding to tau_1^(2) basis vector of interest */
        
    DeltaGRes[0] = -2.*Otilde[0][0][1][1];
    DeltaGRes[0] += Otilde[0][0][2][2];
    DeltaGRes[0] += Otilde[0][0][3][3];
    DeltaGRes[0] += Otilde[1][1][2][2];
    DeltaGRes[0] += Otilde[1][1][3][3];
    DeltaGRes[0] -= 2.*Otilde[2][2][3][3];
    DeltaGRes[0] = DeltaGRes[0]/(8.*sqrt3);
        
    /*this one should be the same as the one above for zero mtm case*/
    DeltaGRes[2] = Otilde[0][0][3][3];
    DeltaGRes[2] += Otilde[1][1][3][3];
    DeltaGRes[2] -= 2.*Otilde[2][2][3][3];
    DeltaGRes[2] = DeltaGRes[2]/(8.*sqrt3);
        
    /*next vector, same tau_1^(2) basis*/
    DeltaGRes[1] = Otilde[0][0][3][3];
    DeltaGRes[1] += Otilde[1][1][2][2];
    DeltaGRes[1] -= Otilde[0][0][2][2];
    DeltaGRes[1] -= Otilde[1][1][3][3];
    DeltaGRes[1] = DeltaGRes[1]/(8.);
        
    /*this one should be the same as the one above for zero mtm case*/
    DeltaGRes[3] = -Otilde[0][0][3][3];
    DeltaGRes[3] += Otilde[1][1][3][3];
    DeltaGRes[3] = DeltaGRes[3]/(8.);
        
    /* now for tau_2^(2) basis */
    DeltaGRes[4] = Otilde[0][2][1][3];
    DeltaGRes[4] += Otilde[0][1][2][3];
    DeltaGRes[4] = DeltaGRes[4]/(4.);
        
    /*this one should be the same as above when mtm in z-direction*/
    DeltaGRes[6] = Otilde[0][1][2][3];
    DeltaGRes[6] = DeltaGRes[6]/(4.);
        
    /* next vector, tau_2^(2) basis */
    DeltaGRes[5] = Otilde[0][2][1][3];
    DeltaGRes[5] -= Otilde[0][1][2][3];
    DeltaGRes[5] -= 2.*Otilde[0][1][3][2];
    DeltaGRes[5] = DeltaGRes[5]/(4.*sqrt3);
        
    /*this one should be the same as above when mtm in z-direction*/
    DeltaGRes[7] = Otilde[0][1][2][3];
    DeltaGRes[7] -= 2.*Otilde[0][1][3][2];
    DeltaGRes[7] = DeltaGRes[7]/(4.*sqrt3);
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[8] = Otilde[0][0][1][2];
    DeltaGRes[8] -= Otilde[1][2][3][3];
    DeltaGRes[8] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[9] = Otilde[0][0][1][3];
    DeltaGRes[9] += Otilde[1][2][2][3];
    DeltaGRes[9] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    /*THERE WAS A TYPO IN THIS BASIS VECTOR*/
    DeltaGRes[10] = Otilde[0][1][1][2];
    DeltaGRes[10] += Otilde[0][2][3][3];
    DeltaGRes[10] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[11] = Otilde[0][1][1][3];
    DeltaGRes[11] -= Otilde[0][2][2][3];
    DeltaGRes[11] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[12] = Otilde[0][0][2][3];
    DeltaGRes[12] -= Otilde[1][1][2][3];
    DeltaGRes[12] *=1./4.;
        
    /*take a vector from tau_2^(6) */
    DeltaGRes[13] = Otilde[0][1][2][2];
    DeltaGRes[13] -= Otilde[0][1][3][3];
    DeltaGRes[13] *=1./4.;
        
        
    return DeltaGRes;
        
  };

    
    
  multi1d<LatticeComplex> constructDeltaGn3(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=3 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeColorMatrix> pl;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    multi5d<LatticeComplex> Opn3; /*Delta G operator*/
    multi5d<LatticeComplex> Otilde3; /*Box-sym operator*/
        
    multi1d<LatticeComplex> DeltaGRes;
    DeltaGRes.resize(4);
        
    int Nadj = Nc*Nc - 1;
    pl.resize(Nd,Nd);
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    Opn3.resize(Nd, Nd, Nd, Nd, Nd);
    Otilde3.resize(Nd, Nd, Nd, Nd, Nd);
        
    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
        
    /*construct Opn3=G_{mu,mu1}D_{mu3}G_{nu,mu2}*/
    for(int mu = 0; mu < Nd; mu++)
      {
	for(int nu = 0; nu < Nd; nu++)
	  {
	    for(int mu1 = 0; mu1 < Nd; mu1++)
	      {
		for(int mu2 = 0; mu2 < Nd; mu2++)
		  {
		    for(int mu3 = 0; mu3 < Nd; mu3++)
		      {
			Opn3[mu][nu][mu1][mu2][mu3] = myzero;
                            
			for (int c = 0; c < Nadj; ++c)
			  {
                                
			    Opn3[mu][nu][mu1][mu2][mu3] += F[mu][nu][c]*Real(2.)*traceColor(tSU3[c]*u[mu3]*shift(pl[nu][mu2],FORWARD,mu3));
			    Opn3[mu][nu][mu1][mu2][mu3] -= F[mu][nu][c]*Real(2.)*traceColor(tSU3[c]*adj(shift(u[mu3],BACKWARD,mu3))*shift(pl[nu][mu2],BACKWARD,mu3));
			    Opn3[mu][nu][mu1][mu2][mu3] -= Real(2.)*traceColor(tSU3[c]*shift(pl[mu][nu],FORWARD,mu3)*adj(u[mu3]))*F[nu][mu2][c];
			    Opn3[mu][nu][mu1][mu2][mu3] += Real(2.)*traceColor(tSU3[c]*shift(pl[mu][nu],BACKWARD,mu3)*shift(u[mu3],BACKWARD,mu3))*F[nu][mu2][c];
			  }
		      }
		  }
	      }
	  }
      }

        
        
    /* construct box-symmetrized operator */
    /*        for(int alpha = 0; alpha < Nd; ++alpha)
        {
            for(int beta = 0; beta < Nd; ++beta)
            {
                for(int gamma = 0; gamma < Nd; ++gamma)
                {
                    for(int delta = 0; delta < Nd; ++delta)
                    {
                        for(int epsilon = 0; epsilon < Nd; ++epsilon)
                        {
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] = Opn3[delta][epsilon][alpha][beta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[epsilon][delta][alpha][beta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[delta][epsilon][alpha][gamma][beta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[epsilon][delta][alpha][gamma][beta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[beta][delta][alpha][gamma][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[delta][beta][alpha][gamma][epsilon];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[beta][delta][alpha][epsilon][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[delta][beta][alpha][epsilon][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[delta][epsilon][beta][alpha][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[epsilon][delta][beta][alpha][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[delta][epsilon][beta][gamma][alpha];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[epsilon][delta][beta][gamma][alpha];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[alpha][epsilon][beta][gamma][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[epsilon][alpha][beta][gamma][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[alpha][epsilon][beta][delta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[epsilon][alpha][beta][delta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[delta][epsilon][gamma][alpha][beta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[epsilon][delta][gamma][alpha][beta];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[beta][delta][gamma][alpha][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[delta][beta][gamma][alpha][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[delta][epsilon][gamma][beta][alpha];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[epsilon][delta][gamma][beta][alpha];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[alpha][epsilon][gamma][beta][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[epsilon][alpha][gamma][beta][delta];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[alpha][epsilon][gamma][delta][beta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[epsilon][alpha][gamma][delta][beta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[alpha][beta][gamma][delta][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[beta][alpha][gamma][delta][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[beta][delta][gamma][epsilon][alpha];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[delta][beta][gamma][epsilon][alpha];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[alpha][beta][gamma][epsilon][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[beta][alpha][gamma][epsilon][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[alpha][epsilon][delta][beta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[epsilon][alpha][delta][beta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[alpha][epsilon][delta][gamma][beta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[epsilon][alpha][delta][gamma][beta];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[alpha][beta][delta][gamma][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[beta][alpha][delta][gamma][epsilon];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[alpha][beta][delta][epsilon][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[beta][alpha][delta][epsilon][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[beta][delta][epsilon][alpha][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[delta][beta][epsilon][alpha][gamma];
                            
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[beta][delta][epsilon][gamma][alpha];
                            Otilde3[alpha][beta][gamma][delta][epsilon] -= Opn3[delta][beta][epsilon][gamma][alpha];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[alpha][beta][epsilon][gamma][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[beta][alpha][epsilon][gamma][delta];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[alpha][beta][epsilon][delta][gamma];
                            Otilde3[alpha][beta][gamma][delta][epsilon] += Opn3[beta][alpha][epsilon][delta][gamma];
                            
  }
                        
                        
                        
  }
  }
  }
  }
        */

    Otilde3 = Opn3;
        
    /* combination corresponding to tau_3^(4) basis vector of interest */

    DeltaGRes[0] = 3.*Otilde3[0][0][1][1][2];
    DeltaGRes[0] -= sqrt3*Otilde3[0][0][2][1][1];
    DeltaGRes[0] += sqrt3*Otilde3[0][0][2][3][3];
    DeltaGRes[0] -= 3.*Otilde3[0][0][3][2][3];
    DeltaGRes[0] -= sqrt3*Otilde3[1][1][2][3][3];
    DeltaGRes[0] += 3.*Otilde3[1][1][3][2][3];
    DeltaGRes[0] *= 3.*sqrt(Real(2.));
        
    DeltaGRes[1] = 3.*Otilde3[0][0][1][1][3];
    DeltaGRes[1] -= 3.*Otilde3[0][0][2][2][3];
    DeltaGRes[1] -= sqrt3*Otilde3[0][0][3][1][1];
    DeltaGRes[1] += sqrt3*Otilde3[0][0][3][2][2];
    DeltaGRes[1] += 3.*Otilde3[1][1][2][2][3];
    DeltaGRes[1] -= sqrt3*Otilde3[1][1][3][2][2];
    DeltaGRes[1] *= 3.*sqrt(Real(2.));
        
    DeltaGRes[2] = -2.*sqrt3*Otilde3[0][0][1][2][2];
    DeltaGRes[2] += 2.*sqrt3*Otilde3[0][0][1][3][3];
    DeltaGRes[2] += 6.*Otilde3[0][0][2][1][2];
    DeltaGRes[2] -= 6.*Otilde3[0][0][3][1][3];
    DeltaGRes[2] -= 3.*Otilde3[1][2][2][3][3];
    DeltaGRes[2] -= 3.*Otilde3[1][2][3][2][3];
    DeltaGRes[2] *= 6.*sqrt(Real(2.));
        
    DeltaGRes[3] = Otilde3[0][1][1][2][2];
    DeltaGRes[3] -= Otilde3[0][1][1][3][3];
    DeltaGRes[3] += Otilde3[0][1][2][1][2];
    DeltaGRes[3] -= Otilde3[0][1][3][1][3];
    DeltaGRes[3] += Otilde3[0][2][2][3][3];
    DeltaGRes[3] += Otilde3[0][2][3][2][3];
    DeltaGRes[3] *= 2.*sqrt(Real(2.));
        
        
    return DeltaGRes;
        
  };

    
    
    
    
    
  multi1d<LatticeComplex> constructF2(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*SU(3) generators*/
    multi2d<LatticeColorMatrix> pl;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    multi2d<LatticeComplex> Fop; /*F_{mu_1,alpha} F^{alpha}_{mu_2}*/
        
    multi1d<LatticeComplex> F2Res;
    F2Res.resize(9);
        
    int Nadj = Nc*Nc - 1;
    pl.resize(Nd,Nd);
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    Fop.resize(Nd, Nd);
        
    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
    Real sqrt2 = sqrt(Real(2.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
        
    /*construct F_{mu_1,alpha} F^{alpha}_{mu_2}   Euclidean space 
     This is sum_c F_c F_c  rather than tr[F F ] = 1/2 \sum{F_c F_c} */
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
	for(int mu2 = 0; mu2 < Nd; ++mu2)
	  {
	    Fop[mu1][mu2] = myzero;
                
	    for(int alpha = 0; alpha < Nd; ++alpha)
	      {
		for(int c = 0; c < Nadj; ++c)
		  {
		    Fop[mu1][mu2] += F[mu1][alpha][c]*F[mu2][alpha][c];
		  }
	      }
	  }
      }
        
        
    /* combination corresponding to tau_1^(3) basis vector of interest */
        
    F2Res[0] = Fop[0][0];
    F2Res[0] += Fop[1][1];
    F2Res[0] -= Fop[2][2];
    F2Res[0] -= Fop[3][3];
    F2Res[0] = F2Res[0]/(2.);
        
    /*next vector, same tau_1^(3) basis*/
    F2Res[1] = Fop[2][2];
    F2Res[1] -= Fop[3][3];
    F2Res[1] = F2Res[1]/(sqrt2);
        
    /*next vector, same tau_1^(3) basis*/
    F2Res[2] = Fop[0][0];
    F2Res[2] -= Fop[1][1];
    F2Res[2] = F2Res[2]/(sqrt2);
        
    /* now for tau_3^(6) basis */
    F2Res[3] = Fop[0][1];
    F2Res[3] += Fop[1][0];
    F2Res[3] = F2Res[3]/(sqrt2);
        
    F2Res[4] = Fop[0][2];
    F2Res[4] += Fop[2][0];
    F2Res[4] = F2Res[4]/(sqrt2);
        
    F2Res[5] = Fop[0][3];
    F2Res[5] += Fop[3][0];
    F2Res[5] = F2Res[5]/(sqrt2);
        
    F2Res[6] = Fop[1][2];
    F2Res[6] += Fop[2][1];
    F2Res[6] = F2Res[6]/(sqrt2);
        
    F2Res[7] = Fop[1][3];
    F2Res[7] += Fop[3][1];
    F2Res[7] = F2Res[7]/(sqrt2);
        
    F2Res[8] = Fop[2][3];
    F2Res[8] += Fop[3][2];
    F2Res[8] = F2Res[8]/(sqrt2);
        

    return F2Res;
        
  };
    
    
  multi1d<LatticeComplex> constructFFtil(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeColorMatrix> pl;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    multi2d<LatticeComplex> Fop; /*F_{mu_1,alpha} F^{alpha}_{mu_2}*/
        
    multi1d<LatticeComplex> FFtilRes;
    FFtilRes.resize(10);
        
    int Nadj = Nc*Nc - 1;
    pl.resize(Nd,Nd);
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    Fop.resize(Nd, Nd);
        
    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
    Real sqrt2 = sqrt(Real(2.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
        
    /*construct F_{mu_1,alpha} F^{alpha}_{mu_2}*/
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
	for(int mu2 = 0; mu2 < Nd; ++mu2)
	  {
	    Fop[mu1][mu2] = myzero;
                
	    for(int alpha = 0; alpha < Nd; ++alpha)
	      {
		for(int beta = 0; beta < Nd; ++beta)
		  {
		    for(int gamma = 0; gamma < Nd; ++gamma)
		      {
			for(int c = 0; c < Nadj; ++c)
			  {
			    Fop[mu1][mu2] += LeviCivita4D(alpha,beta,gamma,mu1)*F[alpha][beta][c]*F[mu2][gamma][c];
			  }
		      }
		  }
	      }
	  }
      }
        
        
    /* combination corresponding to tau_1^(3) basis vector of interest */
        
    FFtilRes[0] = Fop[0][0];
    FFtilRes[0] += Fop[1][1];
    FFtilRes[0] -= Fop[2][2];
    FFtilRes[0] -= Fop[3][3];
    FFtilRes[0] = FFtilRes[0]/(2.);
        
    /*next vector, same tau_1^(3) basis*/
    FFtilRes[1] = Fop[2][2];
    FFtilRes[1] -= Fop[3][3];
    FFtilRes[1] = FFtilRes[1]/(sqrt2);
        
    /*next vector, same tau_1^(3) basis*/
    FFtilRes[2] = Fop[0][0];
    FFtilRes[2] -= Fop[1][1];
    FFtilRes[2] = FFtilRes[2]/(sqrt2);
        
    /* now for tau_3^(6) basis */
    FFtilRes[3] = Fop[0][1];
    FFtilRes[3] += Fop[1][0];
    FFtilRes[3] = FFtilRes[3]/(sqrt2);
        
    FFtilRes[4] = Fop[0][2];
    FFtilRes[4] += Fop[2][0];
    FFtilRes[4] = FFtilRes[4]/(sqrt2);
        
    FFtilRes[5] = Fop[0][3];
    FFtilRes[5] += Fop[3][0];
    FFtilRes[5] = FFtilRes[5]/(sqrt2);
        
    FFtilRes[6] = Fop[1][2];
    FFtilRes[6] += Fop[2][1];
    FFtilRes[6] = FFtilRes[6]/(sqrt2);
        
    FFtilRes[7] = Fop[1][3];
    FFtilRes[7] += Fop[3][1];
    FFtilRes[7] = FFtilRes[7]/(sqrt2);
        
    FFtilRes[8] = Fop[2][3];
    FFtilRes[8] += Fop[3][2];
    FFtilRes[8] = FFtilRes[8]/(sqrt2);
        
    FFtilRes[9] = Fop[0][0];
    FFtilRes[9] += Fop[1][1];
    FFtilRes[9] += Fop[2][2];
    FFtilRes[9] += Fop[3][3];
    FFtilRes[9] = FFtilRes[9]/(2.);
        
        
    return FFtilRes;
        
  };


    
  LatticeComplex constructF2Test(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeColorMatrix> pl;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    LatticeComplex F2; /*Delta G operator*/
        
    int Nadj = Nc*Nc - 1;
    pl.resize(Nd,Nd);
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
        
        
    F = CalculateDumbFmunu(u);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
        
    /*construct Opn2=G_{mu,mu1}G_{nu,mu2}*/
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    F2 = myzero;
                        
	    for(int c = 0; c < Nadj; ++c)
	      {
		F2 += F[mu][nu][c]*F[mu][nu][c];
	      }
	  }
      }
        
    return F2;
        
  };
    
    
  LatticeComplex constructF2TestChroma(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
    multi1d<LatticeColorMatrix> f;
    f.resize(Nd*(Nd-1)/2);
        
    mesField(f,u);
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    int Nadj = Nc*Nc - 1;
    tSU3.resize(Nadj);
    F.resize(Nd*(Nd-1)/2, Nadj);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
    constructSU3generators(tSU3);
        
    LatticeComplex F2; /*Delta G operator*/
        
    /* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
    for(int mu = 0; mu < Nd*(Nd-1)/2; ++mu)
      {
	for(int c = 0; c < Nadj; ++c){
            
	  F[mu][c] = Real(2.)*traceColor(tSU3[c]*f[mu]);
                
	};
      }

    F2 =  myzero;
        
    /*construct Opn2=G_{mu,mu1}G_{nu,mu2}*/
    for(int mu = 0; mu < Nd*(Nd-1)/2; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
	    F2 += F[mu][c]*F[mu][c];
	  }
      }
        
    return F2;
        
  };


  multi1d<LatticeComplex> constructF2threeindex(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
    
    multi1d<LatticeColorMatrix> f;
    multi2d<LatticeColorMatrix> ff;
    ff.resize(Nd, Nd);
    f.resize(Nd*(Nd-1)/2);

    mesField(f,u);

    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    Complex myzero=cmplx(Real(0.),Real(0.));

    multi3d<LatticeComplex> Fmunu;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
    multi3d<LatticeComplex> Fop; /*F_{mu_1,alpha} F^{alpha}_{mu_2}*/
    multi3d<LatticeComplex> FopSYM;
    multi1d<LatticeComplex> F2Res;
        
    int Nadj = Nc*Nc - 1;
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);       
    Fop.resize(Nd, Nd, Nd);
    FopSYM.resize(Nd, Nd, Nd);
    Fmunu.resize(Nd,Nd,Nadj);
    F2Res.resize(8);

    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
    Real sqrt2 = sqrt(Real(2.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);

    ff=0;
    int count = 0;
    for(int mu = 0; mu < Nd-1; ++mu)
      {
	for(int nu = mu+1; nu < Nd; ++nu)
	  {
	    ff[mu][nu] = f[count];
	    ff[nu][mu] = -f[count];

	    ++count;
	  }
      }

    LatticeColorMatrix tmp;
    LatticeColorMatrix tmp1;
        
    /*construct F_{mu_1,alpha} D_{mu_3} F_{mu_2}^{alpha}*/
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
	for(int mu2 = 0; mu2 < Nd; ++mu2)
	  {
	    for(int mu3 = 0; mu3 < Nd; ++mu3)
	      {
		Fop[mu1][mu2][mu3] = myzero;
                
		for(int alpha = 0; alpha < Nd; ++alpha)
		  {
		    for(int c = 0; c < Nadj; ++c)
		      {
			tmp = shift(ff[mu2][alpha],FORWARD,mu3);
			Fop[mu1][mu2][mu3] += F[mu1][alpha][c]*Real(2.)*traceColor(tSU3[c]*u[mu3]*tmp);
			tmp = shift(u[mu3],BACKWARD,mu3);
			tmp1 = shift(ff[mu2][alpha],BACKWARD,mu3);
			Fop[mu1][mu2][mu3] -= F[mu1][alpha][c]*Real(2.)*traceColor(tSU3[c]*adj(tmp)*tmp1);
			tmp = shift(ff[mu1][alpha],FORWARD,mu3);
			Fop[mu1][mu2][mu3] -= Real(2.)*traceColor(tSU3[c]*tmp*adj(u[mu3]))*F[mu2][alpha][c];
			tmp = shift(ff[mu1][alpha],BACKWARD,mu3);
			tmp1 = shift(u[mu3],BACKWARD,mu3);
			Fop[mu1][mu2][mu3] += Real(2.)*traceColor(tSU3[c]*tmp*tmp1)*F[mu2][alpha][c];
		      }
		  }
		  
	      }
	  }
      }
        
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
        for(int mu2 = 0; mu2 < Nd; ++mu2)
          {
            for(int mu3 = 0; mu3 < Nd; ++mu3)
              {
		FopSYM[mu1][mu2][mu3] = Real(1./6.)*(Fop[mu1][mu2][mu3]+Fop[mu1][mu3][mu2]+Fop[mu2][mu1][mu3]+Fop[mu2][mu3][mu1]+Fop[mu3][mu1][mu2]+Fop[mu3][mu2][mu1]);
	      }
	  }
      }


    /* combination corresponding to tau_1^(8) basis vector of interest */
    /* factor of (1/2) for fwd-bwd derivative*/
        
    F2Res[0] = 0.5*(sqrt3/sqrt2)*(FopSYM[0][1][1]-FopSYM[0][2][2]);
    F2Res[1] = 0.5*(sqrt3/sqrt2)*(FopSYM[1][0][0]-FopSYM[1][2][2]);
    F2Res[2] = 0.5*(sqrt3/sqrt2)*(FopSYM[2][0][0]-FopSYM[2][1][1]);
    F2Res[3] = 0.5*(sqrt3/sqrt2)*(FopSYM[3][0][0]-FopSYM[3][1][1]);

    F2Res[4] = 0.5*(1./sqrt2)*(FopSYM[0][1][1]+FopSYM[0][2][2]-2.*FopSYM[0][3][3]);
    F2Res[5] = 0.5*(1./sqrt2)*(FopSYM[1][0][0]+FopSYM[1][2][2]-2.*FopSYM[1][3][3]);
    F2Res[6] = 0.5*(1./sqrt2)*(FopSYM[2][0][0]+FopSYM[2][1][1]-2.*FopSYM[2][3][3]);
    F2Res[7] = 0.5*(1./sqrt2)*(FopSYM[3][0][0]+FopSYM[3][1][1]-2.*FopSYM[3][2][2]);

    return F2Res;

  };


  multi1d<LatticeComplex> constructF2threeindexNEW(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
    
    multi1d<LatticeColorMatrix> f;
    multi2d<LatticeColorMatrix> ff;
    ff.resize(Nd, Nd);
    f.resize(Nd*(Nd-1)/2);

    mesField(f,u);

    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    Complex myzero=cmplx(Real(0.),Real(0.));

    multi3d<LatticeComplex> Fmunu;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
    multi3d<LatticeComplex> Fop; /*F_{mu_1,alpha} F^{alpha}_{mu_2}*/
    multi3d<LatticeComplex> FopSYM;
    multi1d<LatticeComplex> F2Res;
       
    multi2d<LatticeComplex> ucol;
    multi1d<LatticeComplex> utr;
 
    int Nadj = Nc*Nc - 1;
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);       
    Fop.resize(Nd, Nd, Nd);
    FopSYM.resize(Nd, Nd, Nd);
    Fmunu.resize(Nd,Nd,Nadj);
    F2Res.resize(12);
    ucol.resize(Nd, Nadj);
    utr.resize(Nd);

    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
    Real sqrt6 = sqrt(Real(6.));
    Real sqrt2 = sqrt(Real(2.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);

    for(int mu = 0; mu < Nd; ++mu)
      {
	/* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
	for(int c = 0; c < Nadj; ++c)
	  {
	    ucol[mu][c] = Real(2.)*traceColor(tSU3[c]*u[mu]);
	  }
	utr[mu] = Real(2.)*traceColor(u[mu]);
      };    

    int count = 0;
    ff=myzero;
    for(int mu = 0; mu < Nd-1; ++mu)
      {
	for(int nu = mu+1; nu < Nd; ++nu)
	  {
	    ff[mu][nu] = f[count];
	    ff[nu][mu] = -f[count];

	    ++count;
	  }
      }

    LatticeColorMatrix tmp;
    LatticeColorMatrix tmp1;

    Complex fact;
    /* CHECK THERE*/
    /*construct F_{mu_1,alpha} D_{mu_3} F_{mu_2}^{alpha}*/
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
	for(int mu2 = 0; mu2 < Nd; ++mu2)
	  {
	    for(int mu3 = 0; mu3 < Nd; ++mu3)
	      {
		Fop[mu1][mu2][mu3] = myzero;
                
		for(int alpha = 0; alpha < Nd; ++alpha)
		  {

		    tmp = shift(ff[mu2][alpha],FORWARD,mu3);
		    Fop[mu1][mu2][mu3] += Real(0.5)*traceColor(ff[mu1][alpha]*u[mu3]*tmp*adj(u[mu3]));
		    
		    tmp = shift(u[mu3],BACKWARD,mu3);
		    tmp1 = shift(ff[mu2][alpha],BACKWARD,mu3);
		    Fop[mu1][mu2][mu3] -= Real(0.5)*traceColor(ff[mu1][alpha]*adj(tmp)*tmp1*tmp);
		     
		    tmp = shift(ff[mu1][alpha],FORWARD,mu3);
		    Fop[mu1][mu2][mu3] -= Real(0.5)*traceColor(tmp*adj(u[mu3])*ff[mu2][alpha]*u[mu3]);

		    tmp = shift(ff[mu1][alpha],BACKWARD,mu3);
		    tmp1 = shift(u[mu3],BACKWARD,mu3);
		    Fop[mu1][mu2][mu3] += Real(0.5)*traceColor(tmp*tmp1*ff[mu2][alpha]*adj(tmp1));
		  }
	      }
	  }
      }

    
    /*
    F2Res[0] = traceColor(u[1]*shift(ff[1][2],FORWARD,1)*adj(u[1]));
    F2Res[1] = traceColor(ff[1][2]*u[1]*shift(ff[1][2],FORWARD,1)*adj(u[1]));
    F2Res[2] = traceColor(u[1]*shift(ff[1][2],BACKWARD,1)*adj(u[1]));
    F2Res[3]= traceColor(u[1]*shift(ff[1][2],BACKWARD,1)*shift(adj(u[1]),BACKWARD,1));
    F2Res[4] = traceColor(ff[1][2]);
    F2Res[5] = traceColor(shift(ff[1][2],FORWARD,1));
    F2Res[6] = traceColor(ff[1][2]*shift(ff[1][2],FORWARD,1));
    */

    /*construct F_{mu_1,alpha} D_{mu_3} F_{mu_2}^{alpha}*/
    /*for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
      for(int mu2 = 0; mu2 < Nd; ++mu2)
        {
	    for(int mu3 = 0; mu3 < Nd; ++mu3)
	          {
		  Fop[mu1][mu2][mu3] = myzero;
                
		  for(int alpha = 0; alpha < Nd; ++alpha)
		    {
		        for(int a = 0; a < Nadj; ++a)
			      {
    for(int b =0; b < Nadj; ++b)
      {
    for(int c=0; c < Nadj; ++c)
      {*/
    /*this is factor of 2 too big, just like the toher methods*/
    /*fact = 0.5*cmplx(dabc(a,b,c),fabc(a,b,c));

      tmp = shift(F[mu2][alpha][c],FORWARD,mu3);
      Fop[mu1][mu2][mu3] += F[mu1][alpha][a]*ucol[mu3][b]*tmp*fact;

      tmp = shift(ucol[mu3][b],BACKWARD,mu3);
      tmp1 = shift(F[mu2][alpha][c],BACKWARD,mu3);
      Fop[mu1][mu2][mu3] -= F[mu1][alpha][a]*adj(tmp)*tmp1*fact;

      tmp = shift(F[mu1][alpha][a],FORWARD,mu3);
      Fop[mu1][mu2][mu3] -= tmp*adj(ucol[mu3][b])*F[mu2][alpha][c]*fact;

      tmp = shift(F[mu1][alpha][a],BACKWARD,mu3);
      tmp1 = shift(ucol[mu3][b],BACKWARD,mu3);
      Fop[mu1][mu2][mu3] += tmp*tmp1*F[mu2][alpha][c]*fact;



            }
	      }
	            }




		        for(int a = 0; a < Nadj; ++a)
			{*/
    /*this is factor of 2 too big, just like the toher methods*/
    /*tmp = shift(F[mu2][alpha][a],FORWARD,mu3);
      Fop[mu1][mu2][mu3] += F[mu1][alpha][a]*utr[mu3]*tmp;

      tmp = shift(utr[mu3],BACKWARD,mu3);
      tmp1 = shift(F[mu2][alpha][a],BACKWARD,mu3);
      Fop[mu1][mu2][mu3] -= F[mu1][alpha][a]*adj(tmp)*tmp1;
      
      tmp = shift(F[mu1][alpha][a],FORWARD,mu3);
      Fop[mu1][mu2][mu3] -= tmp*adj(utr[mu3])*F[mu2][alpha][a];
      
      tmp = shift(F[mu1][alpha][a],BACKWARD,mu3);
      tmp1 = shift(utr[mu3],BACKWARD,mu3);
      Fop[mu1][mu2][mu3] += tmp*tmp1*F[mu2][alpha][a];

            }

	      }
	        
	            }
		      }
      }
    */  
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
        for(int mu2 = 0; mu2 < Nd; ++mu2)
          {
            for(int mu3 = 0; mu3 < Nd; ++mu3)
              {
		FopSYM[mu1][mu2][mu3] = Real(1./6.)*(Fop[mu1][mu2][mu3]+Fop[mu1][mu3][mu2]+Fop[mu2][mu1][mu3]+Fop[mu2][mu3][mu1]+Fop[mu3][mu1][mu2]+Fop[mu3][mu2][mu1]);
	      }
	  }
      }

    /* combination corresponding to tau_1^(8) basis vector of interest */
    /* factor of (1/2) for fwd-bwd derivative*/
        
    F2Res[0] = 0.5*(sqrt3/sqrt2)*(FopSYM[0][1][1]-FopSYM[0][2][2]);   
    F2Res[1] = 0.5*(sqrt3/sqrt2)*(FopSYM[1][0][0]-FopSYM[1][2][2]);
    F2Res[2] = 0.5*(sqrt3/sqrt2)*(FopSYM[2][0][0]-FopSYM[2][1][1]);
    F2Res[3] = 0.5*(sqrt3/sqrt2)*(FopSYM[3][0][0]-FopSYM[3][1][1]);

    F2Res[4] = 0.5*(1./sqrt2)*(FopSYM[0][1][1]+FopSYM[0][2][2]-2.*FopSYM[0][3][3]);
    F2Res[5] = 0.5*(1./sqrt2)*(FopSYM[1][0][0]+FopSYM[1][2][2]-2.*FopSYM[1][3][3]);
    F2Res[6] = 0.5*(1./sqrt2)*(FopSYM[2][0][0]+FopSYM[2][1][1]-2.*FopSYM[2][3][3]);
    F2Res[7] = 0.5*(1./sqrt2)*(FopSYM[3][0][0]+FopSYM[3][1][1]-2.*FopSYM[3][2][2]);
    
 
    F2Res[8] = sqrt6*FopSYM[1][2][3];
    F2Res[9] = sqrt6*FopSYM[0][2][3];
    F2Res[10] = sqrt6*FopSYM[0][1][3];
    F2Res[11] = sqrt6*FopSYM[0][1][2];

    return F2Res;

  };

  multi1d<LatticeComplex> constructF2fourindex(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   *construct the box-sym delta G operator for n=2 for a given gauge field u
   *then correct basis
   *include checks for components that are zero for zero three-mtm
   */
    
  {
    
    multi1d<LatticeColorMatrix> f;
    multi2d<LatticeColorMatrix> ff;
    ff.resize(Nd, Nd);
    f.resize(Nd*(Nd-1)/2);

    mesField(f,u);

    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    Complex myzero=cmplx(Real(0.),Real(0.));

    multi3d<LatticeComplex> Fmunu;
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
    multi4d<LatticeComplex> Fop; /*F_{mu_1,alpha} F^{alpha}_{mu_2}*/
    multi4d<LatticeComplex> FopSYM;
    multi1d<LatticeComplex> F2Res;
       
    multi2d<LatticeComplex> ucol;
    multi1d<LatticeComplex> utr;
 
    int Nadj = Nc*Nc - 1;
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);       
    Fop.resize(Nd, Nd, Nd,  Nd);
    FopSYM.resize(Nd, Nd, Nd, Nd);
    Fmunu.resize(Nd,Nd,Nadj);
    F2Res.resize(16);
    ucol.resize(Nd, Nadj);
    utr.resize(Nd);

    constructSU3generators(tSU3);
        
    Real sqrt3 = sqrt(Real(3.));
    Real sqrt6 = sqrt(Real(6.));
    Real sqrt2 = sqrt(Real(2.));
        
    F = GetChromaFmunu(u);
    //F = GetCloverFmunu(u, 1);

    int count = 0;
    ff=myzero;
    for(int mu = 0; mu < Nd-1; ++mu)
      {
	for(int nu = mu+1; nu < Nd; ++nu)
	  {
	    ff[mu][nu] = f[count];
	    ff[nu][mu] = -f[count];

	    ++count;
	  }
      }

    LatticeColorMatrix tmp;
    LatticeColorMatrix tmp1;
    multi2d<LatticeColorMatrix> shiftF_fwd(Nd,Nd); shiftF_fwd=0;
    multi2d<LatticeColorMatrix> shiftF_bwd(Nd,Nd); shiftF_bwd=0;
    multi3d<LatticeColorMatrix> shiftF_fwd_fwd(Nd,Nd,Nd); shiftF_fwd_fwd=0;
    multi3d<LatticeColorMatrix> shiftF_fwd_bwd(Nd,Nd,Nd); shiftF_fwd_bwd=0;
    multi3d<LatticeColorMatrix> shiftF_bwd_fwd(Nd,Nd,Nd); shiftF_bwd_fwd=0;
    multi3d<LatticeColorMatrix> shiftF_bwd_bwd(Nd,Nd,Nd); shiftF_bwd_bwd=0;
    multi2d<LatticeColorMatrix> shiftG_fwd(Nd,Nd); shiftG_fwd=0;
    multi2d<LatticeColorMatrix> shiftG_bwd(Nd,Nd); shiftG_bwd=0;
    multi3d<LatticeColorMatrix> shiftG_fwd_fwd(Nd,Nd,Nd); shiftG_fwd_fwd=0;
    multi3d<LatticeColorMatrix> shiftG_fwd_bwd(Nd,Nd,Nd); shiftG_fwd_bwd=0;
    multi3d<LatticeColorMatrix> shiftG_bwd_fwd(Nd,Nd,Nd); shiftG_bwd_fwd=0;
    multi3d<LatticeColorMatrix> shiftG_bwd_bwd(Nd,Nd,Nd); shiftG_bwd_bwd=0;

    LatticeColorMatrix shiftU; shiftU=0;
    LatticeComplex tmpFF; tmpFF=0;
    LatticeComplex tmpFB; tmpFB=0;
    LatticeComplex tmpBF; tmpBF=0;
    LatticeComplex tmpBB; tmpBB=0;
  

    /* CHECK THERE*/
    /*construct F_{mu_1,a} D_{mu_3} D_{mu_4} G_{mu_2}^{a}*/
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
	for(int mu2 = 0; mu2 < Nd; ++mu2)
	  {
	    /* Setting up some shifts */
	    for(int a =0; a<Nd; ++a){

	      for(int alpha = 0; alpha < Nd; ++alpha)
		{
		  shiftU = shift(u[alpha],BACKWARD,alpha);
		  shiftF_fwd[a][alpha] = u[alpha]*shift(ff[mu1][a],FORWARD,alpha)*adj(u[alpha]);
		  shiftF_bwd[a][alpha] = adj(shiftU)*shift(ff[mu1][a],BACKWARD,alpha)*shiftU;
		  shiftG_fwd[a][alpha] = u[alpha]*shift(ff[mu2][a],FORWARD,alpha)*adj(u[alpha]);
		  shiftG_bwd[a][alpha] = adj(shiftU)*shift(ff[mu2][a],BACKWARD,alpha)*shiftU;
		}
	      for(int alpha = 0; alpha < Nd; ++alpha)
		{
		  for(int beta = 0; beta < Nd; ++beta)
		    {
		      shiftU = shift(u[beta],BACKWARD,beta);

		      shiftF_fwd_fwd[a][alpha][beta] = u[beta]*shift(shiftF_fwd[a][alpha],FORWARD,beta)*adj(u[beta]);
		      shiftF_fwd_bwd[a][alpha][beta] = adj(shiftU)*shift(shiftF_fwd[a][alpha],BACKWARD,beta)*shiftU;
		      shiftF_bwd_fwd[a][alpha][beta] = u[beta]*shift(shiftF_bwd[a][alpha],FORWARD,beta)*adj(u[beta]);
		      shiftF_bwd_bwd[a][alpha][beta] = adj(shiftU)*shift(shiftF_bwd[a][alpha],BACKWARD,beta)*shiftU;


		      shiftG_fwd_fwd[a][alpha][beta] = u[beta]*shift(shiftG_fwd[a][alpha],FORWARD,beta)*adj(u[beta]);
		      shiftG_fwd_bwd[a][alpha][beta] = adj(shiftU)*shift(shiftG_fwd[a][alpha],BACKWARD,beta)*shiftU;
		      shiftG_bwd_fwd[a][alpha][beta] = u[beta]*shift(shiftG_bwd[a][alpha],FORWARD,beta)*adj(u[beta]);
		      shiftG_bwd_bwd[a][alpha][beta] = adj(shiftU)*shift(shiftG_bwd[a][alpha],BACKWARD,beta)*shiftU;

		    }
		}

	    } //a loop


	    for(int mu4 = 0; mu4 < Nd; ++mu4)
	      {
		for(int mu3 = 0; mu3 < Nd; ++mu3)
		  {

		    Fop[mu1][mu2][mu3][mu4] = myzero;
		    for(int a = 0; a < Nd; ++a)
		      {
			tmpFF = Real(0.5)*traceColor(ff[mu1][a] * (shiftG_fwd_fwd[a][mu3][mu4] 
								    - shiftG_fwd_bwd[a][mu3][mu4] 
								    - shiftG_bwd_fwd[a][mu3][mu4] 
								   + shiftG_bwd_bwd[a][mu3][mu4] ) );

			tmpFB = Real(-0.5)*traceColor( shiftF_bwd[a][mu4] * shiftG_fwd[a][mu3] 
						              - shiftF_fwd[a][mu4] * shiftG_fwd[a][mu3] 
						              - shiftF_bwd[a][mu4] * shiftG_bwd[a][mu3] 
						       + shiftF_fwd[a][mu4] * shiftG_bwd[a][mu3] );

			tmpBF = Real(-0.5)*traceColor( shiftF_bwd[a][mu3] * shiftG_fwd[a][mu4] 
						              - shiftF_fwd[a][mu3] * shiftG_fwd[a][mu4] 
						              - shiftF_bwd[a][mu3] * shiftG_bwd[a][mu4] 
						       + shiftF_fwd[a][mu3] * shiftG_bwd[a][mu4] );

			tmpBB = Real(0.5)*traceColor((shiftF_fwd_fwd[a][mu3][mu4] 
						            - shiftF_fwd_bwd[a][mu3][mu4] 
						            - shiftF_bwd_fwd[a][mu3][mu4] 
						      + shiftF_bwd_bwd[a][mu3][mu4] ) *ff[mu2][a]);


			
			Fop[mu1][mu2][mu3][mu4] += Real(0.125)*Real(0.25)*(tmpFF - tmpFB - tmpBF + tmpBB);
			
		      }
		  }
	      }
	  }
      }

    
    for(int mu1 = 0; mu1 < Nd; ++mu1)
      {
        for(int mu2 = 0; mu2 < Nd; ++mu2)
          {
            for(int mu3 = 0; mu3 < Nd; ++mu3)
              {
		for(int mu4 = 0; mu4 < Nd; ++mu4)
		  {
		    FopSYM[mu1][mu2][mu3][mu4] = Real(1./24.)*(Fop[mu1][mu2][mu3][mu4]+Fop[mu1][mu3][mu2][mu4]+Fop[mu2][mu1][mu3][mu4]+Fop[mu2][mu3][mu1][mu4]+Fop[mu3][mu1][mu2][mu4]+Fop[mu3][mu2][mu1][mu4]+Fop[mu1][mu2][mu4][mu3]+Fop[mu1][mu3][mu4][mu2]+Fop[mu2][mu1][mu4][mu3]+Fop[mu2][mu3][mu4][mu1]+Fop[mu3][mu1][mu4][mu2]+Fop[mu3][mu2][mu4][mu1]+Fop[mu1][mu4][mu2][mu3]+Fop[mu1][mu4][mu3][mu2]+Fop[mu2][mu4][mu1][mu3]+Fop[mu2][mu4][mu3][mu1]+Fop[mu3][mu4][mu1][mu2]+Fop[mu3][mu4][mu2][mu1]+Fop[mu4][mu1][mu2][mu3]+Fop[mu4][mu1][mu3][mu2]+Fop[mu4][mu2][mu1][mu3]+Fop[mu4][mu2][mu3][mu1]+Fop[mu4][mu3][mu1][mu2]+Fop[mu4][mu3][mu2][mu1]);
		  }
	      }
	  }
      }

    // tau_1^(2) basis
      
    //    F2Res[0] = traceColor(shiftF_fwd[0][1]);
    //F2Res[1] = traceColor(shiftF_fwd_bwd[0][1][2]);

    F2Res[0] = (sqrt3/sqrt2)*(FopSYM[0][0][1][1]+FopSYM[2][2][3][3]-FopSYM[0][0][2][2]-FopSYM[1][1][3][3]);
    F2Res[1] = (-1./sqrt2)*(FopSYM[0][0][1][1]+FopSYM[2][2][3][3]+FopSYM[0][0][2][2]+FopSYM[1][1][3][3] -2.*FopSYM[0][0][3][3]-2.*FopSYM[1][1][2][2]);


    // tau_2^(1) basis

    F2Res[3] = 2.*sqrt6*FopSYM[0][1][2][3];

    // tau_3^(6) basis

    F2Res[4] = (1./sqrt2)*(FopSYM[0][0][0][1]+FopSYM[1][1][1][0])-(3./sqrt2)*(FopSYM[2][2][0][1]+FopSYM[3][3][0][1]);
    F2Res[5] = (1./sqrt2)*(FopSYM[0][0][0][2]+FopSYM[2][2][2][0])-(3./sqrt2)*(FopSYM[1][1][0][2]+FopSYM[3][3][0][2]);
    F2Res[6] = (1./sqrt2)*(FopSYM[0][0][0][3]+FopSYM[3][3][3][0])-(3./sqrt2)*(FopSYM[1][1][0][3]+FopSYM[2][2][0][3]);
    F2Res[7] = (1./sqrt2)*(FopSYM[1][1][1][2]+FopSYM[2][2][2][1])-(3./sqrt2)*(FopSYM[0][0][1][2]+FopSYM[3][3][1][2]);
    F2Res[8] = (1./sqrt2)*(FopSYM[1][1][1][3]+FopSYM[3][3][3][1])-(3./sqrt2)*(FopSYM[0][0][1][3]+FopSYM[2][2][1][3]);
    F2Res[9] = (1./sqrt2)*(FopSYM[2][2][2][3]+FopSYM[3][3][3][2])-(3./sqrt2)*(FopSYM[0][0][2][3]+FopSYM[1][1][2][3]);

    // tau_2^(6) basis

    F2Res[10] = sqrt6*(FopSYM[2][2][0][1] - FopSYM[3][3][0][1]);
    F2Res[11] = sqrt6*(FopSYM[1][1][0][2] - FopSYM[3][3][0][2]);
    F2Res[12] = sqrt6*(FopSYM[1][1][0][3] - FopSYM[2][2][0][3]);
    F2Res[13] = sqrt6*(FopSYM[0][0][1][2] - FopSYM[3][3][1][2]);
    F2Res[14] = sqrt6*(FopSYM[0][0][1][3] - FopSYM[2][2][1][3]);
    F2Res[15] = sqrt6*(FopSYM[0][0][2][3] - FopSYM[1][1][2][3]);

    return F2Res;

  };

    
  multi3d<LatticeComplex> GetChromaFmunu(const multi1d<LatticeColorMatrix> & u)
  {
    multi1d<LatticeColorMatrix> f;
    f.resize(Nd*(Nd-1)/2);
        
    mesField(f,u);
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    int Nadj = Nc*Nc - 1;
    tSU3.resize(Nadj);
    F.resize(Nd*(Nd-1)/2, Nadj);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
    constructSU3generators(tSU3);
        
    multi3d<LatticeComplex> Fmunu;
    Fmunu.resize(Nd,Nd,Nadj);
        
    /* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
    for(int mu = 0; mu < Nd*(Nd-1)/2; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
                
	    F[mu][c] = Real(2.)*traceColor(tSU3[c]*f[mu]);
                
	  };
      }
        
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int c = 0; c < Nadj; ++c)
	      {
		Fmunu[mu][nu][c] = myzero;
	      }
                
	  }
      }

        
    int count = 0;
        
    for(int mu = 0; mu < Nd-1; ++mu)
      {
	for(int nu = mu+1; nu < Nd; ++nu)
	  {
	    for(int c = 0; c < Nadj; ++c)
	      {
		Fmunu[mu][nu][c] = F[count][c];
		Fmunu[nu][mu][c] = -F[count][c];
	      }
                
	    ++count;
	  }
      }
        
    return Fmunu;
        
  }
    
    
  multi3d<LatticeComplex> GetCloverFmunu(const multi1d<LatticeColorMatrix> & u, int clover_loops)
    
  /*! \param u gauge field (read)
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    int Nadj = Nc*Nc - 1;
        
    multi2d<LatticeColorMatrix> clsum;
    clsum.resize(Nd,Nd);
        
    multi2d<LatticeColorMatrix> cl_tmp1;
    cl_tmp1.resize(Nd,Nd);
    multi2d<LatticeColorMatrix> cl_tmp2;
    cl_tmp2.resize(Nd,Nd);
    multi2d<LatticeColorMatrix> cl_tmp3;
    cl_tmp3.resize(Nd,Nd);
    multi2d<LatticeColorMatrix> cl_tmp4;
    cl_tmp4.resize(Nd,Nd);
    multi2d<LatticeColorMatrix> cl_tmp32;
    cl_tmp32.resize(Nd,Nd);
    multi2d<LatticeColorMatrix> cl_tmp42;
    cl_tmp42.resize(Nd,Nd);
        
        
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    constructSU3generators(tSU3);
        
    double k1 = 0.;
    double k2 = 0.;
    double k3 = 0.;
    double k4 = 0.;
    double k5 = 0.;
        


        
    switch( clover_loops )
      {
      case 1:
                
	k1 = 1.;
	cl_tmp1 = CalculateCloverTerm(u, 1, 1);
                
	for(int mu = 0; mu < Nd; ++mu)
	  {
	    for(int nu = 0; nu < Nd; ++nu)
	      {
		clsum[mu][nu] = 2.*k1*cl_tmp1[mu][nu];
	      }
	  };
                
	break;
                
      case 3:
                
	k5 = 1./90.;
	k1 = 19./9. - 55.*k5;
	k2 = 1./36. - 16.*k5;
                
	cl_tmp1 = CalculateCloverTerm(u, 1, 1);
	cl_tmp2 = CalculateCloverTerm(u, 2, 2);
	cl_tmp3 = CalculateCloverTerm(u, 3, 3);
                
	for(int mu = 0; mu < Nd; ++mu)
	  {
	    for(int nu = 0; nu < Nd; ++nu)
	      {
		clsum[mu][nu] = 2.*k1*cl_tmp1[mu][nu];
		clsum[mu][nu] += 2.*k2*cl_tmp2[mu][nu];
		clsum[mu][nu] += 2.*k5*cl_tmp3[mu][nu];
	      }
	  };

	break;
                
      case 4:
                
	k1 = 19./9.;
	k2 = 1./36.;
	k3 = - 32./45;
	k4 = 1./15.;
                
	cl_tmp1 = CalculateCloverTerm(u, 1, 1);
	cl_tmp2 = CalculateCloverTerm(u, 2, 2);
	cl_tmp3 = CalculateCloverTerm(u, 1, 2);
	cl_tmp4 = CalculateCloverTerm(u, 1, 3);
	cl_tmp32 = CalculateCloverTerm(u, 2, 1);
	cl_tmp42 = CalculateCloverTerm(u, 3, 1);
                
	for(int mu = 0; mu < Nd; ++mu)
	  {
	    for(int nu = 0; nu < Nd; ++nu)
	      {
		clsum[mu][nu] = 2.*k1*cl_tmp1[mu][nu];
		clsum[mu][nu] += 2.*k2*cl_tmp2[mu][nu];
		clsum[mu][nu] += k3*cl_tmp3[mu][nu];
		clsum[mu][nu] += k4*cl_tmp4[mu][nu];
		clsum[mu][nu] += k3*cl_tmp32[mu][nu];
		clsum[mu][nu] += k4*cl_tmp42[mu][nu];
	      }
	  };
                
	break;
                
                
      default:
	QDP_error_exit("Unknown number of clover loops", clover_loops);
      };

        
 
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
                
	    /* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
	    for(int c = 0; c < Nadj; ++c){
                    
	      F[mu][nu][c] = Real(2.)*traceColor(tSU3[c]*clsum[mu][nu]);
	    };
	  };
      };
        
        
    return F;
        
  }
    
    
  multi3d<LatticeComplex> CalculateDumbFmunu(const multi1d<LatticeColorMatrix> & u)
    
  /*! \param u gauge field (read)
   */
    
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi3d<LatticeComplex> F; /*F{\mu\nu}^{a}*/
        
    int Nadj = Nc*Nc - 1;
        
    tSU3.resize(Nadj);
    F.resize(Nd, Nd, Nadj);
        
    constructSU3generators(tSU3);
        
    multi2d<LatticeColorMatrix> clover;
        
    clover.resize(Nd,Nd);
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    /* calculate the musize x nusize clover term */
	    clover[mu][nu] = u[mu]*shift(u[nu], FORWARD, mu)*adj(shift(u[mu], FORWARD, nu))*adj(u[nu]);
                
	    clover[mu][nu] += adj(shift(u[nu], BACKWARD, nu))*shift(u[mu], BACKWARD, nu)*shift(shift(u[nu], BACKWARD, nu), FORWARD, mu)*adj(u[mu]);
                
	    clover[mu][nu] += u[nu]*adj(shift(shift(u[mu], FORWARD, nu), BACKWARD, mu))*adj(shift(u[nu], BACKWARD, mu))*shift(u[mu], BACKWARD, mu);
                
	    clover[mu][nu] += adj(shift(u[mu],BACKWARD,mu))*adj(shift(shift(u[nu], BACKWARD, mu),BACKWARD, nu))*shift(shift(u[mu],BACKWARD,mu),BACKWARD,nu)*shift(u[nu],BACKWARD,nu);
                
	    clover[mu][nu] -= adj(clover[mu][nu]);
	    clover[mu][nu] = clover[mu][nu]/8.0;
	  };
      };
        
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
                
	    /* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
	    for(int c = 0; c < Nadj; ++c){
                    
	      F[mu][nu][c] = Real(2.)*traceColor(tSU3[c]*clover[mu][nu]);
	    };
	  };
      };
        
        
    return F;

        
  }
    
    
  /*REMEMBER TO GAUGE_FIX BEFORE RUNNING THIS*/
  multi2d<LatticeComplex> ConstructGluon(const multi1d<LatticeColorMatrix> & u)
  {
        
    multi1d<ColorMatrix> tSU3; /*adjoint SU(3) generators*/
    multi2d<LatticeComplex> MYgluon; /*F{\mu\nu}^{a}*/
        
    int Nadj = Nc*Nc - 1;
    tSU3.resize(Nadj);
    MYgluon.resize(Nd,Nadj);
        
    Complex myzero=cmplx(Real(0.),Real(0.));
        
    constructSU3generators(tSU3);
        
    /* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
    /*for(int mu = 0; mu < Nd; ++mu)
        {
            for(int nu = 0; nu < Nd; ++nu)
            {
                MYgluon[mu][nu] = myzero;
                
                for(int c = 0; c < Nadj; ++c)
                {
                    
		MYgluon[mu][nu] += Real(4.)*traceColor(tSU3[c]*(u[mu]-adj(u[mu]))-traceColor(u[mu]-adj(u[mu])))*traceColor(tSU3[c]*(u[nu]-adj(u[nu]))-traceColor(u[nu]-adj(u[nu])));
                
  };
  }
  }*/
        
        
    /* Projection onto  generators : Y = y_i t_i where y_i = 2 tr(t_i Y) */
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int c = 0; c < Nadj; ++c)
	  {
                    
	    MYgluon[mu][c] = Real(2.)*traceColor(tSU3[c]*(u[mu]-adj(u[mu])))-traceColor(u[mu]-adj(u[mu])/Real(3.));
                    
	  };
      }
        
    return MYgluon;
        
  }


    
  multi2d<LatticeColorMatrix> CalculateCloverTerm(const multi1d<LatticeColorMatrix> & u, int musize, int nusize)
    
  /*! \param u gauge field (read)
   */
    
  {
        
    multi2d<LatticeColorMatrix> clover;
        
    clover.resize(Nd,Nd);
        
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    /* calculate the musize x nusize clover term */
	    clover[mu][nu] = u[mu]*multishift(u[nu], FORWARD, mu, musize)*adj(multishift(u[mu], FORWARD, nu, nusize))*adj(u[nu]);
                
	    clover[mu][nu] += adj(multishift(u[nu], BACKWARD, nu, nusize))*multishift(u[mu], BACKWARD, nu, nusize)*multishift(multishift(u[nu], BACKWARD, nu, nusize), FORWARD, mu, musize)*adj(u[mu]);
                
	    clover[mu][nu] += u[nu]*adj(multishift(multishift(u[mu], FORWARD, nu, nusize), BACKWARD, mu, musize))*adj(multishift(u[nu], BACKWARD, mu, musize))*multishift(u[mu], BACKWARD, mu, musize);
                
	    clover[mu][nu] += adj(multishift(u[mu],BACKWARD,mu,musize))*adj(multishift(multishift(u[nu], BACKWARD, mu, musize),BACKWARD, nu, nusize))*multishift(multishift(u[mu],BACKWARD,mu,musize),BACKWARD,nu,nusize)*multishift(u[nu],BACKWARD,nu,nusize);
                
	    clover[mu][nu] -= adj(clover[mu][nu]);
	    clover[mu][nu] = clover[mu][nu]/8.0;
	  };
      };
        

    return clover;

  }
    
    
  const LatticeColorMatrix multishift(const LatticeColorMatrix & uu, int sign, int dir, int mag)
  {
    LatticeColorMatrix my_res;
    LatticeColorMatrix my_temp = uu;
        
    for(int ii = 0; ii < mag; ++ii)
      {
	my_res = shift(my_temp, sign, dir);
	my_temp = my_res;
      };
        
    return my_res;
  };

    
    
  double LeviCivita4D(int ii, int jj, int kk, int ll)
  /* Construct the 4D levi-civita */
  {
    multi4d<int> antisym_tensor4d;
        
    antisym_tensor4d.resize(4,4,4,4);
    antisym_tensor4d = 0.;
        
    antisym_tensor4d(0,1,2,3) = 1.;
    antisym_tensor4d(0,1,3,2) = -1.;
    antisym_tensor4d(0,2,1,3) = -1.;
    antisym_tensor4d(0,2,3,1) = 1.;
    antisym_tensor4d(0,3,1,2) = 1.;
    antisym_tensor4d(0,3,2,1) = -1.;
        
    antisym_tensor4d(1,0,2,3) = -1.;
    antisym_tensor4d(1,0,3,2) = 1.;
    antisym_tensor4d(1,2,0,3) = 1.;
    antisym_tensor4d(1,2,3,0) = -1.;
    antisym_tensor4d(1,3,0,2) = -1.;
    antisym_tensor4d(1,3,2,0) = 1.;
        
    antisym_tensor4d(2,0,1,3) = 1.;
    antisym_tensor4d(2,0,3,1) = -1.;
    antisym_tensor4d(2,1,0,3) = -1.;
    antisym_tensor4d(2,1,3,0) = 1.;
    antisym_tensor4d(2,3,0,1) = 1.;
    antisym_tensor4d(2,3,1,0) = -1.;
        
    antisym_tensor4d(3,0,1,2) = -1.;
    antisym_tensor4d(3,0,2,1) = 1.;
    antisym_tensor4d(3,1,0,2) = 1.;
    antisym_tensor4d(3,1,2,0) = -1.;
    antisym_tensor4d(3,2,0,1) = -1.;
    antisym_tensor4d(3,2,1,0) = 1.;
        
    return antisym_tensor4d(ii,jj,kk,ll);
        
        
  };

  Real fabc(int a, int b, int c)
  {
    Real root3on2=Real(sqrt(3.)/2.);
    multi3d<Real> f(8,8,8); f =0;
    f(0,1,2) = 1;
    f(0,3,6)=0.5;
    f(0,4,5) = -0.5;
    f(1,3,5)=0.5;
    f(1,4,6)=0.5;
    f(2,3,4)=0.5;
    f(2,5,6)=-0.5;
    f(3,4,7)=root3on2;
    f(5,6,7)=root3on2;
     
    int sign=1;
    int signtmp = sign;
    multi1d<int> index(3); index(0)=a; index(1)=b; index(2)=c;
    multi1d<int> indextmp(3); indextmp=index;
    if(index(0)<index(1)) 
      {     
      }
    else
      {
	signtmp=sign*(-1.);
	sign = signtmp;
	indextmp(1)=index(0);indextmp(0)=index(1);
	index=indextmp;
      }
    if(index(1)<index(2)) 
      {
      }
    else
      {
	signtmp=sign*(-1.);
	sign = signtmp;
	indextmp(1)=index(2);indextmp(2)=index(1);
	index=indextmp;
      }
    if(index(0)<index(1)) 
      {
      }
    else
      {
	signtmp=sign*(-1.);
	sign = signtmp;
	indextmp(1)=index(0);indextmp(0)=index(1);
	index=indextmp;
      }
    
    return f(index(0),index(1),index(2))*sign;

  }

  Real dabc(int a, int b, int c)
  {
    Real oneonroot3=Real(1./sqrt(3.));
    multi3d<Real> d(8,8,8); d =0;
    d(0,0,7) = oneonroot3;
    d(1,1,7) = oneonroot3;
    d(2,2,7) = oneonroot3;
    d(7,7,7) = -1.*oneonroot3;
    d(3,3,7) = -0.5*oneonroot3;
    d(4,4,7) = -0.5*oneonroot3;
    d(5,5,7) = -0.5*oneonroot3;
    d(6,6,7) = -0.5*oneonroot3;
    d(0,3,5) = 0.5;
    d(0,4,6) = 0.5;
    d(1,3,6) = -0.5;
    d(1,4,5) = 0.5;
    d(2,3,3) = 0.5;
    d(2,4,4) = 0.5;
    d(2,5,5) = -0.5;
    d(2,6,6) = -0.5;
     
    multi1d<int> index(3); index(0)=a; index(1)=b; index(2)=c;
    multi1d<int> indextmp(3); indextmp=index;
    if(index(0)<index(1)) 
      {     
      }
    else
      {
	indextmp(1)=index(0);indextmp(0)=index(1);
	index=indextmp;
      }
    if(index(1)<index(2)) 
      {
      }
    else
      {
	indextmp(1)=index(2);indextmp(2)=index(1);
	index=indextmp;
      }
    if(index(0)<index(1)) 
      {
      }
    else
      {
	indextmp(1)=index(0);indextmp(0)=index(1);
	index=indextmp;
      }
    
    return d(index(0),index(1),index(2));

  }

    
  void constructSU3generators(multi1d < ColorMatrix > & tSU3)
  /* Construct the generators of SU(3) */
  {
    Real oneover2sqrt3 = Real(1./sqrt(3.)/2.);
    Complex plusIon2=cmplx(Real(0.),Real(0.5));
    Complex minusIon2=cmplx(Real(0.),Real(-0.5));
        
    tSU3 = 0;
        
    pokeColor(tSU3[0], Real(0.5), 0, 1);
    pokeColor(tSU3[0], Real(0.5), 1, 0);
        
    pokeColor(tSU3[1], minusIon2, 0, 1);
    pokeColor(tSU3[1], plusIon2, 1, 0);
        
    pokeColor(tSU3[2], Real(0.5), 0, 0);
    pokeColor(tSU3[2], Real(-0.5), 1, 1);
        
    pokeColor(tSU3[3], Real(0.5), 0, 2);
    pokeColor(tSU3[3], Real(0.5), 2, 0);
        
    pokeColor(tSU3[4], minusIon2, 0, 2);
    pokeColor(tSU3[4], plusIon2, 2, 0);
        
    pokeColor(tSU3[5], Real(0.5), 1, 2);
    pokeColor(tSU3[5], Real(0.5), 2, 1);
        
    pokeColor(tSU3[6], minusIon2, 1, 2);
    pokeColor(tSU3[6], plusIon2, 2, 1);
        
    pokeColor(tSU3[7], oneover2sqrt3, 0, 0);
    pokeColor(tSU3[7], oneover2sqrt3, 1, 1);
    pokeColor(tSU3[7], Real(-2.*oneover2sqrt3), 2, 2);
        
  };
    
    
    
}  // end namespace Chroma
