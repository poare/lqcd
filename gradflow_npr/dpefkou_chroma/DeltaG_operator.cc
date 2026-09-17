#include "chromabase.h"
#include "util/ft/sftmom.h"
#include "util/ft/time_slice_set.h"
#include "util/ft/single_phase.h"
#include "qdp_util.h"
#include "util/info/proginfo.h"
#include "chroma.h"
#include <fftw3.h>
#include "meas/glue/mesfield.h"

#include "meas/hadron/DeltaG_operator.h"


namespace Chroma
{

multi2d<DComplex>
FFT4d(
    const LatticeComplex & cf, 
    const SftMom & phases
) {
    int Nx = Layout::lattSize()[0];
    int Ny = Layout::lattSize()[1];
    int Nz = Layout::lattSize()[2];
    int Nt = Layout::lattSize()[3];
    int V4 = Nx * Ny * Nz * Nt;
    const Real twopi = 6.283185307179586476925286;
    // Get ordered coords for this node
    multi1d<multi1d<Int>> coord(Nd);
    for (int mu = 0; mu < Nd; mu++) {
        coord[mu].resize(Layout::sitesOnNode());
        QDP_extract(coord[mu], Layout::latticeCoordinate(mu), all);
    }
    // Allocate a 4d FFTW lattice
    fftw_complex * field;
    field = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*V4);
    // make sure padding is all zeros
    for (int ii = 0; ii < V4; ii ++) {
        field[ii][0] = 0.0;
        field[ii][1] = 0.0;
    }
    // Get the data for this lattice in an unraveled 1d array
    multi1d<DComplex> cf_ravel(Layout::sitesOnNode());
    QDP_extract(cf_ravel, cf, all);
    // Copy local data into field to be transformed
    for (int ii = 0; ii < Layout::sitesOnNode(); ii++) {
        int x = toInt(coord[0][ii]); int y = toInt(coord[1][ii]); 
        int z = toInt(coord[2][ii]); int t = toInt(coord[3][ii]);
        int fft_idx = t + Nt * (z + Nz * (y + Ny * x)); // row-major order
        field[fft_idx][0] = toDouble(real(cf_ravel[ii]));
        field[fft_idx][1] = toDouble(imag(cf_ravel[ii]));
    }
  // Do inplace 4d FFT on zero-padded field
    int fftdims[4];
    fftdims[0] = Nx; fftdims[1] = Ny; fftdims[2] = Nz; fftdims[3] = Nt;
    fftw_plan fft_plan = fftw_plan_dft(4, fftdims, 
        field, field, //inplace FFT
        FFTW_BACKWARD, //FFTW_FORWARD, // match stupid chroma exp[+ipx] convention
        FFTW_ESTIMATE
    );
    fftw_execute(fft_plan);
    fftw_destroy_plan(fft_plan);
    // Pick out momenta we actually want, load into array to sum
    multi2d<DComplex> hsum(phases.numMom(),1);
    multi1d<int> mom;
    multi1d<int> x0 = phases.getOriginOffset();
    for (int mom_num = 0; mom_num < phases.numMom(); mom_num++) {
        mom = phases.numToMom(mom_num);
        int qx = mom[0]; int qy = mom[1]; int qz = mom[2]; int qt = mom[3];
        qx = (qx+Nx)%Nx; qy = (qy+Ny)%Ny; qz = (qz+Nz)%Nz; qt = (qt+Nt)%Nt; // wrap negative momenta around for indexing
        int fft_idx = qt + Nt * (qz + Nz * (qy + Ny * qx)); // row-major order
        // multiply by extra factor of exp[-i p.x0] for source position
        Real minus_p_dot_x0 = -twopi * (
              Real(qx*x0[0])/Real(Nx) + Real(qy*x0[1])/Real(Ny) 
            + Real(qz*x0[2])/Real(Nz) + Real(qt*x0[3])/Real(Nt));
        DComplex unphased = cmplx(Real(field[fft_idx][0]), Real(field[fft_idx][1]));
        hsum[mom_num] = cmplx(
            real(unphased) * cos(minus_p_dot_x0) - imag(unphased) * sin(minus_p_dot_x0),
            real(unphased) * sin(minus_p_dot_x0) + imag(unphased) * cos(minus_p_dot_x0)
        );
    }
    // Sum over ranks
    QDPInternal::globalSumArray(hsum);
    // Free FFT memory
    fftw_free(field);
    return hsum;
}



  void QuarkGFF(const multi1d<LatticeColorMatrix>& u,
		const LatticePropagator& quark_propagator,
		const LatticePropagator& seq_quark_prop,
		const SftMom& phases,
		int gamma_insertion,
		int t0,
		XMLWriter& xml)
  {
    int G5 = Ns*Ns-1;
    LatticePropagator anti_quark_prop = adj(Gamma(G5) * seq_quark_prop * Gamma(G5));

    multi2d<LatticeComplex> Ot; //O_mu nu
    Ot.resize(Nd,Nd);
    SpinMatrix g_one = 1.0;
    multi1d<SpinMatrix> mygamma;
    mygamma.resize(Nd);
    mygamma[0] = g_one*Gamma(1);
    mygamma[1] = g_one*Gamma(2);
    mygamma[2] = g_one*Gamma(4);
    mygamma[3] = g_one*Gamma(8);


    
    for (int mu=0; mu<Nd; ++mu)
      {
      for (int nu=0; nu<Nd; ++nu)
	{
       	  Ot[mu][nu] = 0.25*trace(anti_quark_prop*mygamma[mu]*u[nu]*shift(quark_propagator, FORWARD, nu)*Gamma(gamma_insertion)-anti_quark_prop*mygamma[mu]*shift(adj(u[nu]),BACKWARD,nu)*shift(quark_propagator,BACKWARD,nu)*Gamma(gamma_insertion)-shift(anti_quark_prop,FORWARD,nu)*mygamma[mu]*adj(u[nu])*quark_propagator*Gamma(gamma_insertion)+shift(anti_quark_prop, BACKWARD, nu)*mygamma[mu]*shift(u[nu],BACKWARD,nu)*quark_propagator*Gamma(gamma_insertion));


	}
      }

    multi1d<LatticeComplex> Ops;
    Ops.resize(9);
    Ops[0] = Real(1./2.)*(Ot[0][0]+Ot[1][1]-Ot[2][2]-Ot[3][3]);
    Ops[1] = Real(1./sqrt(2.))*(Ot[2][2]-Ot[3][3]);
    Ops[2] = Real(1./sqrt(2.))*(Ot[0][0]-Ot[1][1]);
    Ops[3] = Real(1./sqrt(2.))*(Ot[0][1]+Ot[1][0]);
    Ops[4] = Real(1./sqrt(2.))*(Ot[0][2]+Ot[2][0]);
    Ops[5] = Real(1./sqrt(2.))*(Ot[0][3]+Ot[3][0]);
    Ops[6] = Real(1./sqrt(2.))*(Ot[1][2]+Ot[2][1]);
    Ops[7] = Real(1./sqrt(2.))*(Ot[1][3]+Ot[3][1]);
    Ops[8] = Real(1./sqrt(2.))*(Ot[2][3]+Ot[3][2]);

    multi2d<DComplex> Ops_FT;
    int llength = phases.numSubsets();
    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml, 9);
    push(xml_array, "Operators");
    for(int ii = 0; ii < 9; ii++)
      {
        Ops_FT = phases.sft(Ops[ii]);
        push(xml_array);
        write(xml_array, "Op_No", ii);
        XMLArrayWriter xml_sink_mom(xml, phases.numMom());
        push(xml_sink_mom, "momenta");

	for(int mom_num = 0; mom_num < phases.numMom(); mom_num++)
          {
            push(xml_sink_mom);
            write(xml_sink_mom, "sink_mom_num", mom_num);
            write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
	    multi1d<Complex> cur3ptfn(llength);
	    for (int t = 0; t < llength; ++t)
	      {
	    	int t_eff = (t-t0+llength) % llength;
	    cur3ptfn[t_eff] = Complex(Ops_FT[mom_num][t]);
	      }
            write(xml_sink_mom, "operator", cur3ptfn);
            pop(xml_sink_mom);

          }
        pop(xml_sink_mom);
        pop(xml_array);

      }
    pop(xml);
  };

  void Stupid(const multi1d<LatticeColorMatrix>& u)
  {
    multi1d<LatticeColorMatrix> ff;
    ff.resize(6);
    ff = cmplx(Real(0.),Real(0.));
    mesField(ff,u);
    Complex meow;
    meow = cmplx(Real(0.),Real(0.));
    for (int mu = 0; mu < 6; ++mu)
      {meow = sum(traceColor(ff[mu]));
	QDPIO::cout << meow << std::endl;}

  }


  void GluonPropagator_new(const multi1d<LatticeColorMatrix>& u,
			   multi1d<int>& nn,
			   XMLWriter& xml)

  {
    
    Set sft_set;
    sft_set.make(TimeSliceFunc(-1));
    int n_min = nn[0];
    int n_max = nn[1];

    /*int num_mom = 4*(n_max-n_min+1);
    multi3d<int> mom_list;
    mom_list.resize(4,num_mom,Nd);
    mom_list = 0;
    for(int tau = 0; tau < Nd; ++ tau)
      {
	int mom_num = 0;
	mom_list[tau][mom_num][tau] = 0;
	for (int k = n_min; k <= n_max; ++k)
	  {
	  for (int i = 1; i <= 3; ++i)
	    {
	    for (int j = 0; j <= 3; ++j)
	      {
		if ((tau+i)%4 == 3){mom_list[tau][mom_num+j][(tau+i)%4] = 2*k;}
		else {mom_list[tau][mom_num+j][(tau+i)%4] = k;}
	      }
	    mom_list[tau][mom_num+i][(tau+i)%4] *= -1;
	    }
	  mom_num += 4;
	  }
	  }*/

    int num_mom = 0;                                                                        
    for(int i = 0; i < Nd; ++i){                                                              
      int blep = 1;                                                                           
      for(int j = 0; j < Nd; ++j)                                                             
        if (j!=i){                                                                            
          blep *= 2*(n_max-n_min+1);}                                                         
      num_mom += blep;                                                                        
      }
    multi2d<int> mom_list;
    mom_list.resize(num_mom, Nd) ;
    mom_list = 0; 

    int mom_num = 0;
    for(int i = 0; i < Nd; ++i)
      {
	mom_list[mom_num][i] = 0;
	for(int n1 = n_min; n1 <= n_max; ++n1)
	  for(int n2 = n_min; n2 <= n_max; ++n2)
	    for(int j2 = 0; j2 < 2; ++j2)
	      for(int n3 = n_min; n3 <= n_max; ++n3)
		for(int j3 = 0; j3 < 2; ++j3)
		  {
		    mom_list[mom_num][(i+1)%4] = n1;
		    mom_list[num_mom-mom_num-1][(i+1)%4] = -n1;
		    if(j2==0) {
		      mom_list[mom_num][(i+2)%4] = n2;
		      mom_list[num_mom-mom_num-1][(i+2)%4] = -n2;}
		    else{
		      mom_list[mom_num][(i+2)%4] = -n2;
		      mom_list[num_mom-mom_num-1][(i+2)%4] = n2;}
		    if(j3==0) {
		      mom_list[mom_num][(i+3)%4] = n3;
		      mom_list[num_mom-mom_num-1][(i+3)%4] = -n3;}
		    else{
		      mom_list[mom_num][(i+3)%4] = -n3;
		      mom_list[num_mom-mom_num-1][(i+3)%4] = n3;}
		    mom_num += 1;
		  }
      }

	  

    Complex myzero=cmplx(Real(0.),Real(0.));
    multi1d<LatticeComplex> meow;
    meow.resize(Nc*Nc-1);
    multi1d<int> orig;
    orig.resize(4);
    orig = 0;

    multi1d<DComplex> result1d(1);

    Real p_mu_ao2;
    Real mp_mu_ao2;
    Complex phase1;
    Complex phase2;
    LatticeComplex phasep;
    LatticeComplex phasem;
    const Real pi = 3.14159265359;
    multi1d<ColorMatrix> tSU3;
    tSU3.resize(Nc*Nc-1);
    constructSU3generators(tSU3);

    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml, 4);
    push(xml_array, "polarization");
    for(int nu=0; nu < Nd; ++nu)
      {
	push(xml_array);
	write(xml_array, "mu", nu);
	for(int c = 0; c < Nc*Nc-1; ++c)
	  {
	    meow[c] = Real(2.)*traceColor(tSU3[c]*(u[nu]-adj(u[nu])));
	  }
	XMLArrayWriter xml_sink_mom(xml,num_mom/2);
	push(xml_sink_mom, "momenta");
	for(int mom_num = 0; mom_num < num_mom/2; ++mom_num)
	  {
	    phasep = singlePhase(orig, mom_list[mom_num]);
	    phasem = singlePhase(orig, mom_list[num_mom-1-mom_num]);
            p_mu_ao2 = pi * Real(mom_list[mom_num][nu]) / Layout::lattSize()[nu];
            mp_mu_ao2 = pi * Real(mom_list[num_mom-1-mom_num][nu]) / Layout::lattSize()[nu];

	    /*phasep = singlePhase(orig, mom_list[nu][mom_num]);
	    phasem = conj(phasep);
	    p_mu_ao2 = pi * Real(mom_list[nu][mom_num][nu]) / Layout::lattSize()[nu];
            mp_mu_ao2 = -pi * Real(mom_list[nu][mom_num][nu]) / Layout::lattSize()[nu];*/
            phase1 = cmplx(cos(p_mu_ao2), sin(p_mu_ao2));
            phase2 = cmplx(cos(mp_mu_ao2), sin(mp_mu_ao2));
	    result1d = myzero;
	    for(int c = 0; c < Nc*Nc-1; ++c)
	      {
		result1d +=  sumMulti(phasep*meow[c], sft_set)*sumMulti(phasem*meow[c], sft_set)*phase1*phase2;
	      }

	    push(xml_sink_mom);
	    write(xml_sink_mom, "sink_mom_num", mom_num);
	    write(xml_sink_mom, "sink_mom", mom_list[mom_num]);
	    write(xml_sink_mom, "my2pt", result1d);
	    pop(xml_sink_mom);
	  }
	pop(xml_sink_mom);
	pop(xml_array);
      }
    pop(xml_array);
    pop(xml);
  };

  multi1d<LatticeComplex> myGluonCDER(const multi1d<LatticeColorMatrix>& u,
				      multi2d<int>& r_list,
				      int& mu,
				      multi1d<int> momentum)
  {
    multi1d<LatticeComplex> glue;
    Complex dummy;
    Complex myzero = cmplx(Real(0.),Real(0.));
    Complex phase;
    Real phasearg;
    multi1d<ColorMatrix> tSU3;
    tSU3.resize(Nc*Nc-1);
    constructSU3generators(tSU3);
    const Real pi = 3.14159265359;
    const Real twopi = 6.283185307179586476925286;
    multi1d<int> coords;
    coords.resize(Nd);
    glue.resize(Nc*Nc-1);
    for (int c = 0; c < Nc*Nc-1; c++){
      glue[c] = Real(2.)*traceColor(tSU3[c]*(u[mu]-adj(u[mu])));
    }

    multi1d<LatticeComplex> glueCDER;
    glueCDER.resize(Nc*Nc-1);
    for (int c = 0; c < Nc*Nc-1; c++)
      for (int ix = 0; ix < Layout::lattSize()[0]; ix++)
	for (int iy = 0; iy < Layout::lattSize()[1]; iy++)
	  for (int iz = 0; iz < Layout::lattSize()[2]; iz++)
	    for (int it = 0; it < Layout::lattSize()[3]; it++){
	      dummy = myzero;
	      for (int nn = 0; nn < r_list.size1(); nn++){
		coords[0] = (ix - r_list[nn][0])%Layout::lattSize()[0];
		coords[1] = (iy - r_list[nn][1])%Layout::lattSize()[1];
		coords[2] = (iz - r_list[nn][2])%Layout::lattSize()[2];
		coords[3] = (it - r_list[nn][3])%Layout::lattSize()[3];
		phasearg = zero;
		for (int ii = 0; ii < Nd; ii++){
		  phasearg += momentum[ii]*r_list[nn][ii]*twopi/Layout::lattSize()[ii];
		}
		phase = cmplx(cos(phasearg),sin(phasearg));
		  
		dummy += peekSite(glue[c],coords)*phase;
	      }
	      coords[0] = ix;
	      coords[1] = iy;
	      coords[2] = iz;
	      coords[3] = it;
	      phasearg = pi * Real(momentum[mu]) / Layout::lattSize()[mu]; //extra phase depending on mu
	      phase = cmplx(cos(phasearg),sin(phasearg));
	      dummy *= phase;
	      pokeSite(glueCDER[c],dummy,coords);
	    }
    return glueCDER;
  };
		
	      
    
      
  void CDER(const multi1d<LatticeColorMatrix>& u,
	    multi1d<int>& nn,
	    XMLWriter& xml)

  {
    Set sft_set;
    sft_set.make(TimeSliceFunc(-1));
    int n_min = nn[0];
    int n_max = nn[1];
    int r2_max = nn[2];
    int rp2_max = nn[3];
    multi2d<int> r_list, rp_list;
    r_list = Coord_list(r2_max);
    rp_list = Coord_list(rp2_max);
    int num_mom = 0;
    for(int i = 0; i < Nd; ++i){
      int blep = 1;
      for(int j = 0; j < Nd; ++j)
        if (j!=i){
          blep *= 2*(n_max-n_min+1);}
      num_mom += blep;
      }
    multi2d<int> mom_list;
    mom_list.resize(num_mom, Nd) ;
    mom_list = 0;

    int mom_num = 0;
    for(int i = 0; i < Nd; ++i)
      {
        mom_list[mom_num][i] = 0;
        for(int n1 = n_min; n1 <= n_max; ++n1)
          for(int n2 = n_min; n2 <= n_max; ++n2)
            for(int j2 = 0; j2 < 2; ++j2)
              for(int n3 = n_min; n3 <= n_max; ++n3)
                for(int j3 = 0; j3 < 2; ++j3)
		  {
                    mom_list[mom_num][(i+1)%4] = n1;
                    mom_list[num_mom-mom_num-1][(i+1)%4] = -n1;
                    if(j2==0) {
                      mom_list[mom_num][(i+2)%4] = n2;
                      mom_list[num_mom-mom_num-1][(i+2)%4] = -n2;}
                    else{
                      mom_list[mom_num][(i+2)%4] = -n2;
                      mom_list[num_mom-mom_num-1][(i+2)%4] = n2;}
                    if(j3==0) {
                      mom_list[mom_num][(i+3)%4] = n3;
                      mom_list[num_mom-mom_num-1][(i+3)%4] = -n3;}
                    else{
                      mom_list[mom_num][(i+3)%4] = -n3;
                      mom_list[num_mom-mom_num-1][(i+3)%4] = n3;}
                    mom_num += 1;
                  }
      }

    multi1d<LatticeComplex> glueCDER1;
    multi1d<LatticeComplex> glueCDER2;
    multi3d<LatticeComplex> F;
    F.resize(Nd,Nd,Nc*Nc-1);
    multi2d<LatticeComplex> Ot; //O_mu nu
    Ot.resize(Nd,Nd);
    multi1d<LatticeComplex> Ops;
    Ops.resize(9);
    F = GetChromaFmunu(u);
    Complex myzero = cmplx(Real(0.),Real(0.));
    
    for(int mu = 0; mu < Nd; ++mu)
      for(int nu = 0; nu < Nd; ++nu)
	{
	  Ot[mu][nu] = myzero;
	  for(int alpha = 0; alpha < Nd; ++alpha)
	    {
	      for(int c = 0; c < Nc*Nc-1; ++c)
		{
		  Ot[mu][nu] += F[mu][alpha][c]*F[nu][alpha][c];
		}
	    }
	}
    Ops[0] = Real(1./2.)*(Ot[0][0]+Ot[1][1]-Ot[2][2]-Ot[3][3]);
    Ops[1] = Real(1./sqrt(2.))*(Ot[2][2]-Ot[3][3]);
    Ops[2] = Real(1./sqrt(2.))*(Ot[0][0]-Ot[1][1]);
    Ops[3] = Real(1./sqrt(2.))*(Ot[0][1]+Ot[1][0]);
    Ops[4] = Real(1./sqrt(2.))*(Ot[0][2]+Ot[2][0]);
    Ops[5] = Real(1./sqrt(2.))*(Ot[0][3]+Ot[3][0]);
    Ops[6] = Real(1./sqrt(2.))*(Ot[1][2]+Ot[2][1]);
    Ops[7] = Real(1./sqrt(2.))*(Ot[1][3]+Ot[3][1]);
    Ops[8] = Real(1./sqrt(2.))*(Ot[2][3]+Ot[3][2]);
    multi1d<DComplex> result1d;
    result1d.resize(1);
    LatticeComplex dummy1;
    LatticeComplex dummy2;
    multi1d<int> indeces;
    indeces.resize(2);
    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml, 4*9);
    push(xml_array, "polarization");
    for (int rho=0; rho < Nd; ++rho)
      for (int opnum=0; opnum < 9; ++opnum)
	{
	  push(xml_array);
	  indeces[0] = rho;
	  indeces[1] = opnum;
	  write(xml_array, "rho opnum", indeces);
	  XMLArrayWriter xml_sink_mom(xml,num_mom/2);
	  push(xml_sink_mom, "momenta");
	  for(int mom_num = 0; mom_num < num_mom/2; ++mom_num){
	    glueCDER1 = myGluonCDER(u,r_list,rho,mom_list[mom_num]);
	    glueCDER2 = myGluonCDER(u,rp_list,rho,mom_list[num_mom-1-mom_num]);
	    result1d = myzero;
	    for(int c = 0; c < Nc*Nc-1; ++c)
	      {
		dummy1 = glueCDER1[c]*glueCDER2[c];
		dummy2 = dummy1*Ops[opnum];
		result1d +=  sumMulti(dummy2, sft_set);
	      } 

	    push(xml_sink_mom);
	    write(xml_sink_mom, "sink_mom_num", mom_num);
	    write(xml_sink_mom, "sink_mom", mom_list[mom_num]);
	    write(xml_sink_mom, "my3pt", result1d);
	    pop(xml_sink_mom);
	  }
	  pop(xml_sink_mom);
	  pop(xml_array);
	}
    pop(xml_array);
    pop(xml);
  };
    
    
  
  void GluonPropagator(const multi1d<LatticeColorMatrix>& u,
                       const SftMom& phases,
                       XMLWriter& xml)

  {

    Complex myzero=cmplx(Real(0.),Real(0.));
    LatticeColorMatrix A_g;
    LatticeComplex meow;

    multi2d<multi2d<DComplex>> A_ft;
    A_ft.resize(Nd,Nc*Nc-1);
    DComplex result;

    Real p_mu_ao2;
    Real mp_mu_ao2;
    Complex phase1;
    Complex phase2;
    const Real pi = 3.14159265359;
    multi1d<ColorMatrix> tSU3;
    tSU3.resize(Nc*Nc-1);
    constructSU3generators(tSU3);

    push(xml, "Measurement");
    for(int nu=0; nu < Nd; ++nu)
      {
	A_g = myzero;
	A_g = u[nu]-adj(u[nu]);// - traceColor(u[nu]-adj(u[nu]))/Real(3.);
        for(int c = 0; c < Nc*Nc-1; ++c)
          {
	    meow = myzero;
	    //meow = Real(1.)/cmplx(Real(0.),Real(2.))*Real(2.)*traceColor(tSU3[c]*(u[nu]-adj(u[nu])));
	    meow = traceColor(tSU3[c]*(u[nu]-adj(u[nu])));
	    A_ft[nu][c].resize(phases.numMom(),1);
	    A_ft[nu][c] = myzero;
	    //A_ft[nu][c] = phases.sft(meow);
	    A_ft[nu][c] = FFT4d(meow,phases);
          }
      }
    XMLArrayWriter xml_array(xml, 4);
    push(xml_array, "polarization");
    for(int nu = 0; nu < Nd; ++nu)
      {
      push(xml_array);
      write(xml_array, "mu", nu);
      XMLArrayWriter xml_sink_mom(xml,phases.numMom()/2);
      push(xml_sink_mom, "momenta");
      for(int mom_num = 0; mom_num < phases.numMom()/2; ++mom_num)
	{
	  p_mu_ao2 = pi * Real(phases.numToMom(mom_num)[nu]) / Layout::lattSize()[nu];
	  mp_mu_ao2 = pi * Real(phases.numToMom(phases.numMom()-mom_num-1)[nu]) / Layout::lattSize()[nu];
	  phase1 = cmplx(cos(p_mu_ao2), sin(p_mu_ao2));
	  phase2 = cmplx(cos(mp_mu_ao2), sin(mp_mu_ao2));
	  result = myzero;
	  for(int c = 0; c < Nc*Nc-1; ++c)
	    {
	      //result += Real(0.5)*A_ft[nu][c][mom_num][0]*A_ft[nu][c][phases.numMom()-mom_num-1][0]*phase1*phase2;
	      result += A_ft[nu][c][mom_num][0]*A_ft[nu][c][phases.numMom()-mom_num-1][0]*phase1*phase2;
	    }
	  push(xml_sink_mom);
	  write(xml_sink_mom, "sink_mom_num", mom_num);
	  write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
	  write(xml_sink_mom, "my2pt", result);
	  pop(xml_sink_mom);
	}
      pop(xml_sink_mom);
      pop(xml_array);
      }
    pop(xml_array);
    pop(xml);
  };

  void twist4(const multi1d<LatticeColorMatrix> &u,const SftMom &phases,XMLWriter &xml)
  {
    Complex myzero=cmplx(Real(0.),Real(0.));
    multi2d<LatticeColorMatrix> F;
    F.resize(Nd,Nd);
    F = GetChromaFmunu_matrix(u);
    
    LatticeComplex Ot;
    multi2d<DComplex> Ops_FT;
    XMLArrayWriter xml_array(xml,4);
    push(xml, "Measurement");
    push(xml_array, "TwistFourGluon");

    for (int mu=0; mu<Nd; mu++)
      {
	Ot = myzero;
	for (int nu=0; nu<Nd; nu++)
	for (int alpha = 0; alpha<Nd; alpha++)
	  {
	    Ot += 0.25*traceColor(F[nu][mu]*u[alpha]*shift(F[alpha][nu],FORWARD,alpha)*adj(u[alpha])-F[nu][mu]*shift(adj(u[alpha])*F[alpha][nu]*u[alpha],BACKWARD,alpha));
	    Ot -= 0.25*traceColor(u[alpha]*shift(F[nu][mu],FORWARD,alpha)*adj(u[alpha])*F[alpha][nu] - shift(adj(u[alpha])*F[nu][mu]*u[alpha],BACKWARD,alpha)*F[alpha][nu]);
	  }

	push(xml_array);
	write(xml_array, "mu",mu);
	Ops_FT = phases.sft(Ot);
	XMLArrayWriter xml_sink_mom(xml,phases.numMom());
	push(xml_sink_mom, "momenta");
	for(int mom_num = 0; mom_num < phases.numMom(); mom_num++)
	  {
	    push(xml_sink_mom);
	    write(xml_sink_mom, "sink_mom_num", mom_num);
	    write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
	    write(xml_sink_mom, "operator", Ops_FT[mom_num]);
	    pop(xml_sink_mom);
	  }
	  
	pop(xml_sink_mom);
	pop(xml_array);
      }

      pop(xml_array);
      pop(xml);
  };
  
  
  void measure_2link_glue(const multi1d<LatticeColorMatrix> &u,const SftMom &phases,XMLWriter &xml)
  {
    multi1d<LatticeColorMatrix> ff;
    multi2d<LatticeColorMatrix> F;
    Complex myzero=cmplx(Real(0.),Real(0.));
    F.resize(Nd,Nd);
    ff.resize(Nd*(Nd-1)/2);
    mesField(ff,u);
    int count = 0;
        for(int mu = 0; mu < Nd-1; mu++)
      {
	for(int nu = mu+1; nu < Nd; nu++)
	  {
	    F[mu][nu] = ff[count];
	    F[nu][mu] = -ff[count];
	    ++count;
	  }
      }

    multi5d<LatticeComplex>Ota;
    Ota.resize(Nd,Nd,Nd,Nd,Nd);
    LatticeComplex Ot;
    multi2d<DComplex> Ops_FT;
    LatticeColorMatrix dummy;
    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml,8*8*4*4);
    push(xml_array, "TwoLinkGluon");
    for (int mu2_fwd=0; mu2_fwd<=1; mu2_fwd++)
    for (int mu1_fwd=0; mu1_fwd<=1; mu1_fwd++)
      {
	for(int nu = 0; nu < Nd; nu ++)
	  {
	    for (int mu2=0; mu2 < Nd; mu2++)
	      {
		for (int alpha=0;alpha <Nd; alpha++)
		  {
		    if (mu2_fwd)
		      dummy = u[mu2]*shift(F[nu][alpha],FORWARD,mu2)*adj(u[mu2]);
		    else
		      dummy = shift(adj(u[mu2])*F[nu][alpha]*u[mu2],BACKWARD,mu2);
		    for (int mu = 0; mu < Nd; mu ++)
		      {
			for (int mu1=0; mu1 < Nd; mu1++)
			  {
			    Ota[mu][nu][mu1][mu2][alpha] = myzero;
			    //if (!( (mu1_fwd != mu2_fwd) && (mu1 == mu2) ))
			    // {
				if (mu1_fwd)
				  Ota[mu][nu][mu1][mu2][alpha] = trace(adj(u[mu1])*F[mu][alpha]*u[mu1]*shift(dummy,FORWARD,mu1));
				else
				  Ota[mu][nu][mu1][mu2][alpha] = trace(shift(u[mu1],BACKWARD,mu1)*F[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*shift(dummy,BACKWARD,mu1));
			    // }
			  }
		      }
		  }
	      }
	  }
	
	for (int mu=0; mu < Nd; mu++)
	for (int nu=0; nu <Nd; nu++)
	for (int mu1=0; mu1 <Nd; mu1++)
	for (int mu2=0; mu2 <Nd; mu2++)
	  //if (!( (mu1_fwd != mu2_fwd) && (mu1 == mu2) ))
	  {
	    Ot = myzero;
	    for (int alpha = 0; alpha < Nd; alpha++)
	      {
		Ot += Ota[mu][nu][mu1][mu2][alpha];
	      }
	    push(xml_array);
	    write(xml_array, "mu", mu);
	    write(xml_array, "nu", nu);
	    write(xml_array, "mu1", mu1);
	    if (mu1_fwd)
	      write(xml_array, "mu1_dir", "FORWARD");
	    else
	      write(xml_array, "mu1_dir", "BACKWARD");
	    write(xml_array, "mu2", mu2);
	    if (mu2_fwd)
	      write(xml_array, "mu2_dir", "FORWARD");
	    else
	      write(xml_array, "mu2_dir", "BACKWARD");
	    Ops_FT = phases.sft(Ot);
	    XMLArrayWriter xml_sink_mom(xml, (phases.numMom()-1)/2+1);
	    push(xml_sink_mom, "momenta");
	    for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
	      {
		push(xml_sink_mom);
		write(xml_sink_mom, "sink_mom_num", mom_num);
		write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
		write(xml_sink_mom, "operator", Ops_FT[mom_num]);
		pop(xml_sink_mom);
	      }
	    pop(xml_sink_mom);
	    pop(xml_array);
	  }

      }
    pop(xml_array);
    pop(xml);
  };

  void sec_mom_unprojected(const multi1d<LatticeColorMatrix>& u,
                        const SftMom& phases,
                        XMLWriter& xml)

  {
    multi5d<LatticeComplex> Ot2;
    Ot2.resize(Nd,Nd,Nd,Nd,Nd);
    multi2d<DComplex> Ops_FT;

    Complex myzero=cmplx(Real(0.),Real(0.));

    multi1d<LatticeColorMatrix> f;
    multi2d<LatticeColorMatrix> ff;
    ff.resize(Nd,Nd);
    f.resize(Nd*(Nd-1)/2);
    mesField(f,u);
    int count = 0;
    ff = myzero;
    for(int mu = 0; mu < Nd-1; mu++)
      {
        for(int nu = mu+1; nu < Nd; nu++)
          {
            ff[mu][nu] = f[count];
            ff[nu][mu] = -f[count];
            ++count;
          }
      }
    LatticeColorMatrix tempf;
    LatticeColorMatrix tempu;
    LatticeColorMatrix tempu2;

    for(int mu = 0; mu < Nd; mu++)
      {
        for(int nu= 0; nu <  Nd; nu++)
          {
            for(int mu1 = 0; mu1 < Nd; mu1++)
              {
                for(int mu2 = 0; mu2 < Nd; mu2++)
                  {
                    Ot2[0][mu][nu][mu1][mu2] = myzero;
		    Ot2[1][mu][nu][mu1][mu2] = myzero;
		    Ot2[2][mu][nu][mu1][mu2] = myzero;
		    Ot2[3][mu][nu][mu1][mu2] = myzero;
                    for(int alpha = 0; alpha < Nd; alpha++)
                      {
                        // RIGHT * RIGHT DERIVATIVE                                                                                                                                                     

                        tempf = shift(ff[nu][alpha],FORWARD,mu2);
                        Ot2[0][mu][nu][mu1][mu2] += traceColor(ff[mu][alpha]*u[mu1]*shift(u[mu2],FORWARD,mu1)*shift(tempf,FORWARD,mu1)*adj(shift(u[mu2],FORWARD,mu1))*adj(u[mu1]));

                        Ot2[0][mu][nu][mu1][mu2] -= traceColor(ff[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*shift(u[mu2],BACKWARD,mu1)*shift(tempf,BACKWARD,mu1)*adj(shift(u[mu2],BACKWARD,mu1))*shift(u[mu1],BACKWARD,mu1));
                        tempf = shift(ff[nu][alpha],BACKWARD,mu2);
                        tempu = shift(u[mu2],BACKWARD,mu2);
                        Ot2[0][mu][nu][mu1][mu2] -= traceColor(ff[mu][alpha]*u[mu1]*adj(shift(tempu,FORWARD,mu1))*shift(tempf,FORWARD,mu1)*shift(tempu,FORWARD,mu1)*adj(u[mu1]));

                        Ot2[0][mu][nu][mu1][mu2] += traceColor(ff[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*adj(shift(tempu,BACKWARD,mu1))*shift(tempf,BACKWARD,mu1)*shift(tempu,BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1));
			// RIGHT * LEFT DERIVATIVE                                                                                                                                                      

                        tempu = shift(u[mu1],BACKWARD,mu1);
                        Ot2[1][mu][nu][mu1][mu2] -= traceColor(adj(shift(u[mu1],FORWARD,mu2))*shift(ff[mu][alpha],FORWARD,mu2)*shift(u[mu1],FORWARD,mu2)*adj(shift(u[mu2],FORWARD,mu1))*shift(ff[nu][alpha],FORWARD,mu1)*shift(u[mu2],FORWARD,mu1));

                        Ot2[1][mu][nu][mu1][mu2] += traceColor(shift(tempu,FORWARD,mu2)*shift(ff[mu][alpha],FORWARD,mu2)*adj(shift(tempu,FORWARD,mu2))*adj(shift(u[mu2],BACKWARD,mu1))*shift(ff[nu][alpha],BACKWARD,mu1)*shift(u[mu2],BACKWARD,mu1));

                        tempu2 = shift(u[mu2],BACKWARD,mu2);
                        Ot2[1][mu][nu][mu1][mu2] += traceColor(adj(shift(u[mu1],BACKWARD,mu2))*shift(ff[mu][alpha],BACKWARD,mu2)*shift(u[mu1],BACKWARD,mu2)*shift(tempu2,FORWARD,mu1)*shift(ff[nu][alpha],FORWARD,mu1)*adj(shift(tempu2,FORWARD,mu1)));

                        Ot2[1][mu][nu][mu1][mu2] -= traceColor(shift(tempu,BACKWARD,mu2)*shift(ff[mu][alpha],BACKWARD,mu2)*adj(shift(tempu,BACKWARD,mu2))*shift(tempu2,BACKWARD,mu1)*shift(ff[nu][alpha],BACKWARD,mu1)*adj(shift(tempu2,BACKWARD,mu1)));

                        // LEFT * RIGHT DERIVATIVE                                                                                                                                                      

                        Ot2[2][mu][nu][mu1][mu2] -= traceColor(u[mu1]*shift(ff[mu][alpha],FORWARD,mu1)*adj(u[mu1])*u[mu2]*shift(ff[nu][alpha],FORWARD,mu2)*adj(u[mu2]));

                        Ot2[2][mu][nu][mu1][mu2] += traceColor(u[mu1]*shift(ff[mu][alpha],FORWARD,mu1)*adj(u[mu1])*adj(shift(u[mu2],BACKWARD,mu2))*shift(ff[nu][alpha],BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2));

                        Ot2[2][mu][nu][mu1][mu2] += traceColor(adj(shift(u[mu1],BACKWARD,mu1))*shift(ff[mu][alpha],BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1)*u[mu2]*shift(ff[nu][alpha],FORWARD,mu2)*adj(u[mu2]));

                        Ot2[2][mu][nu][mu1][mu2] -= traceColor(adj(shift(u[mu1],BACKWARD,mu1))*shift(ff[mu][alpha],BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1)*adj(shift(u[mu2],BACKWARD,mu2))*shift(ff[nu][alpha],BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2));

                        // LEFT * LEFT DERIVATIVE                                                                                                                                                       

                        tempf = shift(ff[mu][alpha],FORWARD,mu1);
                        Ot2[3][mu][nu][mu1][mu2] += traceColor(u[mu2]*shift(u[mu1],FORWARD,mu2)*shift(tempf,FORWARD,mu2)*adj(shift(u[mu1],FORWARD,mu2))*adj(u[mu2])*ff[nu][alpha]);

                        Ot2[3][mu][nu][mu1][mu2] -= traceColor(adj(shift(u[mu2],BACKWARD,mu2))*shift(u[mu1],BACKWARD,mu2)*shift(tempf,BACKWARD,mu2)*adj(shift(u[mu1],BACKWARD,mu2))*shift(u[mu2],BACKWARD,mu2)*ff[nu][alpha]);

                        tempf = shift(ff[mu][alpha],BACKWARD,mu1);
                        tempu = shift(u[mu1],BACKWARD,mu1);

                        Ot2[3][mu][nu][mu1][mu2] -= traceColor(u[mu2]*adj(shift(tempu,FORWARD,mu2))*shift(tempf,FORWARD,mu2)*shift(tempu,FORWARD,mu2)*adj(u[mu2])*ff[nu][alpha]);

                        Ot2[3][mu][nu][mu1][mu2] += traceColor(adj(shift(u[mu2],BACKWARD,mu2))*adj(shift(tempu,BACKWARD,mu2))*shift(tempf,BACKWARD,mu2)*shift(tempu,BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2)*ff[nu][alpha]);
			//                        Ot2[mu][nu][mu1][mu2] *= Real(1./16.);*/

			}
                  }
              }
          }
      }
    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml, 4*4*4*4*4);
    push(xml_array, "Operators");
    for(int term = 0; term < Nd; term++)
    for(int mu = 0; mu < Nd; mu++)
    for(int nu = 0; nu < Nd; nu++)
    for(int mu1 = 0; mu1 < Nd; mu1++)
    for(int mu2 = 0; mu2 < Nd; mu2++)
      {
        Ops_FT = phases.sft(Ot2[term][mu][nu][mu1][mu2]);
        push(xml_array);
	if (term == 0)
	  write(xml_array,"FWD-FWD",term);
	if (term == 1)
	  write(xml_array,"FWD-BWD",term);
	if (term == 2)
	  write(xml_array,"BWD-FWD",term);
	if (term == 3)
	  write(xml_array,"BWD-BWD",term);
	      
        write(xml_array, "mu", mu);
	write(xml_array, "nu", nu);
	write(xml_array, "mu1", mu1);
	write(xml_array, "mu2", mu2);
        XMLArrayWriter xml_sink_mom(xml, (phases.numMom()-1)/2+1);
        push(xml_sink_mom, "momenta");

        for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
          {
            push(xml_sink_mom);
            write(xml_sink_mom, "sink_mom_num", mom_num);
            write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
            write(xml_sink_mom, "operator", Ops_FT[mom_num]);
            pop(xml_sink_mom);

          }
        pop(xml_sink_mom);
        pop(xml_array);

      }

    pop(xml_array);
    pop(xml);
  };

  /*void measureOperators(const multi1d<LatticeColorMatrix>& u,
                        const SftMom& phases,
                        XMLWriter& xml)

  {
    Complex myzero=cmplx(Real(0.),Real(0.));
    multi1d<LatticeComplex> Ops;
    Ops.resize(9);
    multi3d<LatticeComplex> F;
    F.resize(Nd,Nd,Nc*Nc-1);
    multi2d<LatticeComplex> Ot; //O_mu nu                                                                                              
    Ot.resize(Nd,Nd);
    F = GetChromaFmunu(u);
    for(int mu = 0; mu < Nd; ++mu)
      {
        for(int nu = 0; nu < Nd; ++nu)
          {
            Ot[mu][nu] = myzero;
            for(int alpha = 0; alpha < Nd; ++alpha)
              {
                for(int c = 0; c < Nc*Nc-1; ++c)
                  {
                    Ot[mu][nu] += F[mu][alpha][c]*F[nu][alpha][c];
                  }
              }
          }
      }
    Ops[0] = Real(1./2.)*(Ot[0][0]+Ot[1][1]-Ot[2][2]-Ot[3][3]);
    Ops[1] = Real(1./sqrt(2.))*(Ot[2][2]-Ot[3][3]);
    Ops[2] = Real(1./sqrt(2.))*(Ot[0][0]-Ot[1][1]);
    Ops[3] = Real(1./sqrt(2.))*(Ot[0][1]+Ot[1][0]);
    Ops[4] = Real(1./sqrt(2.))*(Ot[0][2]+Ot[2][0]);
    Ops[5] = Real(1./sqrt(2.))*(Ot[0][3]+Ot[3][0]);
    Ops[6] = Real(1./sqrt(2.))*(Ot[1][2]+Ot[2][1]);
    Ops[7] = Real(1./sqrt(2.))*(Ot[1][3]+Ot[3][1]);
    Ops[8] = Real(1./sqrt(2.))*(Ot[2][3]+Ot[3][2]);

    push(xml, "Measurement");
    std::vector<fftw_complex *> in_Ops_FT,out_Ops_FT;
    in_Ops_FT.resize(9);
    out_Ops_FT.resize(9);
    int fftdims[4];
    int fftlen=1;
    for(int i=0;i<Nd;i++)
      {
	fftdims[i] = nrow[i];
	fftlen *= nrow[i];
      }
    for(int ii = 0; ii < 9; ii++)
      {
	in_Ops_FT.at(ii) = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftlen);
	out_Ops_FT.at(ii) = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftlen);

	multi1d<int> fftsite(Nd); fftsite=0;
	for(int x=0;x<nrow[0]; x++)
	  for(int y=0;y<nrow[1]; y++)
	    for(int z=0;z<nrow[2]; z++)
	      for(int t=0;t<nrow[3]; t++)
		{
		  fftsite[0]=x; fftsite[1]=y; fftsite[2]=z; fftsite[3]=t;
		  Double val=peekSite(Ops[ii],fftsite);
		  in_Ops_FT.at(ii)[t + nrow[3] * (z + nrow[2] * (y +nrow[1] *x))][0]=Real(val);
		  in_Ops_FT.at(ii)[t + nrow[3] * (z + nrow[2] * (y +nrow[1] *x))][1]=Imag(val);
		}
	int fftsign=FFTW_FORWARD;
	int fftflags=FFTW_ESTIMATE;
	fftw_plan fftplan=fftw_plan_dft(4, fftdims, in_Ops_FT.at(l), out_Ops_FT.at(l), fftsign,fftflags);
	fftw_execute(fftplan);
	fftw_destroy_plan(fftplan);
      }
    
    XMLArrayWriter xml_array(xml, 9);
    push(xml_array, "Operators");
    for(int ii = 0; ii < 9; ii++)
      {
        push(xml_array);
        write(xml_array, "Op_No", ii);
        XMLArrayWriter xml_sink_mom(xml, fftlen);
        push(xml_sink_mom, "momenta");

        for(int mom_num = 0; mom_num < fftlen; mom_num++)
          {
            push(xml_sink_mom);
            write(xml_sink_mom, "sink_mom_num", mom_num);
            write(xml_sink_mom, "operator", out_Ops_FT.at(ii)[mom_num]);
            pop(xml_sink_mom);

          }
        pop(xml_sink_mom);
        pop(xml_array);
	}

    pop(xml_array);
    pop(xml);
    }; */

  void measureOperators(const multi1d<LatticeColorMatrix>& u, // rename to measureOperators after you're done
			const SftMom& phases,
			XMLWriter& xml)

  {
    Complex myzero=cmplx(Real(0.),Real(0.));
    multi1d<LatticeComplex> Ops;
    Ops.resize(9);
    multi2d<DComplex> Ops_FT;
    multi2d<LatticeComplex> Ot; //O_mu nu
    Ot.resize(Nd,Nd);
    //multi3d<LatticeComplex> F;
    //F.resize(Nd,Nd,Nc*Nc-1);
    //F = GetChromaFmunu(u);
    multi2d<LatticeColorMatrix> F;
    F.resize(Nd,Nd);
    F = GetChromaFmunu_matrix(u);
    for(int mu = 0; mu < Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    Ot[mu][nu] = myzero;
	    for(int alpha = 0; alpha < Nd; ++alpha)
	      {
		Ot[mu][nu] += 2*traceColor(F[mu][alpha]*F[nu][alpha]);
		//for(int c = 0; c < Nc*Nc-1; ++c)
		//  {
		//    Ot[mu][nu] += F[mu][alpha][c]*F[nu][alpha][c];
		//  }
	      }
	  }
      }

    Ops[0] = Ot[3][3];
    Ops[1] = Ot[0][3];
    //Ops[0] = Real(1./2.)*(Ot[0][0]+Ot[1][1]-Ot[2][2]-Ot[3][3]);         //hypercubic ops
    //Ops[1] = Real(1./sqrt(2.))*(Ot[2][2]-Ot[3][3]);
    Ops[2] = Real(1./sqrt(2.))*(Ot[0][0]-Ot[1][1]);
    Ops[3] = Real(1./sqrt(2.))*(Ot[0][1]+Ot[1][0]);
    Ops[4] = Real(1./sqrt(2.))*(Ot[0][2]+Ot[2][0]);
    Ops[5] = Real(1./sqrt(2.))*(Ot[0][3]+Ot[3][0]);
    Ops[6] = Real(1./sqrt(2.))*(Ot[1][2]+Ot[2][1]);
    Ops[7] = Real(1./sqrt(2.))*(Ot[1][3]+Ot[3][1]);
    Ops[8] = Real(1./sqrt(2.))*(Ot[2][3]+Ot[3][2]);
    
    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml, 9);
    push(xml_array, "Operators");
    for(int ii = 0; ii < 9; ii++)
      {
	Ops_FT = phases.sft(Ops[ii]);
	push(xml_array);
	write(xml_array, "Op_No", ii);
	XMLArrayWriter xml_sink_mom(xml, (phases.numMom()-1)/2+1);
	push(xml_sink_mom, "momenta");

	for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
	  {
	    push(xml_sink_mom);
	    write(xml_sink_mom, "sink_mom_num", mom_num);
	    write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
	    write(xml_sink_mom, "operator", Ops_FT[mom_num]);
	    pop(xml_sink_mom);

	  }
	pop(xml_sink_mom);
	pop(xml_array);

      }

    pop(xml_array);
    pop(xml); 
  };

  multi2d<LatticeColorMatrix> GetChromaFmunu_matrix(const multi1d<LatticeColorMatrix>& u)
  {
    multi1d<LatticeColorMatrix> f;
    f.resize(Nd*(Nd-1)/2);

    mesField(f,u);
    Complex myzero=cmplx(Real(0.),Real(0.));

    multi2d<LatticeColorMatrix> Fmunu;
    Fmunu.resize(Nd,Nd);

    for(int mu = 0; mu < Nd; ++mu)
      for(int nu = 0; nu < Nd; ++nu)
	Fmunu[mu][nu] = myzero;

    ColorMatrix Id;
    LatticeColorMatrix LId;
    multi1d<int> site(4);
    LatticeComplex Ldummy;
    Complex dummy;
    int count = 0;
    for(int mu = 0; mu < Nd-1; ++mu)
      {
	for(int nu = mu+1; nu < Nd; ++nu)
          {
	    LId = 0;
	    Ldummy = 0;
	    Ldummy = traceColor(f[count])/3.;
	    for (int ix = 0; ix < Layout::lattSize()[0]; ++ix)
	      for (int iy = 0; iy < Layout::lattSize()[1]; ++iy)
		for (int iz = 0; iz < Layout::lattSize()[2]; ++iz)
		  for (int it = 0; it < Layout::lattSize()[3]; ++it)
		    {
		      site[0] = ix; site[1] = iy; site[2] = iz; site[3] = it;
		      Id = 0;
		      dummy = peekSite(Ldummy,site);
		      pokeColor(Id,dummy,0,0);
		      pokeColor(Id,dummy,1,1);
		      pokeColor(Id,dummy,2,2);
		      pokeSite(LId,Id,site);
		    }
	    Fmunu[mu][nu] = f[count];// - LId;
	    Fmunu[nu][mu] = -f[count];// + LId;
            ++count;
          }
      }

    //for(int mu = 0; mu < Nd-1; ++mu)
    // for(int nu = mu+1; nu < Nd; ++nu)
    //   {
    //	 Fmunu[mu][nu] = Fmunu[mu][nu]-traceColor(Fmunu[mu][nu])/3.;
    //	 Fmunu[nu][mu] = Fmunu[nu][mu] -traceColor(Fmunu[nu][mu])/3.;
    // }
    
    return Fmunu;
  };
  multi3d<LatticeComplex> GetChromaFmunu(const multi1d<LatticeColorMatrix>& u)
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

	    F[mu][c] = Real(2.)*trace(tSU3[c]*f[mu]);
	  }
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
  };

  multi2d<int> Coord_list(int & r2_max)
  {
    int L;
    int grid_vol = 1;
    multi1d<int> grid_size;
    grid_size.resize(4);
    for (L=1; L*L <= r2_max; ++L) ;

    for(int mu=0; mu < 4; ++mu) {
      grid_vol *= (2*L) + 1;
      grid_size[mu] = (2*L) + 1;
    }
    int num_grid;
    num_grid = 0;

    for(int n=0; n < grid_vol; ++n) {
      multi1d<int> grid = crtesn(n, grid_size);

      int grid2 = 0;
      for(int mu=0; mu < grid_size.size(); ++mu){
	grid2 += grid[mu]*grid[mu];
      }
      if (grid2 > r2_max) {
	continue;
      } else{
	++num_grid ;

      }
    }
    multi2d<int> grid_list;
    grid_list.resize(num_grid, 4);

    int grid_num = 0;
    for(int n=0; n < grid_vol; ++n)
      {
	multi1d<int> grid = crtesn(n, grid_size);
	int grid2 = 0;
	for(int mu=0; mu < grid_size.size(); ++mu) {
	  grid2 += grid[mu]*grid[mu];
	}
	if (grid2 > r2_max) {
	  continue;
	} else {
	  for (int mu=0; mu < grid_size.size(); ++mu) {
	    grid_list[grid_num][mu] = grid[mu];
	  }
	  ++grid_num;
	}
      }
    return grid_list;
  };
	
      



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
  
  /*  void mesField_aniso(multi1d<LatticeColorMatrix>& f,
		      const multi1d<LatticeColorMatrix>& u,
		      int& nn)   //# of temporal timelices based on a_x/a_t. if nn = 1 regular plaquette and clover same as on /mes/glue/mesfield.cc
  {
    f.resize(Nd*(Nd-1)/2);

    LatticeColorMatrix tmp_0;
    LatticeColorMatrix tmp_1;
    LatticeColorMatrix tmp_2;
    LatticeColorMatrix tmp_3;
    LatticeColorMatrix tmp_4;
    multi1d<LatticeColorMatrix> u_tadpole;
    u_tadpole.resize(4);
    //u_tadpole[0] = u[0]/Real(0.7336);   //divide by u_s tadpole param
    //u_tadpole[1] = u[1]/Real(0.7336);
    //u_tadpole[2] = u[2]/Real(0.7336);
    //u_tadpole[3] = u[3];               //u_t ~ 1 for a_s>>a_t

    u_tadpole[0] = u[0];
    u_tadpole[1] = u[1];
    u_tadpole[2] = u[2];
    u_tadpole[3] = u[3];

    Real fact = 0.125;

    int offset = 0;
    
    if (nn==1)
      {
	for(int mu=0; mu < Nd-1; ++mu)
	  {
	    for(int nu=mu+1; nu < Nd; ++nu)
	      {
		tmp_3 = shift(u_tadpole[nu], FORWARD, mu);
		tmp_4 = shift(u_tadpole[mu], FORWARD, nu);
		tmp_0 = u_tadpole[nu] * tmp_4;
		tmp_1 = u_tadpole[mu] * tmp_3;

		f[offset] = tmp_1 * adj(tmp_0);

		tmp_2 = adj(tmp_0) * tmp_1;
		tmp_1 = shift(tmp_2, BACKWARD, nu);
		f[offset] += shift(tmp_1, BACKWARD, mu);
		tmp_1 = tmp_4 * adj(tmp_3);
		tmp_0 = adj(u_tadpole[nu]) * u_tadpole[mu];

		f[offset] += shift(tmp_0*adj(tmp_1), BACKWARD, nu);
		f[offset] += shift(adj(tmp_1)*tmp_0, BACKWARD, mu);

		tmp_0 = adj(f[offset]);
		f[offset] -= tmp_0;
		f[offset] *= fact;
      		//if (nu==3)
		//  {
		//    f[offset] *= Real(2.074);   //multiply F_st by sqrt of bare gauge anisotropy 4.3
		//  }
		//else
		//  {
		//    f[offset] /= Real(2.074);   //divide F_ss' by sqrt of bare gauge anisotropy 4.3
		//  }
		++offset;
	      }
	  }
      }
    else
      {

	for(int mu=0; mu < Nd-1; ++mu)
	  {
	    for(int nu=mu+1; nu < Nd; ++nu)
	      {
		if (nu == 3)
		  {
		    tmp_3 = shift(u[nu],FORWARD,mu);
		    tmp_0 = u[mu]*tmp_3;
		    tmp_2 = u[nu];
		    tmp_1 = tmp_2;
		    tmp_4 = shift(u[mu],FORWARD,nu);
		    for(int rho=1; rho<nn; ++rho)
		      {
			tmp_3= shift(tmp_3,FORWARD,nu);
			tmp_0 = tmp_0 *tmp_3;
			tmp_2= shift(tmp_2,FORWARD,nu);
			tmp_1 = tmp_1*tmp_2;
			tmp_4 = shift(tmp_4,FORWARD,nu);
		      }
		    tmp_1 = tmp_1 * tmp_4;
		    f[offset] = tmp_0 * adj(tmp_1);
		    

		    tmp_3 = u[nu];
		    tmp_0 = tmp_3;
		    tmp_2 = shift(u[nu],BACKWARD,mu);
		    tmp_1 = tmp_2;
		    tmp_4 = shift(u[mu],BACKWARD,mu);
		    tmp_4 = shift(tmp_4,FORWARD,nu);
		    for(int rho=1; rho<nn; ++rho)
		      {
			tmp_3 = shift(tmp_3,FORWARD,nu);
			tmp_0 = tmp_0 * tmp_3;
			tmp_2 = shift(tmp_2,FORWARD,nu);
			tmp_1 = tmp_1 * tmp_2;
			tmp_4 = shift(tmp_4,FORWARD,nu);
		      }
		    tmp_1 = tmp_1 * tmp_4;
		    f[offset] += tmp_0 * adj(tmp_1)*shift(u[mu],BACKWARD,mu);

		    tmp_3 = shift(u[nu],BACKWARD,mu);
		    tmp_3 = shift(tmp_3,BACKWARD,nu);
		    tmp_0 = tmp_3 * shift(u[mu],BACKWARD,mu);
		    tmp_2 = shift(u[nu], BACKWARD, nu);
		    tmp_1 = tmp_2;
		    tmp_4 = shift(u[mu],BACKWARD,mu);
		    tmp_4 = shift(tmp_4,BACKWARD,nu);
		    for(int rho=1; rho<nn; ++rho)
		      {
			tmp_3 = shift(tmp_3, BACKWARD,nu);
			tmp_0 = tmp_3 * tmp_0;
			tmp_2 = shift(tmp_2,BACKWARD,nu);
			tmp_1 = tmp_2 * tmp_1;
			tmp_4 = shift(tmp_4,BACKWARD,nu);
			  
		      }
	     
		    f[offset] += adj(tmp_0) *tmp_4* tmp_1;

		    tmp_3 = shift(u[nu],BACKWARD,nu);
		    tmp_0 = tmp_3;
		    tmp_4 = shift(u[mu],BACKWARD,nu);
		    tmp_2 = shift(u[nu],FORWARD,mu);
		    tmp_2 = shift(tmp_2,BACKWARD,nu);
		    tmp_1 = tmp_2*adj(u[mu]);
		    for(int rho=1; rho>nn; ++rho)
		      {
			tmp_3 = shift(tmp_3,BACKWARD,nu);
			tmp_0 = tmp_3 * tmp_0;
			tmp_4 = shift(tmp_4,BACKWARD,nu);
			tmp_2 = shift(tmp_2,BACKWARD,nu);
			tmp_1 = tmp_2 * tmp_1;
		      }

		    f[offset] += adj(tmp_0) * tmp_4  * tmp_1;
		    
		    tmp_0 = adj(f[offset]);
		    f[offset] -= tmp_0;
		    f[offset] *= fact;
		    //f[offset] *= Real(3.5);
		    //f[offset] /= Real(nn);
		      
		      
		    
		  }
		 else
		  {
			tmp_3 = shift(u[nu], FORWARD, mu);
			tmp_4 = shift(u[mu], FORWARD, nu);
			tmp_0 = u[nu] * tmp_4;
			tmp_1 = u[mu] * tmp_3;

			f[offset] = tmp_1 * adj(tmp_0);

			tmp_2 = adj(tmp_0) * tmp_1;
			tmp_1 = shift(tmp_2, BACKWARD, nu);
			f[offset] += shift(tmp_1, BACKWARD, mu);
			tmp_1 = tmp_4 * adj(tmp_3);
			tmp_0 = adj(u[nu]) * u[mu];

			f[offset] += shift(tmp_0*adj(tmp_1), BACKWARD, nu);
			f[offset] += shift(adj(tmp_1)*tmp_0, BACKWARD, mu);

			tmp_0 = adj(f[offset]);
			f[offset] -= tmp_0;
			f[offset] *= fact;
		   }
		 ++offset;
	      }
	  }
      }
  }
  */

  void measure_all(const multi1d<LatticeColorMatrix> &u,const SftMom &phases,XMLWriter &xml)
  {
    multi1d<LatticeColorMatrix> ff;
    multi2d<LatticeColorMatrix> F;
    Complex myzero=cmplx(Real(0.),Real(0.));
    F.resize(Nd,Nd);
    ff.resize(Nd*(Nd-1)/2);
    mesField(ff,u);
    int count = 0;
    for(int mu = 0; mu < Nd; mu++)
      for(int nu = 0; nu < Nd; nu++)
	{
	  F[mu][nu] = myzero;
	}
    for(int mu = 0; mu < Nd-1; mu++)
      for(int nu = mu+1; nu < Nd; nu++)
	{
	  F[mu][nu] = ff[count];
	  F[nu][mu] = -ff[count];
	  ++count;
	}
    
    LatticeComplex Ot_1;
    multi2d<DComplex> Ops_FT_1;
    push(xml, "Measurement");
    XMLArrayWriter xml_array1(xml,16);
    push(xml_array1, "ZeroLinkGluon");
    for(int mu=0; mu<Nd; mu++)
    for(int nu=0; nu<Nd; nu++)
      {
	Ot_1 = myzero;
        for(int alpha=0; alpha<Nd; alpha++)
          {
            Ot_1 += 2*traceColor(F[mu][alpha]*F[nu][alpha]);
          }
	push(xml_array1);
        write(xml_array1, "mu",mu);
	write(xml_array1, "nu",nu);
	Ops_FT_1 = phases.sft(Ot_1);
	XMLArrayWriter xml_sink_mom1(xml, (phases.numMom()-1)/2+1);
	push(xml_sink_mom1, "momenta");
	for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
	  {
	    push(xml_sink_mom1);
	    write(xml_sink_mom1, "sink_mom_num", mom_num);
	    write(xml_sink_mom1, "sink_mom", phases.numToMom(mom_num));
	    write(xml_sink_mom1, "operator", Ops_FT_1[mom_num]);
	    pop(xml_sink_mom1);
	  }
	pop(xml_sink_mom1);
	pop(xml_array1);
      }
    pop(xml_array1);
	    
    /*LatticeComplex Ot4twist;
    multi2d<DComplex> Ops_FT4twist;
    XMLArrayWriter xml_array3(xml,4);
    push(xml_array3, "TwistFourGluon");
    for (int mu=0; mu<Nd; mu++)
      {
        Ot4twist = myzero;
        for (int nu=0; nu<Nd; nu++)
          for (int alpha=0; alpha<Nd; alpha++)
            {
              Ot4twist = 0.5*traceColor(F[nu][mu]*u[alpha]*shift(F[alpha][nu],FORWARD,alpha)-F[nu][mu]*shift(adj(u[alpha]),BACKWARD,alpha));
            }
        push(xml_array3);
        write(xml_array3, "mu",mu);
        Ops_FT4twist = phases.sft(Ot4twist);
        XMLArrayWriter xml_sink_mom3(xml, (phases.numMom()-1)/2+1);
        push(xml_sink_mom3, "momenta");
        for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
          {
            push(xml_sink_mom3);
            write(xml_sink_mom3, "sink_mom_num", mom_num);
            write(xml_sink_mom3, "sink_mom", phases.numToMom(mom_num));
            write(xml_sink_mom3, "operator", Ops_FT4twist[mom_num]);
            pop(xml_sink_mom3);
          }
        pop(xml_sink_mom3);
	pop(xml_array3);
      }
    pop(xml_array3);
    */

    pop(xml);

  };

}  // end namespace Chroma


/*    multi4d<LatticeComplex> Ot2;
    Ot2.resize(Nd,Nd,Nd,Nd);
    multi1d<LatticeComplex> Ops_2;
    Ops_2.resize(5);
    multi2d<DComplex> Ops_FT_2;
    LatticeColorMatrix tempf;
    LatticeColorMatrix tempu;
    LatticeColorMatrix tempu2;

    for(int mu = 0; mu < Nd; mu++)
      {
        for(int nu= 0; nu <  Nd; nu++)
          {
            for(int mu1 = 0; mu1 < Nd; mu1++)
              {
                for(int mu2 = 0; mu2 < Nd; mu2++)
                  {
                    Ot2[mu][nu][mu1][mu2] = myzero;
                    Ot2[mu][nu][mu1][mu2] = myzero;
                    Ot2[mu][nu][mu1][mu2] = myzero;
                    Ot2[mu][nu][mu1][mu2] = myzero;
                    for(int alpha = 0; alpha < Nd; alpha++)
                      {
			//RIGHT RIGHT
			
                        tempf = shift(F[nu][alpha],FORWARD,mu2);
                        Ot2[mu][nu][mu1][mu2] += traceColor(F[mu][alpha]*u[mu1]*shift(u[mu2],FORWARD,mu1)*shift(tempf,FORWARD,mu1)*adj(shift(u[mu2],FORWARD,mu1))*adj(u[mu1]));

                        Ot2[mu][nu][mu1][mu2] -= traceColor(F[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*shift(u[mu2],BACKWARD,mu1)*shift(tempf,BACKWARD,mu1)*adj(shift(u[mu2],BACKWARD,mu1))*shift(u[mu1],BACKWARD,mu1));
			tempf = shift(F[nu][alpha],BACKWARD,mu2);
                        tempu = shift(u[mu2],BACKWARD,mu2);
                        Ot2[mu][nu][mu1][mu2] -= traceColor(F[mu][alpha]*u[mu1]*adj(shift(tempu,FORWARD,mu1))*shift(tempf,FORWARD,mu1)*shift(tempu,FORWARD,mu1)*adj(u[mu1]));

                        Ot2[mu][nu][mu1][mu2] += traceColor(F[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*adj(shift(tempu,BACKWARD,mu1))*shift(tempf,BACKWARD,mu1)*shift(tempu,BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1));

			//LEFT RIGHT DERIVATIVE
			
                        tempu = shift(u[mu1],BACKWARD,mu1);
                        Ot2[mu][nu][mu1][mu2] -= traceColor(adj(shift(u[mu1],FORWARD,mu2))*shift(F[mu][alpha],FORWARD,mu2)*shift(u[mu1],FORWARD,mu2)*adj(shift(u[mu2],FORWARD,mu1))*shift(F[nu][alpha],FORWARD,mu1)*shift(u[mu2],FORWARD,mu1));

                        Ot2[mu][nu][mu1][mu2] += traceColor(shift(tempu,FORWARD,mu2)*shift(F[mu][alpha],FORWARD,mu2)*adj(shift(tempu,FORWARD,mu2))*adj(shift(u[mu2],BACKWARD,mu1))*shift(F[nu][alpha],BACKWARD,mu1)*shift(u[mu2],BACKWARD,mu1));

                        tempu2 = shift(u[mu2],BACKWARD,mu2);
                        Ot2[mu][nu][mu1][mu2] += traceColor(adj(shift(u[mu1],BACKWARD,mu2))*shift(F[mu][alpha],BACKWARD,mu2)*shift(u[mu1],BACKWARD,mu2)*shift(tempu2,FORWARD,mu1)*shift(F[nu][alpha],FORWARD,mu1)*adj(shift(tempu2,FORWARD,mu1)));

                        Ot2[mu][nu][mu1][mu2] -= traceColor(shift(tempu,BACKWARD,mu2)*shift(F[mu][alpha],BACKWARD,mu2)*adj(shift(tempu,BACKWARD,mu2))*shift(tempu2,BACKWARD,mu1)*shift(F[nu][alpha],BACKWARD,mu1)*adj(shift(tempu2,BACKWARD,mu1)));

			//RIGHT LEFT
			
                        Ot2[mu][nu][mu1][mu2] -= traceColor(u[mu1]*shift(F[mu][alpha],FORWARD,mu1)*adj(u[mu1])*u[mu2]*shift(F[nu][alpha],FORWARD,mu2)*adj(u[mu2]));

                        Ot2[mu][nu][mu1][mu2] += traceColor(u[mu1]*shift(F[mu][alpha],FORWARD,mu1)*adj(u[mu1])*adj(shift(u[mu2],BACKWARD,mu2))*shift(F[nu][alpha],BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2));
			Ot2[mu][nu][mu1][mu2] += traceColor(adj(shift(u[mu1],BACKWARD,mu1))*shift(F[mu][alpha],BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1)*u[mu2]*shift(F[nu][alpha],FORWARD,mu2)*adj(u[mu2]));

                        Ot2[mu][nu][mu1][mu2] -= traceColor(adj(shift(u[mu1],BACKWARD,mu1))*shift(F[mu][alpha],BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1)*adj(shift(u[mu2],BACKWARD,mu2))*shift(F[nu][alpha],BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2));

			//LEFT LEFT
			
                        tempf = shift(F[mu][alpha],FORWARD,mu1);
                        Ot2[mu][nu][mu1][mu2] += traceColor(u[mu2]*shift(u[mu1],FORWARD,mu2)*shift(tempf,FORWARD,mu2)*adj(shift(u[mu1],FORWARD,mu2))*adj(u[mu2])*F[nu][alpha]);

                        Ot2[mu][nu][mu1][mu2] -= traceColor(adj(shift(u[mu2],BACKWARD,mu2))*shift(u[mu1],BACKWARD,mu2)*shift(tempf,BACKWARD,mu2)*adj(shift(u[mu1],BACKWARD,mu2))*shift(u[mu2],BACKWARD,mu2)*F[nu][alpha]);

                        tempf = shift(F[mu][alpha],BACKWARD,mu1);
                        tempu = shift(u[mu1],BACKWARD,mu1);

                        Ot2[mu][nu][mu1][mu2] -= traceColor(u[mu2]*adj(shift(tempu,FORWARD,mu2))*shift(tempf,FORWARD,mu2)*shift(tempu,FORWARD,mu2)*adj(u[mu2])*F[nu][alpha]);

                        Ot2[mu][nu][mu1][mu2] += traceColor(adj(shift(u[mu2],BACKWARD,mu2))*adj(shift(tempu,BACKWARD,mu2))*shift(tempf,BACKWARD,mu2)*shift(tempu,BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2)*F[nu][alpha]);

		      }
		    Ot2[mu][nu][mu1][mu2] *= Real(1./16.);
                  }
              }
          }
      }
    Ops_2[0] = Real(1./8./sqrt(3.))*(-2.*Ot2[0][0][1][1]+Ot2[0][0][2][2]+Ot2[0][0][3][3]+Ot2[1][1][2][2]+Ot2[1][1][3][3]-2.*Ot2[2][2][3][3]);
    Ops_2[1] = Real(1./8.)*(Ot2[0][0][3][3]+Ot2[1][1][2][2]-Ot2[0][0][2][2]-Ot2[1][1][3][3]);
    Ops_2[2] = Real(1./4.)*(Ot2[0][0][1][3]+Ot2[1][2][2][3]);
    Ops_2[3] = Real(1./4.)*(Ot2[0][1][1][3]-Ot2[0][2][2][3]);
    Ops_2[4] = Real(1./4.)*(Ot2[0][0][2][3]-Ot2[1][1][2][3]);
			

    

    XMLArrayWriter xml_array2(xml,5);
    push(xml_array2, "GlueSecondMoment");
    for (int i=0; i<5; i++)
      {
	push(xml_array2);
        write(xml_array2, "Opnum", i);
	XMLArrayWriter xml_sink_mom2(xml, (phases.numMom()-1)/2+1);
	Ops_FT_2 = phases.sft(Ops_2[i]);
	push(xml_sink_mom2, "momenta");
	for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
	  {
	    push(xml_sink_mom2);
	    write(xml_sink_mom2, "sink_mom_num", mom_num);
	    write(xml_sink_mom2, "sink_mom", phases.numToMom(mom_num));
	    write(xml_sink_mom2, "operator", Ops_FT_2[mom_num]);
	    pop(xml_sink_mom2);
	  }
	pop(xml_sink_mom2);
	pop(xml_array2);
      }

    pop(xml_array2);
    
    XMLArrayWriter xml_array3(xml,4);
    push(xml_array3, "TwistFourGluon");
    for (int mu=0; mu<Nd; mu++)
      {
        Ot4twist = myzero;
        for (int nu=0; nu<Nd; nu++)
          for (int alpha=0; alpha<Nd; alpha++)
            {
              Ot4twist = 0.5*traceColor(F[nu][mu]*u[alpha]*shift(F[alpha][nu],FORWARD,alpha)-F[nu][mu]*shift(adj(u[alpha]),BACKWARD,alpha));
            }
        push(xml_array3);
        write(xml_array3, "mu",mu);
        Ops_FT4twist = phases.sft(Ot4twist);
        XMLArrayWriter xml_sink_mom3(xml, (phases.numMom()-1)/2+1);
        push(xml_sink_mom3, "momenta");
        for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; mom_num++)
          {
            push(xml_sink_mom3);
            write(xml_sink_mom3, "sink_mom_num", mom_num);
            write(xml_sink_mom3, "sink_mom", phases.numToMom(mom_num));
            write(xml_sink_mom3, "operator", Ops_FT4twist[mom_num]);
            pop(xml_sink_mom3);
          }
        pop(xml_sink_mom3);
        pop(xml_array3);
      }
    pop(xml_array3);

  };
  
}  // end namespace Chroma
*/
