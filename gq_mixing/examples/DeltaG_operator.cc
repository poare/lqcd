
#include "chromabase.h"
#include "util/ft/sftmom.h"
#include "util/info/proginfo.h"
#include "chroma.h"

#include "meas/glue/mesfield.h"

#include "meas/hadron/DeltaG_operator.h"


namespace Chroma
{


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


    /*REMEMBER TO GAUGE FIX BEFORE DOING THIS*/
  void GluonPropagator( const multi1d<LatticeColorMatrix>& u,
                         const SftMom& phases,
                         XMLWriter& xml)
  {
      multi2d<LatticeComplex> A_g;
      A_g.resize(Nd,Nc*Nc-1);
      multi2d<multi2d<DComplex>> A_ft;
      A_ft.resize(Ns,Nc*Nc-1);
      multi1d<DComplex> result1d;
      Complex result;
      Real p_mu_a_ontwo;
      Real p_nu_a_ontwo;
      Complex phase_mu;
      Complex phase_nu;
      const Real pi = 3.141592653589793238462643383279502;

      multi1d<ColorMatrix> tSU3;
      tSU3.resize(Nc*Nc-1);

      constructSU3generators(tSU3);

      push(xml, "Measurement");
      for(int mu=0; mu < Nd; ++mu)
      {
        for(int c = 0; c < Nc*Nc-1; ++c)
        {
          A_g[mu][c] = Real(2.)*traceColor(tSU3[c]*(u[mu]-adj(u[mu])))-traceColor(u[mu]-adj(u[mu])/Real(3.));
	  A_ft[mu][c] = phases.sft(A_g[mu][c]);
        }
      }
      XMLArrayWriter xml_array(xml, 16);
      push(xml_array, "polarization");
      for(int mu = 0; mu < Nd; ++mu)
      {
        for(int nu = 0; nu < Nd; ++nu)
        {
          push(xml_array);
          write(xml_array, "mu_nu", 4*mu + nu);
	  XMLArrayWriter xml_sink_mom(xml,(phases.numMom()-1)/2+1);
	  push(xml_sink_mom, "momenta");


	  for(int mom_num = 0; mom_num < (phases.numMom()-1)/2+1; ++mom_num)
	  {
	    p_mu_a_ontwo = pi * Real(phases.numToMom(mom_num)[mu]) / Layout::lattSize()[mu];
	    p_nu_a_ontwo = pi * Real(phases.numToMom(phases.numMom()-1-mom_num)[nu]) / Layout::lattSize()[nu];

	    phase_mu = cmplx(cos(p_mu_a_ontwo), sin(p_mu_a_ontwo));
	    phase_nu = cmplx(cos(p_nu_a_ontwo), sin(p_nu_a_ontwo));

            result1d =A_ft[mu][0][mom_num]*A_ft[nu][0][phases.numMom()-1-mom_num]*phase_mu*phase_nu;
            for(int c = 1; c < Nc*Nc-1; ++c)
            {
              result1d += A_ft[mu][c][mom_num]*A_ft[nu][c][phases.numMom()-1-mom_num]*phase_mu*phase_nu;
            }
	    result = result1d[0];
            push(xml_sink_mom);
            write(xml_sink_mom, "sink_mom_num", mom_num);
            write(xml_sink_mom, "sink_mom", phases.numToMom(mom_num));
            write(xml_sink_mom, "my2pt", result);
	    pop(xml_sink_mom);

	  }
          pop(xml_sink_mom);
	  pop(xml_array);
	}
      }
      pop(xml_array);
      pop(xml);
  };


  void measureOperators(const multi1d<LatticeColorMatrix>& u,
			const SftMom& phases,
			XMLWriter& xml,
			int& nn)

  {
    multi3d<LatticeComplex> F;
    F.resize(Nd,Nd,Nc*Nc-1);
    //    multi4d<LatticeComplex> Of; //O_alpha beta mu nu
    //    Of.resize(Nd,Nd,Nd,Nd);
    multi2d<LatticeComplex> Ot; //O_mu nu
    Ot.resize(Nd,Nd);
    multi4d<LatticeComplex> Ot2;
    Ot2.resize(Nd,Nd,Nd,Nd);
    multi4d<LatticeComplex> Ot2SYM;
    Ot2SYM.resize(Nd,Nd,Nd,Nd);
    multi1d<LatticeComplex> Ops;
    Ops.resize(17);
    multi2d<DComplex> Ops_FT;

    F = GetChromaFmunu(u,nn);

    Complex myzero=cmplx(Real(0.),Real(0.));

    /*    for(int mu = 0; mu <Nd; ++mu)
      {
	for(int nu = 0; nu < Nd; ++nu)
	  {
	    for(int alpha = 0; alpha < Nd; ++alpha)
	      {
		for(int beta = 0; beta < Nd; ++beta)
		  {
		    Of[mu][nu][alpha][beta] = myzero;
		    for(int c = 0; c < Nc*Nc-1; c++)
		      {
			Of[mu][nu][alpha][beta] += F[mu][alpha][c]*F[nu][beta][c];
		      }
		  }
	      }
	  }
      }
    */
    for(int mu = 0; mu < Nd; mu++)
      {
      	for(int nu = 0; nu < Nd; nu++)
      	  {
      	    Ot[mu][nu] = myzero;
      	    for(int alpha = 0; alpha < Nd; alpha++)
      	      {
            		for(int c = 0; c < Nc*Nc-1; c++)
            		  {
                    // EMT is F^{\mu\alpha} F^\nu_\alpha - 1/4 g^{\mu\nu} F^2 -- see Ji's paper.
            		    Ot[mu][nu] += F[mu][alpha][c]*F[nu][alpha][c];
            		  }
      	      }
      	  }
      }

    //basis for 2nd moment of GPD

    multi1d<LatticeColorMatrix> f;
    multi2d<LatticeColorMatrix> ff;
    ff.resize(Nd,Nd);
    f.resize(Nd*(Nd-1)/2);
    mesField(f,u);     // mesField seems to get the gauge field and store something like that in f
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
            		    Ot2[mu][mu1][mu2][nu] = myzero;
            		    for(int alpha = 0; alpha < Nd; alpha++)
            		      {
            			// RIGHT * RIGHT DERIVATIVE

            			tempf = shift(ff[nu][alpha],FORWARD,mu2);
            			Ot2[mu][mu1][mu2][nu] += traceColor(ff[mu][alpha]*u[mu1]*shift(u[mu2],FORWARD,mu1)*shift(tempf,FORWARD,mu1)*adj(shift(u[mu2],FORWARD,mu1))*adj(u[mu1]));

            			Ot2[mu][mu1][mu2][nu] -= traceColor(ff[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*shift(u[mu2],BACKWARD,mu1)*shift(tempf,BACKWARD,mu1)*adj(shift(u[mu2],BACKWARD,mu1))*shift(u[mu1],BACKWARD,mu1));
            			tempf = shift(ff[nu][alpha],BACKWARD,mu2);
            			tempu = shift(u[mu2],BACKWARD,mu2);
            			Ot2[mu][mu1][mu2][nu] -= traceColor(ff[mu][alpha]*u[mu1]*adj(shift(tempu,FORWARD,mu1))*shift(tempf,FORWARD,mu1)*shift(tempu,FORWARD,mu1)*adj(u[mu1]));

            			Ot2[mu][mu1][mu2][nu] += traceColor(ff[mu][alpha]*adj(shift(u[mu1],BACKWARD,mu1))*adj(shift(tempu,BACKWARD,mu1))*shift(tempf,BACKWARD,mu1)*shift(tempu,BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1));
            			// RIGHT * LEFT DERIVATIVE

            			tempu = shift(u[mu1],BACKWARD,mu1);
            			Ot2[mu][mu1][mu2][nu] -= traceColor(adj(shift(u[mu1],FORWARD,mu2))*shift(ff[mu][alpha],FORWARD,mu2)*shift(u[mu1],FORWARD,mu2)*adj(shift(u[mu2],FORWARD,mu1))*shift(ff[nu][alpha],FORWARD,mu1)*shift(u[mu2],FORWARD,mu1));

            			Ot2[mu][mu1][mu2][nu] += traceColor(shift(tempu,FORWARD,mu2)*shift(ff[mu][alpha],FORWARD,mu2)*adj(shift(tempu,FORWARD,mu2))*adj(shift(u[mu2],BACKWARD,mu1))*shift(ff[nu][alpha],BACKWARD,mu1)*shift(u[mu2],BACKWARD,mu1));

            			tempu2 = shift(u[mu2],BACKWARD,mu2);
            			Ot2[mu][mu1][mu2][nu] += traceColor(adj(shift(u[mu1],BACKWARD,mu2))*shift(ff[mu][alpha],BACKWARD,mu2)*shift(u[mu1],BACKWARD,mu2)*shift(tempu2,FORWARD,mu1)*shift(ff[nu][alpha],FORWARD,mu1)*adj(shift(tempu2,FORWARD,mu1)));

            			Ot2[mu][mu1][mu2][nu] -= traceColor(shift(tempu,BACKWARD,mu2)*shift(ff[mu][alpha],BACKWARD,mu2)*adj(shift(tempu,BACKWARD,mu2))*shift(tempu2,BACKWARD,mu1)*shift(ff[nu][alpha],BACKWARD,mu1)*adj(shift(tempu2,BACKWARD,mu1)));

            			// LEFT * RIGHT DERIVATIVE

            			Ot2[mu][mu1][mu2][nu] -= traceColor(u[mu1]*shift(ff[mu][alpha],FORWARD,mu1)*adj(u[mu1])*u[mu2]*shift(ff[nu][alpha],FORWARD,mu2)*adj(u[mu2]));

            			Ot2[mu][mu1][mu2][nu] += traceColor(u[mu1]*shift(ff[mu][alpha],FORWARD,mu1)*adj(u[mu1])*adj(shift(u[mu2],BACKWARD,mu2))*shift(ff[nu][alpha],BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2));

            			Ot2[mu][mu1][mu2][nu] += traceColor(adj(shift(u[mu1],BACKWARD,mu1))*shift(ff[mu][alpha],BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1)*u[mu2]*shift(ff[nu][alpha],FORWARD,mu2)*adj(u[mu2]));

            			Ot2[mu][mu1][mu2][nu] -= traceColor(adj(shift(u[mu1],BACKWARD,mu1))*shift(ff[mu][alpha],BACKWARD,mu1)*shift(u[mu1],BACKWARD,mu1)*adj(shift(u[mu2],BACKWARD,mu2))*shift(ff[nu][alpha],BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2));

            			// LEFT * LEFT DERIVATIVE

            			tempf = shift(ff[mu][alpha],FORWARD,mu1);
            			Ot2[mu][mu1][mu2][nu] += traceColor(u[mu2]*shift(u[mu1],FORWARD,mu2)*shift(tempf,FORWARD,mu2)*adj(shift(u[mu1],FORWARD,mu2))*adj(u[mu2])*ff[nu][alpha]);

            			Ot2[mu][mu1][mu2][nu] -= traceColor(adj(shift(u[mu2],BACKWARD,mu2))*shift(u[mu1],BACKWARD,mu2)*shift(tempf,BACKWARD,mu2)*adj(shift(u[mu1],BACKWARD,mu2))*shift(u[mu2],BACKWARD,mu2)*ff[nu][alpha]);

            			tempf = shift(ff[mu][alpha],BACKWARD,mu1);
            			tempu = shift(u[mu1],BACKWARD,mu1);

            			Ot2[mu][mu1][mu2][nu] -= traceColor(u[mu2]*adj(shift(tempu,FORWARD,mu2))*shift(tempf,FORWARD,mu2)*shift(tempu,FORWARD,mu2)*adj(u[mu2])*ff[nu][alpha]);

            			Ot2[mu][mu1][mu2][nu] += traceColor(adj(shift(u[mu2],BACKWARD,mu2))*adj(shift(tempu,BACKWARD,mu2))*shift(tempf,BACKWARD,mu2)*shift(tempu,BACKWARD,mu2)*shift(u[mu2],BACKWARD,mu2)*ff[nu][alpha]);
            			Ot2[mu][mu1][mu2][nu] *= Real(1./16.);

            		      }
            		  }
      	      }
      	  }
      }

    for(int mu = 0; mu < Nd; mu++)
      {
      	for(int nu = 0; nu < Nd; nu++)
      	  {
      	    for(int mu1 = 0; mu1 < Nd; mu1++)
      	      {
            		for(int mu2 = 0; mu2 < Nd; mu2++)
            		  {
            		    Ot2SYM[mu][nu][mu1][mu2] = Real(1./24.)*(Ot2[mu][mu1][mu2][nu]+Ot2[mu][mu1][nu][mu2]+Ot2[mu][nu][mu1][mu2]+Ot2[mu][nu][mu2][mu1]+Ot2[mu][mu2][mu1][nu]+Ot2[mu][mu2][nu][mu1]
            							     +Ot2[mu1][mu][mu2][nu]+Ot2[mu1][mu][nu][mu2]+Ot2[mu1][mu2][mu][nu]+Ot2[mu1][mu2][nu][mu]+Ot2[mu1][nu][mu][mu2]+Ot2[mu1][nu][mu2][mu]
            							     +Ot2[mu2][mu][nu][mu1]+Ot2[mu2][mu][mu1][nu]+Ot2[mu2][nu][mu][mu1]+Ot2[mu2][nu][mu1][mu]+Ot2[mu2][mu1][mu][nu]+Ot2[mu2][mu1][nu][mu]
            							     +Ot2[nu][mu][mu1][mu2]+Ot2[nu][mu][mu2][mu1]+Ot2[nu][mu1][mu][mu2]+Ot2[nu][mu1][mu2][mu]+Ot2[nu][mu2][mu][mu1]+Ot2[nu][mu2][mu1][mu]);

            		      }
      	      }
      	  }
      }





    /*    Ops[0] = Real(1./sqrt(6.))*(Real(2.)*Ot[0][0]-Ot[1][1]-Ot[2][2]);   //cubic ops
    Ops[1] = Real(1./sqrt(6.))*(Real(2.)*Ot[1][1]-Ot[2][2]-Ot[0][0]);
    Ops[2] = Real(1./sqrt(6.))*(Real(2.)*Ot[2][2]-Ot[0][0]-Ot[1][1]);
    Ops[3] = Real(1./sqrt(2.))*(Ot[1][1]-Ot[2][2]);
    Ops[4] = Real(1./sqrt(2.))*(Ot[2][2]-Ot[0][0]);
    Ops[5] = Real(1./sqrt(2.))*(Ot[0][0]-Ot[1][1]);
    Ops[6] = Real(1./sqrt(2.))*(Ot[0][3]+Ot[3][0]);
    Ops[7] = Real(1./sqrt(2.))*(Ot[1][3]+Ot[3][1]);
    Ops[8] = Real(1./sqrt(2.))*(Ot[2][3]+Ot[3][2]);
    */
    Ops[0] = Real(1./2.)*(Ot[0][0]+Ot[1][1]-Ot[2][2]-Ot[3][3]);         //hypercubic ops
    Ops[1] = Real(1./sqrt(2.))*(Ot[2][2]-Ot[3][3]);
    Ops[2] = Real(1./sqrt(2.))*(Ot[0][0]-Ot[1][1]);
    Ops[3] = Real(1./sqrt(2.))*(Ot[0][1]+Ot[1][0]);
    Ops[4] = Real(1./sqrt(2.))*(Ot[0][2]+Ot[2][0]);
    Ops[5] = Real(1./sqrt(2.))*(Ot[0][3]+Ot[3][0]);
    Ops[6] = Real(1./sqrt(2.))*(Ot[1][2]+Ot[2][1]);
    Ops[7] = Real(1./sqrt(2.))*(Ot[1][3]+Ot[3][1]);
    Ops[8] = Real(1./sqrt(2.))*(Ot[2][3]+Ot[3][2]);


    // second moment
    // tau_1^(2)
    Ops[9] =  Real(1./8./sqrt(3.))*(-2.*Ot2SYM[0][0][1][1]+Ot2SYM[0][0][2][2]+Ot2SYM[0][0][3][3]+Ot2SYM[1][1][2][2]+Ot2SYM[1][1][3][3]-2.*Ot2SYM[2][2][3][3]);
    Ops[10] = Real(1./8.)*(Ot2SYM[0][0][3][3]+Ot2SYM[1][1][2][2]-Ot2SYM[0][0][2][2]-Ot2SYM[1][1][3][3]);

    // tau_2^(6)
    Ops[11] = Real(1./4.)*(Ot2SYM[0][0][1][2]-Ot2SYM[1][2][3][3]);
    Ops[12] = Real(1./4.)*(Ot2SYM[0][0][1][3]+Ot2SYM[1][2][2][3]);
    Ops[13] = Real(1./4.)*(Ot2SYM[0][1][1][2]+Ot2SYM[0][2][3][3]);
    Ops[14] = Real(1./4.)*(Ot2SYM[0][1][1][3]-Ot2SYM[0][2][2][3]);
    Ops[15] = Real(1./4.)*(Ot2SYM[0][0][2][3]-Ot2SYM[1][1][2][3]);
    Ops[16] = Real(1./4.)*(Ot2SYM[0][1][2][2]-Ot2SYM[0][1][3][3]);


    push(xml, "Measurement");
    XMLArrayWriter xml_array(xml, 17);
    push(xml_array, "Operators");
    for(int ii = 0; ii < 17; ii++)
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


  multi3d<LatticeComplex> GetChromaFmunu(const multi1d<LatticeColorMatrix>& u, int& nn)
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

}  // end namespace Chroma
