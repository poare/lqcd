//  $Id: mesons2_w.cc,v 1.1 2006-07-10 19:53:37 edwards Exp $
//  $Log: mesons2_w.cc,v $
//  Revision 1.1  2006-07-10 19:53:37  edwards
//  A complex version.

#include "chromabase.h"
#include "util/ft/sftmom.h"
#include "meas/hadron/mesons_w.h"

namespace Chroma {

//! Meson 2-pt functions
/* This routine is specific to Wilson fermions!
 *
 * Construct meson propagators and writes in COMPLEX
 *
 * The two propagators can be identical or different.
 *
 * \param quark_prop_1  first quark propagator ( Read )
 * \param quark_prop_2  second (anti-) quark propagator ( Read )
 * \param k             momentum index for RI/sMOM
 * \param ferm_bc				true if we want to add bvec
 * \param xml           xml file object ( Write )
 * \param xml_group     std::string used for writing xml data ( Read )
 */

void zero_nubb(const LatticePropagator& quark_prop_k1,
	     const LatticePropagator& quark_prop_k2,
			 const LatticePropagator& quark_prop_q,
       int k,
			 bool ferm_bc,
	     XMLWriter& xml,
	     const std::string& xml_group)
{
  START_CODE();

  // Length of lattice in decay direction
  // int length = phases.numSubsets();

  // Construct phases (can't use SFT because we need fermionic bcs, and SFT only takes integer moms)

  // initialize momenta k1, k2, q
  // Problem looks like it may be in the computation of the propagator-- need to add bvec into the
  // source term when it's initialized
  multi1d<double> bvec;
  bvec.resize(4);
  bvec[0] = 0.0;
  bvec[1] = 0.0;
  bvec[2] = 0.0;
	if (ferm_bc) {
		bvec[3] = 0.5;
	} else {
		bvec[3] = 0.0;
	}

	// use these for writing k1, k2, q to files
	int vol = Layout::vol();
	multi1d<int> k1_int;
	k1_int.resize(Nd);
	k1_int[0] = -k;
	k1_int[1] = 0;
	k1_int[2] = k;
	k1_int[3] = 0;

	multi1d<int> k2_int;
	k2_int.resize(Nd);
	k2_int[0] = 0;
	k2_int[1] = k;
	k2_int[2] = k;
	k2_int[3] = 0;

	multi1d<int> q_int;
	q_int.resize(Nd);
	q_int[0] = k;
	q_int[1] = k;
	q_int[2] = 0;
	q_int[3] = 0;


  // TODO going to use the SFTMom structure to implement the momentum projection (for now! bvec is not allowed with this infrastructure)
	// Begin first method (this one makes it possible to use fermionic b.c.s) {

	multi1d<double> k1;
	multi1d<double> k2;
	multi1d<double> q;
	k1.resize(Nd);
	k2.resize(Nd);
	q.resize(Nd);
	for (int mu = 0; mu < Nd; mu++) {
		k1[mu] = (double) k1_int[mu];
		k2[mu] = (double) k2_int[mu];
		q[mu] = (double) q_int[mu];
	}

  // multi1d<double> k1;
  // k1.resize(4);
  // k1[0] = (double) -k;
  // k1[1] = 0.0;
  // k1[2] = (double) k;
  // k1[3] = 0.0;
	//
  // multi1d<double> k2;
  // k2.resize(4);
  // k2[0] = 0.0;
  // k2[1] = (double) k;
  // k2[2] = (double) k;
  // k2[3] = 0.0;
	//
  // multi1d<double> q;
  // q.resize(4);
  // q[0] = (double) k;
  // q[1] = (double) k;
  // q[2] = 0.0;
  // q[3] = 0.0;

  LatticeReal phase_k1_arg = zero;
  LatticeReal phase_k2_arg = zero;
  LatticeReal phase_q_arg = zero;
  LatticeReal phase_mq_arg = zero;  // no bvec for these-- they cancel in the momentum proj step
  LatticeReal phase_m2q_arg = zero;
  for (int mu = 0; mu < Nd; mu++) {
		phase_k1_arg -= Layout::latticeCoordinate(mu) * (k1[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
		phase_k2_arg -= Layout::latticeCoordinate(mu) * (k2[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
		phase_q_arg -= Layout::latticeCoordinate(mu) * (q[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
		phase_mq_arg += Layout::latticeCoordinate(mu) * q[mu] * twopi / Real(Layout::lattSize()[mu]);
    phase_m2q_arg += Layout::latticeCoordinate(mu) * 2 * q[mu] * twopi / Real(Layout::lattSize()[mu]);

		// phase_k1_arg += Layout::latticeCoordinate(mu) * (k1[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
		// phase_k2_arg += Layout::latticeCoordinate(mu) * (k2[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
		// phase_q_arg += Layout::latticeCoordinate(mu) * (q[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
		// phase_mq_arg -= Layout::latticeCoordinate(mu) * q[mu] * twopi / Real(Layout::lattSize()[mu]);
    // phase_m2q_arg -= Layout::latticeCoordinate(mu) * 2 * q[mu] * twopi / Real(Layout::lattSize()[mu]);
  }
  LatticeComplex phase_k1 = cmplx(cos(phase_k1_arg), sin(phase_k1_arg));
  LatticeComplex phase_k2 = cmplx(cos(phase_k2_arg), sin(phase_k2_arg));
  LatticeComplex phase_q = cmplx(cos(phase_q_arg), sin(phase_q_arg));
  LatticeComplex phase_mq = cmplx(cos(phase_mq_arg), sin(phase_mq_arg));
  LatticeComplex phase_m2q = cmplx(cos(phase_m2q_arg), sin(phase_m2q_arg));

  // momentum project propagators. TODO DPropagator vs Propagator (or DiracPropagator?)? In npr_vertex_w.cc they use DPropagator
  SftMom dummyPhases(1, false, -1); // use dummyPhases.getSet() for the sumMulti subset
  // int vol = Layout::vol();
  DPropagator momproj_prop_k1 = sumMulti(phase_k1 * quark_prop_k1, dummyPhases.getSet())[0] / (double) vol;
  DPropagator momproj_prop_k2 = sumMulti(phase_k2 * quark_prop_k2, dummyPhases.getSet())[0] / (double) vol;
  DPropagator momproj_prop_q = sumMulti(phase_q * quark_prop_q, dummyPhases.getSet())[0] / (double) vol;

	// } end first method

	// TODO may need to negate these. Begin other method {

	/*

  multi2d<int> k_list;  // k_list[0] = k1, k_list[1] = k2, k_list[2] = q
  k_list.resize(5, 4);
  // k_list[0][0] = -k;     // k1
  // k_list[0][1] = 0;
  // k_list[0][2] = k;
  // k_list[0][3] = 0;
  k_list[0][0] = k;
  k_list[0][1] = 0;
  k_list[0][2] = -k;
  k_list[0][3] = 0;

  // k_list[1][0] = 0;      // k2
  // k_list[1][1] = k;
  // k_list[1][2] = k;
  // k_list[1][3] = 0;
  k_list[1][0] = 0;
  k_list[1][1] = -k;
  k_list[1][2] = -k;
  k_list[1][3] = 0;

  // k_list[2][0] = k;        // q
  // k_list[2][1] = k;
  // k_list[2][2] = 0;
  // k_list[2][3] = 0;
  k_list[2][0] = -k;
  k_list[2][1] = -k;
  // k_list[2][1] = 0; // TODO only doing this temporarily while testing
  k_list[2][2] = 0;
  k_list[2][3] = 0;

  // k_list[3][0] = -k;        // -q
  // k_list[3][1] = -k;
  // k_list[3][2] = 0;
  // k_list[3][3] = 0;
  k_list[3][0] = k;
  k_list[3][1] = k;
  k_list[3][2] = 0;
  k_list[3][3] = 0;

  // k_list[4][0] = -2 * k;        // -2q
  // k_list[4][1] = -2 * k;
  // k_list[4][2] = 0;
  // k_list[4][3] = 0;
  k_list[4][0] = 2 * k;
  k_list[4][1] = 2 * k;
  k_list[4][2] = 0;
  k_list[4][3] = 0;

  SftMom phases (k_list, Nd);
  DPropagator momproj_prop_k1 = sumMulti(phases[0] * quark_prop_k1, phases.getSet())[0] / (double) vol;
  DPropagator momproj_prop_k2 = sumMulti(phases[1] * quark_prop_k2, phases.getSet())[0] / (double) vol;
  DPropagator momproj_prop_q = sumMulti(phases[2] * quark_prop_q, phases.getSet())[0] / (double) vol;
  // TODO change the method and make sure it works the way I was doing it before

	*/
  // } end other method

  int G5 = Ns*Ns-1;
  LatticePropagator antiprop_k2 = Gamma(G5) * adj(quark_prop_k2) * Gamma(G5);

  // Write mom projected props to file-- need Sq for Zq analysis
	int n_props = 3;
	XMLArrayWriter xml_props(xml, n_props);
	push(xml_props, "propagators");

	push(xml_props, "S_k1");
	write(xml_props, "k1", k1_int);
	write(xml_props, "prop", momproj_prop_k1);
	pop(xml_props);

	push(xml_props, "S_k2");
	write(xml_props, "k2", k2_int);
	write(xml_props, "prop", momproj_prop_k2);
	pop(xml_props);

	push(xml_props, "S_q");
	write(xml_props, "q", q_int);
	write(xml_props, "prop", momproj_prop_q);
	pop(xml_props);

	pop(xml_props);

  // Compute vector and axial currents
  multi1d<Propagator> GV;
  multi1d<Propagator> GA;
  GV.resize(Nd);
  GA.resize(Nd);
	multi1d<int> vectorGamma;				// 2**mu - 1
	vectorGamma.resize(Nd);
	vectorGamma[0] = 1;
	vectorGamma[1] = 2;
	vectorGamma[2] = 4;
	vectorGamma[3] = 8;
  for(int mu = 0; mu < Nd; mu++) {
		int gamIdx = vectorGamma[mu];
    GV[mu] = sumMulti(phase_mq * (antiprop_k2 * (Gamma(gamIdx) * quark_prop_k1)), dummyPhases.getSet())[0] / (double) vol;
    GA[mu] = sumMulti(phase_mq * (antiprop_k2 * (Gamma(gamIdx) * (Gamma(G5) * quark_prop_k1))), dummyPhases.getSet())[0] / (double) vol;
    // GV[mu] = sumMulti(phases[3] * (antiprop_k2 * (Gamma(gamIdx) * quark_prop_k1)), phases.getSet())[0] / (double) vol;
    // GA[mu] = sumMulti(phases[3] * (antiprop_k2 * (Gamma(gamIdx) * (Gamma(G5) * quark_prop_k1))), phases.getSet())[0] / (double) vol;
  }

  // Write current correlators
	XMLArrayWriter xml_GV(xml, Nd);
	push(xml_GV, "GV");
  for (int mu = 0; mu < Nd; mu++) {
		push(xml_GV);
		write(xml_GV, "mu", mu);
		write(xml_GV, "correlator", GV[mu]);
		pop(xml_GV);
  }
  pop(xml_GV);

	XMLArrayWriter xml_GA(xml, Nd);
	push(xml_GA, "GA");
  for (int mu = 0; mu < Nd; mu++) {
		push(xml_GA);
		write(xml_GA, "mu", mu);
    write(xml_GA, "correlator", GA[mu]);
		pop(xml_GA);
  }
	pop(xml_GA);

  // Loop over gamma matrix insertions
  XMLArrayWriter xml_gamma(xml, Ns * Ns);    // makes 16 entries under this tag
  push(xml_gamma, "four_point_function");
  for (int n = 0; n < (Ns * Ns); n++) {
    push(xml_gamma);     // next array element
    write(xml_gamma, "gamma_value", n);
		// TODO may need to do const and reference with & here
    LatticePropagator A_gamma = antiprop_k2 * (Gamma(n) * quark_prop_k1);
    // Tie it up
    for (int alpha = 0; alpha < Nd; alpha++) {
      for (int beta = 0; beta < Nd; beta++) {
        for (int rho = 0; rho < Nd; rho++) {
          for (int sigma = 0; sigma < Nd; sigma++) {
            for (int a = 0; a < Nc; a++) {
              for (int b = 0; b < Nc; b++) {
                for (int c = 0; c < Nc; c++) {
                  for (int d = 0; d < Nc; d++) {

										LatticeSpinMatrix Aab;
										LatticeSpinMatrix Acd;
										LatticeSpinMatrix Aad;
										LatticeSpinMatrix Acb;

										LatticeComplex Aab_comp;
										LatticeComplex Acd_comp;
										LatticeComplex Aad_comp;
										LatticeComplex Acb_comp;

										Aab = peekColor(A_gamma, a, b);
										Acd = peekColor(A_gamma, c, d);
										Aad = peekColor(A_gamma, a, d);
										Acb = peekColor(A_gamma, c, b);

										Aab_comp = peekSpin(Aab, alpha, beta);
										Acd_comp = peekSpin(Acd, rho, sigma);
										Aad_comp = peekSpin(Aad, alpha, sigma);
										Acb_comp = peekSpin(Acb, rho, beta);

										Complex Gcomp = 2 * sumMulti(phase_m2q * (Aab_comp * Acd_comp - Aad_comp * Acb_comp), dummyPhases.getSet())[0] / (double) vol;
                    // Complex Gcomp = 2 * sumMulti(phases[4] * (Aab_comp * Acd_comp - Aad_comp * Acb_comp), phases.getSet())[0] / (double) vol;

                    double epsilon = 1.0e-15;    // tolerance
                    Real reG = real(Gcomp); // may need to do this in 2 steps
                    Real imG = imag(Gcomp);
                    Real normG = sqrt(pow(reG, 2) + pow(imG, 2));
                    double normG_d = toDouble(normG);
                    if (normG_d > epsilon) {    // only print nonzero components to save memory
                      push(xml_gamma, "elem");
                      write(xml_gamma, "alpha", alpha);
                      write(xml_gamma, "beta", beta);
                      write(xml_gamma, "rho", rho);
                      write(xml_gamma, "sigma", sigma);
                      write(xml_gamma, "a", a);
                      write(xml_gamma, "b", b);
                      write(xml_gamma, "c", c);
                      write(xml_gamma, "d", d);
                      write(xml_gamma, "comp", Gcomp);
                      pop(xml_gamma);
                    }

                  }
                }
              }
            }
          }
        }
      }
    } // end iteration over propagator indices
    pop(xml_gamma);
  } // end iteration over gamma matrix index n

  pop(xml_gamma);
	// pop(xml);

  END_CODE();
}

}  // end namespace Chroma
