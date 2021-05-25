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
 * \param t0            timeslice coordinate of the source ( Read )
 * \param phases        object holds list of momenta and Fourier phases ( Read )
 * \param xml           xml file object ( Write )
 * \param xml_group     std::string used for writing xml data ( Read )
 *
 *        ____
 *        \
 * m(t) =  >  < m(t_source, 0) m(t + t_source, x) >
 *        /
 *        ----
 *          x
 */

// TODO add extra prop
void zero_nubb(const LatticePropagator& quark_prop_k1,
	     const LatticePropagator& quark_prop_k2,
			 const LatticePropagator& quark_prop_q,
       int n,
	     // const SftMom& phases,
	     // int t0,
	     XMLWriter& xml,
	     const std::string& xml_group)
{
  START_CODE();

  // Length of lattice in decay direction
  // int length = phases.numSubsets();

  // Construct phases (can't use SFT because we need fermionic bcs, and SFT only takes integer moms)

  // initialize momenta k1, k2, q
  int k = params.param.mom_idx
  multi1d<double> bvec;
  bvec.resize(4);
  bvec[0] = 0.0;
  bvec[1] = 0.0;
  bvec[2] = 0.0;
  bvec[3] = 0.5;

  multi1d<double> k1;
  k1.resize(4);
  k1[0] = (double) -k;
  k1[1] = 0.0;
  k1[2] = (double) k;
  k1[3] = 0.0;

  multi1d<double> k2;
  k2.resize(4);
  k2[0] = 0.0;
  k2[1] = (double) k;
  k2[2] = (double) k;
  k2[3] = 0.0;

  multi1d<double> q;
  q.resize(4);
  q[0] = (double) k;
  q[1] = (double) k;
  q[2] = 0.0;
  q[3] = 0.0;

  // Initialize phases in order (k1 + bvec, k2 + bvec, -q, -2q)
  // multi2d<int> momList;
  // momList.resize(4, 4);
  // for(i = 0; i < 4; i++) {
  //   momList[0][i] = k1[i] + bvec[i];
  //   momList[2][i] = -q[i];
  //   momList[3][i] = -2 * q[i];
  // }


  LatticeReal phase_k1_arg = zero;
  LatticeReal phase_k2_arg = zero;
  LatticeReal phase_q_arg = zero;
  LatticeReal phase_mq_arg = zero;  // no bvec for these-- they cancel in the momentum proj step
  LatticeReal phase_m2q_arg = zero;
  for (mu = 0; mu < 4; mu++) {
    phase_k1_arg += Layout::latticeCoordinate(mu) * (k1[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
    phase_k2_arg += Layout::latticeCoordinate(mu) * (k2[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
    phase_q_arg += Layout::latticeCoordinate(mu) * (q[mu] + bvec[mu]) * twopi / Real(Layout::lattSize()[mu]);
    phase_mq_arg -= Layout::latticeCoordinate(mu) * q[mu] * twopi / Real(Layout::lattSize()[mu]);
    phase_m2q_arg -= Layout::latticeCoordinate(mu) * 2 * q[mu] * twopi / Real(Layout::lattSize()[mu]);
  }
  LatticeComplex phase_k1 = cmplx(cos(phase_k1_arg), sin(phase_k1_arg));
  LatticeComplex phase_k2 = cmplx(cos(phase_k2_arg), sin(phase_k2_arg));
  LatticeComplex phase_q = cmplx(cos(phase_q_arg), sin(phase_q_arg));
  LatticeComplex phase_mq = cmplx(cos(phase_mq_arg), sin(phase_mq_arg));
  LatticeComplex phase_m2q = cmplx(cos(phase_m2q_arg), sin(phase_m2q_arg));

  // momentum project propagators. TODO DPropagator vs Propagator (or DiracPropagator?)? In npr_vertex_w.cc they use DPropagator
  SftMom dummyPhases(1, false, -1); // use dummyPhases.getSet() for the sumMulti subset
  int vol = Layout::vol();
  Propagator momproj_prop_k1 = sumMulti(phase_k1 * quark_prop_k1, dummyPhases.getSet())[0] / (double) vol;
  Propagator momproj_prop_k1 = sumMulti(phase_k2 * quark_prop_k2, dummyPhases.getSet())[0] / (double) vol;
  Propagator momproj_prop_q = sumMulti(phase_q * quark_prop_q, dummyPhases.getSet())[0] / (double) vol;

  int G5 = Ns*Ns-1;
  LatticePropagator antiprop_k2 = Gamma(G5) * adj(quark_prop_k2) * Gamma(G5);

  // Write S(q) to file-- need it for Zq analysis
  push(xml, "S_q")
    write(xml, "q", q)
    write(xml, "propagator", momproj_prop_q)
  pop(xml)

  // Compute vector and axial currents
  multi1d<Propagator> GV;
  multi1d<Propagator> GA;
  GV.resize(Nd);
  GA.resize(Nd);
  for(int mu = 0; mu < Nd; mu++) {
    int gamIdx = (2 ** mu) - 1;
    GV[mu] = sumMulti(phase_mq * (antiprop_k2 * (Gamma(gamIdx) * quark_prop_k1)), dummyPhases.getSet())[0] / (double) vol;
    GA[mu] = sumMulti(phase_mq * (antiprop_k2 * ((Gamma(gamIdx) * Gamma(G5)) * quark_prop_k1)), dummyPhases.getSet())[0] / (double) vol;
  }

  // Write current correlators
  push(xml, "GV")
  for (int mu = 0; mu < Nd; mu++) {
    write(xml, "mu", mu)
    write(xml, "correlator", GV[mu])
  }
  pop(xml)

  push(xml, "GA")
  for (int mu = 0; mu < Nd; mu++) {
    write(xml, "mu", mu)
    write(xml, "correlator", GA[mu])
  }
  pop(xml)

  // Loop over gamma matrix insertions
  XMLArrayWriter xml_gamma(xml, Ns * Ns);    // makes 16 entries under this tag
  push(xml_gamma, "four_point_function");

  for (int n=0; n < (Ns*Ns); n++)
  {
    push(xml_gamma);     // next array element
    write(xml_gamma, "gamma_value", n);

    // Construct the four-point correlation function
    // LatticeComplex corr_fn;
    // corr_fn = trace(adj(antiprop_k2) * (Gamma(gamma_value) *
    //                 quark_prop_1 * Gamma(gamma_value)));
    //
    // multi2d<DComplex> hsum;
    // hsum = phases.sft(corr_fn);

    LatticePropagator A_gamma = antiprop_k2 * (Gamma(n) * quark_prop_k1);
    // Tie it up
    for (int alpha = 0; alpha < Nd; alpha++) {
      push(xml_gamma);
      write(xml_gamma, "alpha", alpha);
      for (int beta = 0; beta < Nd; beta++) {
        push(xml_gamma);
        write(xml_gamma, "beta", beta);
        for (int rho = 0; rho < Nd; rho++) {
          push(xml_gamma);
          write(xml_gamma, "rho", rho);
          for (int sigma = 0; sigma < Nd; sigma++) {
            push(xml_gamma);
            write(xml_gamma, "sigma", sigma);
            for (int a = 0; a < Nc; a++) {
              push(xml_gamma);
              write(xml_gamma, "a", a);
              for (int b = 0; b < Nc; b++) {
                push(xml_gamma);
                write(xml_gamma, "b", b);
                for (int c = 0; c < Nc; c++) {
                  push(xml_gamma);
                  write(xml_gamma, "c", c);
                  for (int d = 0; d < Nc; d++) {
                    push(xml_gamma);
                    write(xml_gamma, "d", d);

                    Complex Gcomp = 2 * sumMulti( phase_m2q * (
                      peekSpin(peekColor(A_gamma, a, b), alpha, beta) * peekSpin(peekColor(A_gamma, c, d), rho, sigma) -
                      peekSpin(peekColor(A_gamma, a, d), alpha, sigma) * peekSpin(peekColor(A_gamma, c, b), rho, beta)
                    ), dummyPhases.getSet())[0] / (double) vol;
                    push(xml_gamma);
                    write(xml_gamma, "comp", Gcomp);
                    pop(xml_gamma);

                    pop(xml_gamma);
                  }
                  pop(xml_gamma);
                }
                pop(xml_gamma);
              }
              pop(xml_gamma);
            }
            pop(xml_gamma);
          }
          pop(xml_gamma);
        }
        pop(xml_gamma);
      }
      pop(xml_gamma);
    } // end iteration over propagator indices


    pop(xml_gamma);
  } // end for(n)

  pop(xml_gamma);

  END_CODE();
}

}  // end namespace Chroma
