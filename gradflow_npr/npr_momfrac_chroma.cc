// NPR momentum-fraction measurement for Chroma.
//
// Reproduces compute_npr_through_op from the QLUA npr_momfrac (old) script,
// used for the quark EMT / <x> renormalization in Detmold et al. 2020.
//
// Two build routes:
//   (a) Standalone binary (default): compile as-is. main() registers the
//       measurement into a stock Chroma and runs Chroma's driver loop.
//   (b) Inside a Chroma or LALIBE tree: compile with -DNPR_MOMFRAC_NO_MAIN
//       and add this file to the inline-hadron aggregate. Use this route when
//       the fermion action you want lives in that tree (e.g. exponential
//       clover), since it must be registered in whichever binary runs.
#include "chroma.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace Chroma;

namespace NprMomfrac {

  struct Params {
    Params() : frequency(0) {}
    Params(XMLReader& xml_in, const std::string& path);
    unsigned long frequency;
    std::string   gauge_id;
    bool          dump_gamma;   // Task 2: dump Gamma(1<<mu) and exit the measurement
    bool          test_project; // Task 3: self-test the momentum projection
    bool          test_seqsrc;  // Task 4: self-test the sequential source
    multi1d<int>  t_srce;       // source point y, explicit -- never drawn
    multi1d<Real> bvec;         // twist, (0,0,0,1/2) for antiperiodic time

    // Momenta: exactly one of mom_list or mom_range, with an optional cut.
    bool                  have_mom_list;
    multi1d<multi1d<int> > mom_list;
    bool                  have_mom_range;
    multi1d<int>          k_min, k_max;
    bool                  have_max_psq;
    double                max_psq;

    std::string output_type;    // TEXT (HDF5 not implemented yet)
    std::string output_file;

    GroupXML_t fermact;         // THE extension point -- swap the action here
    GroupXML_t invParam;
  };

  Params::Params(XMLReader& xml_in, const std::string& path) {
    XMLReader paramtop(xml_in, path);
    if (paramtop.count("Frequency") == 1) read(paramtop, "Frequency", frequency);
    else frequency = 1;
    read(paramtop, "NamedObject/gauge_id", gauge_id);

    if (paramtop.count("Param/dump_gamma") == 1)
      read(paramtop, "Param/dump_gamma", dump_gamma);
    else
      dump_gamma = false;

    if (paramtop.count("Param/test_project") == 1)
      read(paramtop, "Param/test_project", test_project);
    else
      test_project = false;

    if (paramtop.count("Param/test_seqsrc") == 1)
      read(paramtop, "Param/test_seqsrc", test_seqsrc);
    else
      test_seqsrc = false;

    if (paramtop.count("Param/t_srce") == 1) {
      read(paramtop, "Param/t_srce", t_srce);
    } else {
      t_srce.resize(Nd);
      for (int mu = 0; mu < Nd; ++mu) t_srce[mu] = 0;
    }

    if (paramtop.count("Param/bvec") == 1) {
      read(paramtop, "Param/bvec", bvec);
    } else {
      // Default is the production convention: antiperiodic in time.
      bvec.resize(Nd);
      for (int mu = 0; mu < Nd; ++mu) bvec[mu] = Real(0);
      bvec[Nd-1] = Real(0.5);
    }

    // Momenta. Exactly one of the two forms; a run that gives both is almost
    // certainly a mistake, so refuse it rather than silently picking one.
    have_mom_list  = (paramtop.count("Param/mom_list")  == 1);
    have_mom_range = (paramtop.count("Param/mom_range") == 1);

    if (have_mom_list && have_mom_range) {
      QDPIO::cerr << "NPR_MOMFRAC: give exactly one of <mom_list> and "
                  << "<mom_range>, not both" << std::endl;
      QDP_abort(1);
    }
    if (have_mom_list)  read(paramtop, "Param/mom_list", mom_list);
    if (have_mom_range) {
      read(paramtop, "Param/mom_range/k_min", k_min);
      read(paramtop, "Param/mom_range/k_max", k_max);
    }

    have_max_psq = (paramtop.count("Param/max_psq") == 1);
    max_psq = 0.0;
    if (have_max_psq) {
      Real tmp; read(paramtop, "Param/max_psq", tmp); max_psq = toDouble(tmp);
    }

    if (paramtop.count("Param/output_type") == 1)
      read(paramtop, "Param/output_type", output_type);
    else
      output_type = "TEXT";

    output_file = "";
    if (paramtop.count("Param/output_file") == 1)
      read(paramtop, "Param/output_file", output_file);

    // The action and the inverter come from Chroma's own XML factories, which
    // is exactly what makes the action swappable without touching code.
    //
    // Read these through a sub-reader rooted at Param, so the path handed to
    // readXMLGroup is a bare element name. readXMLGroup stores path = "/" +
    // path while capturing only the named subtree, so passing
    // "Param/FermionAction" here records "/Param/FermionAction" against XML
    // whose root is <FermionAction> -- and the factory lookup then fails at
    // measurement time, not at read time, which makes it look like a solver
    // problem rather than a path problem.
    {
      XMLReader ptop(paramtop, "Param");
      if (ptop.count("FermionAction") == 1)
        fermact = readXMLGroup(ptop, "FermionAction", "FermAct");
      if (ptop.count("InvertParam") == 1)
        invParam = readXMLGroup(ptop, "InvertParam", "invType");
    }
  }

  const Real twopi = Real(6.283185307179586476925286766559);

  //! Momentum projection.
  /*!
   * sum_x exp( i sum_mu (x-y)_mu (k_mu + b_mu) 2pi / L_mu ) F(x).
   *
   * Three conventions are baked in here and all three are load-bearing; see
   * chroma-port-design.md. The phase is e^{+ip.(x-y)}, so the source point y
   * enters as an offset rather than an overall phase. The twist b makes the
   * momentum match the fermion boundary conditions, b = (0,0,0,1/2). And
   * there is NO 1/V -- QLUA's production script had that normalisation
   * deleted, and the 2020 analysis assumes it is absent.
   *
   * The sign convention matches Chroma's own SftMom (sftmom.cc:440).
   */
  DPropagator projectMomentum(const LatticePropagator& F,
                              const multi1d<int>& k, const multi1d<int>& y,
                              const multi1d<Real>& bvec)
  {
    LatticeReal arg = zero;
    for (int mu = 0; mu < Nd; ++mu) {
      arg += LatticeReal(Layout::latticeCoordinate(mu) - y[mu])
           * twopi * (Real(k[mu]) + bvec[mu]) / Real(Layout::lattSize()[mu]);
    }
    return sum(cmplx(cos(arg), sin(arg)) * F);
  }

  //! The sequential source for the operator O_{mu mu}.
  /*!
   * b_mu(x) = U_mu(x) gamma_mu S(x+mu) - U_mu^dag(x-mu) gamma_mu S(x-mu)
   *
   * This is emt_npr.qlua's get_sequential_source (line 127), NOT
   * get_sequential_source_full (line 115). The two differ by the trace
   * subtraction, and the production data was made with the plain one -- proved
   * from the data itself, since sum_mu O_{mu mu} is O(1) rather than zero.
   * There is also no factor of 1/2 here; the 1/2 lives in the O() helper,
   * which this routine does not go through.
   *
   * NOTE: u MUST be the raw gauge field, never FermState::getLinks(), which
   * carries the antiperiodic boundary phase. QLUA applied its bcs inside the
   * Dirac operator only; the derivative above used raw links and a periodic
   * shift. This is the predicted source of any O_44 discrepancy.
   */
  LatticePropagator seqSource(const multi1d<LatticeColorMatrix>& u,
                              const LatticePropagator& S, int mu)
  {
    return u[mu] * (Gamma(1 << mu) * shift(S, FORWARD, mu))
         - adj(shift(u[mu], BACKWARD, mu)) * (Gamma(1 << mu) * shift(S, BACKWARD, mu));
  }

  //! The momentum set: a HYPERCUBE in k, not a ball.
  /*!
   * The distinction is not cosmetic: |k|_inf <= 6 is 28561 momenta while
   * |k|_1 <= 6 is 1289, a factor of 22. The 2020 production used the
   * hypercube, so the range form takes explicit per-direction bounds rather
   * than a single half-width -- which also allows a non-cubic range, and this
   * geometry arguably wants one (L_t = 48 against L_s = 16).
   *
   * The optional max_psq cut is applied on top; omit it to reproduce 2020.
   */
  multi1d<multi1d<int> > buildMomenta(const Params& p)
  {
    std::vector<multi1d<int> > out;

    if (p.have_mom_list) {
      for (int i = 0; i < p.mom_list.size(); ++i) out.push_back(p.mom_list[i]);
    } else {
      multi1d<int> k(Nd);
      for (k[0] = p.k_min[0]; k[0] <= p.k_max[0]; ++k[0])
      for (k[1] = p.k_min[1]; k[1] <= p.k_max[1]; ++k[1])
      for (k[2] = p.k_min[2]; k[2] <= p.k_max[2]; ++k[2])
      for (k[3] = p.k_min[3]; k[3] <= p.k_max[3]; ++k[3]) {
        if (p.have_max_psq) {
          double psq = 0.0;
          for (int m = 0; m < Nd; ++m) {
            double s = 2.0 * std::sin(M_PI * (k[m] + toDouble(p.bvec[m]))
                                      / Layout::lattSize()[m]);
            psq += s * s;
          }
          if (psq > p.max_psq) continue;
        }
        out.push_back(k);
      }
    }

    multi1d<multi1d<int> > res(out.size());
    for (size_t i = 0; i < out.size(); ++i) res[i] = out[i];
    return res;
  }

  //! One line per entry:
  //!   <tag> k0 k1 k2 k3 spin_row spin_col colour_row colour_col re im
  /*!
   * DPropagator is site-local and identical on every node after sum(), so
   * guarding on primaryNode() writes each entry exactly once.
   */
  void writeEntry(std::ofstream& fout, const std::string& tag,
                  const multi1d<int>& k, const DPropagator& P)
  {
    if (!Layout::primaryNode()) return;
    for (int s0 = 0; s0 < Ns; ++s0)
    for (int s1 = 0; s1 < Ns; ++s1)
    for (int c0 = 0; c0 < Nc; ++c0)
    for (int c1 = 0; c1 < Nc; ++c1) {
      ColorMatrix cm = peekSpin(P, s0, s1);
      Complex     z  = peekColor(cm, c0, c1);
      fout << tag << " " << k[0] << " " << k[1] << " " << k[2] << " " << k[3]
           << " " << s0 << " " << s1 << " " << c0 << " " << c1
           << " " << toDouble(real(z)) << " " << toDouble(imag(z)) << "\n";
    }
  }

  class InlineNprMomfrac : public AbsInlineMeasurement {
  public:
    InlineNprMomfrac(const Params& p) : params(p) {}
    unsigned long getFrequency() const { return params.frequency; }
    void operator()(unsigned long update_no, XMLWriter& xml_out) {
      QDPIO::cout << "NPR_MOMFRAC: measurement reached, gauge_id = "
                  << params.gauge_id << std::endl;

      if (params.dump_gamma)   { dumpGamma(); }
      if (params.test_project) { testProject(); }
      if (params.test_seqsrc)  { testSeqSource(); }

      // The three flags above are self-tests that run instead of the
      // measurement. With none of them set, do the real thing.
      if (!params.dump_gamma && !params.test_project && !params.test_seqsrc)
        measure(xml_out);
      push(xml_out, "NprMomfrac");
      write(xml_out, "update_no", update_no);
      pop(xml_out);
    }
  private:
    // Dump Chroma's Gamma(1<<mu) entry by entry, so the claim that it matches
    // the 2020 analysis basis can be checked rather than assumed. Column col
    // is extracted by applying the matrix to the spin basis vector e_col.
    // The action is site-independent, so the origin is as good as any site.
    void dumpGamma() const {
      multi1d<int> orig(Nd);
      for (int mu = 0; mu < Nd; ++mu) orig[mu] = 0;

      // The identity propagator is delta_spin * delta_colour, so
      // Gamma(1<<mu) * one has spin entry (row,col) equal to gamma[mu][row][col]
      // times the colour identity. The action is site-independent, so the
      // origin is as good as any site.
      LatticePropagator one = 1;

      for (int mu = 0; mu < Nd; ++mu) {
        LatticePropagator ge = Gamma(1 << mu) * one;
        Propagator p = peekSite(ge, orig);

        for (int row = 0; row < Ns; ++row) {
          for (int col = 0; col < Ns; ++col) {
            ColorMatrix cm = peekSpin(p, row, col);
            Complex     z  = peekColor(cm, 0, 0);
            std::ostringstream os;
            os << std::setprecision(17)
               << "GAMMA " << mu << " " << row << " " << col << " "
               << toDouble(real(z)) << " " << toDouble(imag(z));
            QDPIO::cout << os.str() << std::endl;
          }
        }
      }
    }

    // Self-test for projectMomentum, exploiting exact orthogonality.
    //
    // Feed F(x) = exp(-i q.(x-y)) * 1 at the twisted momentum for k0. The
    // projector carries e^{+i(k+b).(x-y)} (QLUA's sign, emt_npr.qlua:303-307),
    // so field and projector cancel at k = k0 and the sum is exactly V; at
    // every other k the phases are orthogonal and the sum is exactly 0.
    //
    // Note the CONJUGATE phase on F. That is deliberate and it is what makes
    // this a test rather than a tautology: the projector's own sign stays as
    // the physics dictates. A sign error in projectMomentum would match at
    // k = -k0, which is outside the scanned box, so the diagonal would vanish
    // entirely. A dropped twist likewise leaves no diagonal. A dropped y
    // offset leaves the magnitude at V but rotates its phase, which the
    // checker's |diag - V| test catches.
    void testProject() const {
      multi1d<int> k0(Nd); k0[0] = 1; k0[1] = 1; k0[2] = 1; k0[3] = 2;

      LatticeReal arg = zero;
      for (int mu = 0; mu < Nd; ++mu) {
        arg += LatticeReal(Layout::latticeCoordinate(mu) - params.t_srce[mu])
             * twopi * (Real(k0[mu]) + params.bvec[mu])
             / Real(Layout::lattSize()[mu]);
      }

      LatticePropagator one = 1;
      LatticeComplex   ph  = cmplx(cos(arg), -sin(arg));   // conjugate: see above
      LatticePropagator F  = ph * one;

      multi1d<int> ks(Nd);
      for (ks[0] = 0; ks[0] < 3; ++ks[0])
      for (ks[1] = 0; ks[1] < 3; ++ks[1])
      for (ks[2] = 0; ks[2] < 3; ++ks[2])
      for (ks[3] = 0; ks[3] < 4; ++ks[3]) {
        DPropagator P = projectMomentum(F, ks, params.t_srce, params.bvec);
        ColorMatrix cm = peekSpin(P, 0, 0);
        Complex     z  = peekColor(cm, 0, 0);
        std::ostringstream os;
        os << std::setprecision(17)
           << "PROJ " << ks[0] << ks[1] << ks[2] << ks[3] << " "
           << toDouble(real(z)) << " " << toDouble(imag(z));
        QDPIO::cout << os.str() << std::endl;
      }
    }

    //! The measurement proper: one point-source solve, four sequential
    //! solves, then projection at every requested momentum.
    void measure(XMLWriter& xml_out) {
      StopWatch swatch;

      // 1. RAW links. Never state->getLinks() -- that carries the antiperiodic
      //    boundary phase, which belongs in the Dirac operator alone.
      const multi1d<LatticeColorMatrix>& u =
        TheNamedObjMap::Instance()
          .getData<multi1d<LatticeColorMatrix> >(params.gauge_id);

      // 2. Action and solver, from the XML factories.
      typedef LatticeFermion               T;
      typedef multi1d<LatticeColorMatrix>  P;
      typedef multi1d<LatticeColorMatrix>  Q;

      std::istringstream xml_s(params.fermact.xml);
      XMLReader fermacttop(xml_s);
      Handle<FermionAction<T,P,Q> > S_f(
        TheFermionActionFactory::Instance()
          .createObject(params.fermact.id, fermacttop, params.fermact.path));
      Handle<FermState<T,P,Q> > state(S_f->createState(u));

      // 3. Point source at y, then the propagator S.
      LatticePropagator src = zero;
      {
        Propagator one_site = 1;              // identity in spin and colour
        pokeSite(src, one_site, params.t_srce);
      }

      int ncg_had = 0;
      XMLBufferWriter solver_xml;
      push(solver_xml, "Solves");

      LatticePropagator S;
      swatch.reset(); swatch.start();
      S_f->quarkProp(S, solver_xml, src, 0, Nd-1, state, params.invParam,
                     QUARK_SPIN_TYPE_FULL, false, ncg_had);
      swatch.stop();
      QDPIO::cout << "NPR_MOMFRAC: point propagator done, " << ncg_had
                  << " iters, " << swatch.getTimeInSeconds() << " s" << std::endl;

      // 4. Four sequential sources and solves. RAW u again, deliberately.
      multi1d<LatticePropagator> M(Nd);
      for (int mu = 0; mu < Nd; ++mu) {
        LatticePropagator b = seqSource(u, S, mu);
        swatch.reset(); swatch.start();
        S_f->quarkProp(M[mu], solver_xml, b, 0, Nd-1, state, params.invParam,
                       QUARK_SPIN_TYPE_FULL, false, ncg_had);
        swatch.stop();
        QDPIO::cout << "NPR_MOMFRAC: sequential propagator mu = " << mu
                    << " done, " << swatch.getTimeInSeconds() << " s" << std::endl;
      }
      pop(solver_xml);

      // 5. Project and write.
      if (params.output_type != "TEXT") {
        QDPIO::cerr << "NPR_MOMFRAC: output_type '" << params.output_type
                    << "' not implemented; only TEXT is" << std::endl;
        QDP_abort(1);
      }

      multi1d<multi1d<int> > moms = buildMomenta(params);
      QDPIO::cout << "NPR_MOMFRAC: projecting at " << moms.size()
                  << " momenta" << std::endl;

      std::ofstream fout;
      if (Layout::primaryNode()) {
        fout.open(params.output_file.c_str());
        if (!fout) {
          QDPIO::cerr << "NPR_MOMFRAC: cannot open output file "
                      << params.output_file << std::endl;
          QDP_abort(1);
        }
        fout.precision(17);
        fout << std::scientific;
      }

      swatch.reset(); swatch.start();
      for (int i = 0; i < moms.size(); ++i) {
        const multi1d<int>& k = moms[i];

        DPropagator Sp = projectMomentum(S, k, params.t_srce, params.bvec);
        writeEntry(fout, "prop", k, Sp);

        for (int mu = 0; mu < Nd; ++mu) {
          DPropagator Op = projectMomentum(M[mu], k, params.t_srce, params.bvec);
          std::ostringstream tag; tag << "O" << (mu+1) << (mu+1);
          writeEntry(fout, tag.str(), k, Op);
        }
      }
      swatch.stop();

      if (Layout::primaryNode()) fout.close();

      QDPIO::cout << "NPR_MOMFRAC: projection done in "
                  << swatch.getTimeInSeconds() << " s; wrote "
                  << params.output_file << std::endl;

      push(xml_out, "NprMomfracResults");
      write(xml_out, "num_momenta", moms.size());
      write(xml_out, "t_srce", params.t_srce);
      write(xml_out, "output_file", params.output_file);
      pop(xml_out);
    }

    // Free-field test of seqSource.
    //
    // On a unit gauge field b_mu(x) = gamma_mu [S(x+mu) - S(x-mu)]. Feeding
    // the CONJUGATE plane wave S(x) = exp(-i q.x) * 1 gives
    //   b_mu(x) = gamma_mu S(x) (e^{-i q_mu} - e^{+i q_mu})
    //           = -2i sin(q_mu) gamma_mu S(x),
    // so projecting at q -- whose own phase is e^{+i q.x} -- gives exactly
    // -2i sin(q_mu) V gamma_mu. The conjugate is needed for the same reason as
    // in testProject: the projector's sign is fixed by the physics, so the
    // test field is the thing that has to be conjugate for the two to cancel.
    //
    // A swapped FORWARD/BACKWARD negates this exactly, which the checker names
    // explicitly rather than reporting as a generic mismatch.
    //
    // The test runs with bvec = 0, deliberately. A twisted plane wave is
    // antiperiodic in time (e^{-i q_3 L_3} = -1 for b_3 = 1/2) while QDP++'s
    // shift is periodic, so the analytic reference above is simply wrong on the
    // boundary time slices -- with the twist on, mu=3 misses by exactly 1/4
    // while mu=0,1,2 agree to 1e-16. That is a property of the REFERENCE, not
    // of seqSource, and it is worth stating plainly because it CONFIRMS the
    // design rather than contradicting it: the derivative is supposed to use
    // raw links and a periodic shift, with the antiperiodic boundary living in
    // the Dirac operator alone, which is what QLUA did. Task 3 tested the twist
    // on its own, where it belongs.
    //
    // Note the deliberate y = 0 here, in BOTH the plane wave and the projector.
    // The two must agree or they differ by a constant phase and the comparison
    // fails for a reason that has nothing to do with seqSource. Task 3 already
    // tested the y offset on its own.
    void testSeqSource() const {
      multi1d<int> k0(Nd); k0[0] = 1; k0[1] = 1; k0[2] = 1; k0[3] = 2;

      LatticeReal arg = zero;
      for (int mu = 0; mu < Nd; ++mu) {
        arg += LatticeReal(Layout::latticeCoordinate(mu))
             * twopi * (Real(k0[mu]) + params.bvec[mu])
             / Real(Layout::lattSize()[mu]);
      }

      LatticePropagator one = 1;
      LatticePropagator S = cmplx(cos(arg), -sin(arg)) * one;   // conjugate

      const multi1d<LatticeColorMatrix>& u =
        TheNamedObjMap::Instance()
          .getData<multi1d<LatticeColorMatrix> >(params.gauge_id);

      multi1d<int> yzero(Nd);
      for (int m = 0; m < Nd; ++m) yzero[m] = 0;

      for (int mu = 0; mu < Nd; ++mu) {
        LatticePropagator b = seqSource(u, S, mu);
        DPropagator P = projectMomentum(b, k0, yzero, params.bvec);
        for (int r = 0; r < Ns; ++r) {
          for (int c = 0; c < Ns; ++c) {
            ColorMatrix cm = peekSpin(P, r, c);
            Complex     z  = peekColor(cm, 0, 0);
            std::ostringstream os;
            os << std::setprecision(17)
               << "SEQ " << mu << " " << r << " " << c << " "
               << toDouble(real(z)) << " " << toDouble(imag(z));
            QDPIO::cout << os.str() << std::endl;
          }
        }
      }
    }

    Params params;
  };

  namespace {
    AbsInlineMeasurement* createMeasurement(XMLReader& xml_in,
                                            const std::string& path) {
      return new InlineNprMomfrac(Params(xml_in, path));
    }
    bool registered = false;
  }

  const std::string name = "NPR_MOMFRAC";

  bool registerAll() {
    bool success = true;
    if (!registered) {
      success &= WilsonTypeFermActsEnv::registerAll();
      success &= TheInlineMeasurementFactory::Instance()
                   .registerObject(name, createMeasurement);
      registered = true;
    }
    return success;
  }

} // namespace NprMomfrac

#ifndef NPR_MOMFRAC_NO_MAIN

// Driver. This follows mainprogs/main/chroma.cc closely and deliberately: the
// point of the standalone route is to be stock Chroma's own loop with one
// extra registerAll() in front of the measurement read. Deviating from it is
// how the gauge field or the RNG ends up in a subtly different state.

namespace {

  struct DriverInput {
    multi1d<int> nrow;
    std::string  inline_measurement_xml;
    GroupXML_t   cfg;
    QDP::Seed    rng_seed;
  };

  void readDriverInput(XMLReader& xml, const std::string& path, DriverInput& p) {
    XMLReader top(xml, path);
    XMLReader paramtop(top, "Param");
    read(paramtop, "nrow", p.nrow);

    XMLReader measurements_xml(paramtop, "InlineMeasurements");
    std::ostringstream inline_os;
    measurements_xml.print(inline_os);
    p.inline_measurement_xml = inline_os.str();

    p.cfg = readXMLGroup(top, "Cfg", "cfg_type");

    if (top.count("RNG") > 0) read(top, "RNG", p.rng_seed);
    else                      p.rng_seed = 11;
  }

} // anonymous namespace

int main(int argc, char* argv[]) {
  Chroma::initialize(&argc, &argv);
  START_CODE();

  // Stock Chroma's registrations, exactly as chroma.cc's linkageHack() does
  // them, then ours on top. Both must precede the measurement read below --
  // that ordering is what makes runtime registration work.
  //
  // Note: this value is 0 in a stock Chroma too. Some sub-aggregate returns
  // false on this build; it is not a failure signal. The real check is that
  // the factory resolves NPR_MOMFRAC below.
  bool linkage = true;
  linkage &= InlineAggregateEnv::registerAll();
  linkage &= GaugeInitEnv::registerAll();
  linkage &= NprMomfrac::registerAll();
  QDPIO::cout << "NPR_MOMFRAC: linkage = " << linkage << std::endl;

  XMLReader xml_in;
  DriverInput input;
  try {
    xml_in.open(Chroma::getXMLInputFileName());
    readDriverInput(xml_in, "/chroma", input);
  }
  catch (const std::string& e) {
    QDPIO::cerr << "NPR_MOMFRAC: caught exception reading XML: " << e << std::endl;
    QDP_abort(1);
  }

  XMLFileWriter& xml_out = Chroma::getXMLOutputInstance();
  push(xml_out, "npr_momfrac");
  write(xml_out, "Input", xml_in);

  Layout::setLattSize(input.nrow);
  Layout::create();

  proginfo(xml_out);

  QDP::RNG::setrn(input.rng_seed);
  write(xml_out, "RNG", input.rng_seed);

  // Gauge field, through the same factory chroma.cc uses.
  multi1d<LatticeColorMatrix> u(Nd);
  XMLReader gauge_file_xml, gauge_xml;
  try {
    std::istringstream xml_c(input.cfg.xml);
    XMLReader cfgtop(xml_c);
    QDPIO::cout << "NPR_MOMFRAC: gauge initialization, cfg_type = "
                << input.cfg.id << std::endl;
    Handle<GaugeInit> gaugeInit(
      TheGaugeInitFactory::Instance().createObject(input.cfg.id, cfgtop,
                                                   input.cfg.path));
    (*gaugeInit)(gauge_file_xml, gauge_xml, u);
  }
  catch (const std::string& e) {
    QDPIO::cerr << "NPR_MOMFRAC: caught exception during gaugeInit: " << e
                << std::endl;
    QDP_abort(1);
  }

  XMLBufferWriter config_xml;
  config_xml << gauge_xml;
  write(xml_out, "Config_info", gauge_xml);

  // Cheap sanity check: on a unit field this must print plaquette 1.
  MesPlq(xml_out, "Observables", u);

  try {
    std::istringstream Measurements_is(input.inline_measurement_xml);
    XMLReader MeasXML(Measurements_is);
    multi1d<Handle<AbsInlineMeasurement> > the_measurements;
    read(MeasXML, "/InlineMeasurements", the_measurements);

    QDPIO::cout << "NPR_MOMFRAC: there are " << the_measurements.size()
                << " measurements" << std::endl;

    InlineDefaultGaugeField::reset();
    InlineDefaultGaugeField::set(u, config_xml);

    push(xml_out, "InlineObservables");
    xml_out.flush();

    unsigned long cur_update = 0;
    for (int m = 0; m < the_measurements.size(); ++m) {
      AbsInlineMeasurement& the_meas = *(the_measurements[m]);
      if (cur_update % the_meas.getFrequency() == 0) {
        push(xml_out, "elem");
        the_meas(cur_update, xml_out);
        pop(xml_out);
        xml_out.flush();
      }
    }

    pop(xml_out);  // InlineObservables
    InlineDefaultGaugeField::reset();
  }
  catch (const std::string& e) {
    QDPIO::cerr << "NPR_MOMFRAC: caught exception during measurement: " << e
                << std::endl;
    QDP_abort(1);
  }

  pop(xml_out);  // npr_momfrac

  QDPIO::cout << "NPR_MOMFRAC: ran successfully" << std::endl;

  END_CODE();
  Chroma::finalize();
  return 0;
}
#endif
