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
#include <fftw3.h>

using namespace Chroma;

namespace NprMomfrac {

  //! The production twist, b = (0,0,0,1/2): antiperiodic in time.
  multi1d<Real> antiperiodicTwist()
  {
    multi1d<Real> b(Nd);
    for (int mu = 0; mu < Nd; ++mu) b[mu] = Real(0);
    b[Nd-1] = Real(0.5);
    return b;
  }

  struct Params {
    Params() : frequency(0) {}
    Params(XMLReader& xml_in, const std::string& path);
    unsigned long frequency;
    std::string   gauge_id;
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

    if (paramtop.count("Param/test_fft") == 1)
      read(paramtop, "Param/test_fft", test_fft);
    else
      test_fft = false;

    // Source point. If absent it is drawn uniformly at measurement time, from
    // the RNG main() seeds with <RNG><Seed>; see drawSourcePoint().
    //
    // This used to default silently to the origin, which is the worst possible
    // point: t = 0 is adjacent to the antiperiodic boundary slice, where the
    // raw-link O_44 defect is O(1).
    have_tsrc = (paramtop.count("Param/tsrc") == 1);
    if (have_tsrc) read(paramtop, "Param/tsrc", tsrc);
    have_tsrc_from_xml = have_tsrc;

    if (paramtop.count("Param/bvec") == 1) {
      read(paramtop, "Param/bvec", bvec);
    } else {
      // Default is the production convention: antiperiodic in time.
      bvec = antiperiodicTwist();
    }

    // Momenta. Exactly one of the two forms; a run that gives both, or
    // neither, is a mistake, so refuse it rather than silently picking one.
    have_mom_list = (paramtop.count("Param/mom_list") == 1);
    have_ksq_cut  = (paramtop.count("Param/ksq_cut")  == 1);
    if (have_mom_list == have_ksq_cut) {
      QDPIO::cerr << "NPR_MOMFRAC: give exactly one of <mom_list> and "
                  << "<ksq_cut>" << std::endl;
      QDP_abort(1);
    }
    if (have_mom_list) read(paramtop, "Param/mom_list", mom_list);
    ksq_cut = 0;
    if (have_ksq_cut) read(paramtop, "Param/ksq_cut", ksq_cut);

    // Refinements of the ball, on the INTEGER k -- not the lattice momentum,
    // and not k + b.
    h_cut = 1.0;
    if (paramtop.count("Param/h_cut") == 1) {
      Real tmp; read(paramtop, "Param/h_cut", tmp); h_cut = toDouble(tmp);
    }
    all_pos = true;
    if (paramtop.count("Param/all_pos") == 1) read(paramtop, "Param/all_pos", all_pos);

    // Beside an explicit list these would be silently ignored, which reads
    // as though they were applied.
    if (have_mom_list && (paramtop.count("Param/h_cut") == 1
                          || paramtop.count("Param/all_pos") == 1)) {
      QDPIO::cerr << "NPR_MOMFRAC: <h_cut> and <all_pos> apply to <ksq_cut> "
                  << "only; remove them when using <mom_list>" << std::endl;
      QDP_abort(1);
    }

    projection = "FFT";
    if (paramtop.count("Param/projection") == 1)
      read(paramtop, "Param/projection", projection);
    if (projection != "FFT" && projection != "PHASE_SUM") {
      QDPIO::cerr << "NPR_MOMFRAC: <projection> must be FFT or PHASE_SUM, not '"
                  << projection << "'" << std::endl;
      QDP_abort(1);
    }
    check_fft = false;
    if (paramtop.count("Param/check_fft") == 1)
      read(paramtop, "Param/check_fft", check_fft);

    if (paramtop.count("Param/output_type") == 1)
      read(paramtop, "Param/output_type", output_type);
    else
      output_type = "TEXT";

    output_file = "";
    if (paramtop.count("Param/output_file") == 1)
      read(paramtop, "Param/output_file", output_file);

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
  LatticeComplex momentumPhase(const multi1d<int>& k, const multi1d<int>& y,
                               const multi1d<Real>& bvec)
  {
    LatticeReal arg = zero;
    for (int mu = 0; mu < Nd; ++mu) {
      arg += LatticeReal(Layout::latticeCoordinate(mu) - y[mu])
           * twopi * (Real(k[mu]) + bvec[mu]) / Real(Layout::lattSize()[mu]);
    }
    return cmplx(cos(arg), sin(arg));
  }

  //! Contract a precomputed phase against a field. No 1/V.
  DPropagator projectWithPhase(const LatticeComplex& ph,
                               const LatticePropagator& F)
  {
    return sum(ph * F);
  }

  //! Convenience wrapper: phase and contraction in one call.
  DPropagator projectMomentum(const LatticePropagator& F,
                              const multi1d<int>& k, const multi1d<int>& y,
                              const multi1d<Real>& bvec)
  {
    return projectWithPhase(momentumPhase(k, y, bvec), F);
  }

  //! Momentum projection by FFT: the same sums as projectMomentum,
  //!   P(k) = sum_x exp(+i p.(x - y)) F(x),   p_mu = 2 pi (k_mu + b_mu) / L_mu,
  //! no 1/V, returned in the order of moms. One FFTW_BACKWARD transform per
  //! spin-colour component gives every integer k at once. The twist enters as
  //! a premultiplication by exp(+i 2 pi b.x / L), and the source point as the
  //! scalar phase exp(-i p.y) afterward -- never as a circular shift.
  //!
  //! Adapted from FFT4d in Dimitra Pefkou's DeltaG_operator.cc. As there, each
  //! rank transforms a full-volume buffer holding its own sites and zeros
  //! elsewhere, and the picked-out values are summed over ranks; by linearity
  //! that is the transform of the whole field. It costs one volume-sized buffer
  //! per rank but needs no gather, and the ranks transform concurrently.
  multi1d<DPropagator> fftProject(const LatticePropagator& F,
                                  const multi1d<multi1d<int> >& moms,
                                  const multi1d<int>& y,
                                  const multi1d<Real>& bvec = antiperiodicTwist())
  {
    const multi1d<int>& L = Layout::lattSize();
    const long V    = long(L[0]) * L[1] * L[2] * L[3];
    const int  nloc = Layout::sitesOnNode();
    const int  nmom = moms.size();
    const int  ncmp = Ns * Ns * Nc * Nc;
    const double tp = 6.283185307179586476925286766559;

    // Local sites: row-major index (x slowest, t fastest) and twist phase.
    // QDP++ stores sites checkerboarded, so the index comes from coordinates.
    std::vector<long>   site_idx(nloc);
    std::vector<double> tw_re(nloc), tw_im(nloc);
    {
      multi1d<multi1d<Int> > coord(Nd);
      for (int mu = 0; mu < Nd; ++mu) {
        coord[mu].resize(nloc);
        QDP_extract(coord[mu], Layout::latticeCoordinate(mu), all);
      }
      for (int ii = 0; ii < nloc; ++ii) {
        long idx = 0; double arg = 0.0;
        for (int mu = 0; mu < Nd; ++mu) {
          int x = toInt(coord[mu][ii]);
          idx  = idx * L[mu] + x;
          arg += tp * toDouble(bvec[mu]) * x / L[mu];
        }
        site_idx[ii] = idx; tw_re[ii] = std::cos(arg); tw_im[ii] = std::sin(arg);
      }
    }

    // Requested momenta: wrapped index into the transform, and source phase.
    // The phase uses the unwrapped k; wrapping would change it by exp(-2 pi i y)
    // = 1 anyway, but unwrapped is the definition.
    std::vector<long>   mom_idx(nmom);
    std::vector<double> sp_re(nmom), sp_im(nmom);
    for (int i = 0; i < nmom; ++i) {
      long idx = 0; double arg = 0.0;
      for (int mu = 0; mu < Nd; ++mu) {
        int q = ((moms[i][mu] % L[mu]) + L[mu]) % L[mu];
        idx  = idx * L[mu] + q;
        arg -= tp * (double(moms[i][mu]) + toDouble(bvec[mu])) * y[mu] / L[mu];
      }
      mom_idx[i] = idx; sp_re[i] = std::cos(arg); sp_im[i] = std::sin(arg);
    }

    multi1d<Propagator> Floc(nloc);
    QDP_extract(Floc, F, all);

    fftw_complex* buf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * V);
    int dims[Nd];
    for (int mu = 0; mu < Nd; ++mu) dims[mu] = L[mu];
    // FFTW_BACKWARD is sum_x exp(+i k.x), our sign. ESTIMATE leaves buf alone.
    fftw_plan plan = fftw_plan_dft(Nd, dims, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);

    std::vector<double> out(2L * nmom * ncmp, 0.0);   // [mom, comp, re/im]
    for (int s0 = 0; s0 < Ns; ++s0)
    for (int s1 = 0; s1 < Ns; ++s1)
    for (int c0 = 0; c0 < Nc; ++c0)
    for (int c1 = 0; c1 < Nc; ++c1) {
      const int cmp = ((s0 * Ns + s1) * Nc + c0) * Nc + c1;

      for (long j = 0; j < V; ++j) { buf[j][0] = 0.0; buf[j][1] = 0.0; }
      for (int ii = 0; ii < nloc; ++ii) {
        const double re = Floc[ii].elem().elem(s0,s1).elem(c0,c1).real();
        const double im = Floc[ii].elem().elem(s0,s1).elem(c0,c1).imag();
        buf[site_idx[ii]][0] = re * tw_re[ii] - im * tw_im[ii];
        buf[site_idx[ii]][1] = re * tw_im[ii] + im * tw_re[ii];
      }

      fftw_execute(plan);

      for (int i = 0; i < nmom; ++i) {
        const double re = buf[mom_idx[i]][0], im = buf[mom_idx[i]][1];
        out[2L * (long(i) * ncmp + cmp)    ] = re * sp_re[i] - im * sp_im[i];
        out[2L * (long(i) * ncmp + cmp) + 1] = re * sp_im[i] + im * sp_re[i];
      }
    }
    fftw_destroy_plan(plan);
    fftw_free(buf);

    QDPInternal::globalSumArray(out.data(), int(out.size()));

    multi1d<DPropagator> res(nmom);
    for (int i = 0; i < nmom; ++i) {
      res[i] = zero;
      for (int s0 = 0; s0 < Ns; ++s0)
      for (int s1 = 0; s1 < Ns; ++s1)
      for (int c0 = 0; c0 < Nc; ++c0)
      for (int c1 = 0; c1 < Nc; ++c1) {
        const long o = 2L * (long(i) * ncmp + ((s0 * Ns + s1) * Nc + c0) * Nc + c1);
        res[i].elem().elem(s0,s1).elem(c0,c1).real() = out[o];
        res[i].elem().elem(s0,s1).elem(c0,c1).imag() = out[o + 1];
      }
    }
    return res;
  }

  //! The sequential source for the operator O_{mu mu}.
  LatticePropagator seqSource(const multi1d<LatticeColorMatrix>& u,
                              const LatticePropagator& S, int mu)
  {
    return u[mu] * (Gamma(1 << mu) * shift(S, FORWARD, mu))
         - adj(shift(u[mu], BACKWARD, mu)) * (Gamma(1 << mu) * shift(S, BACKWARD, mu));
  }

  //! The momentum set: a Euclidean ball in the integer k.
  multi1d<multi1d<int> > buildMomenta(const Params& p)
  {
    std::vector<multi1d<int> > out;

    if (p.have_mom_list) {
      for (int i = 0; i < p.mom_list.size(); ++i) out.push_back(p.mom_list[i]);
    } else {
      int r  = static_cast<int>(std::floor(std::sqrt(double(p.ksq_cut))));
      int lo = p.all_pos ? 0 : -r;

      long n_ball = 0, n_h = 0;
      multi1d<int> k(Nd);
      for (k[0] = lo; k[0] <= r; ++k[0])
      for (k[1] = lo; k[1] <= r; ++k[1])
      for (k[2] = lo; k[2] <= r; ++k[2])
      for (k[3] = lo; k[3] <= r; ++k[3]) {
        long ksq = 0, k4 = 0;
        for (int m = 0; m < Nd; ++m) {
          long k2 = long(k[m]) * k[m];
          ksq += k2; k4 += k2 * k2;
        }
        if (ksq > p.ksq_cut) continue;
        ++n_ball;
        if (p.h_cut < 1.0) {
          double denom = double(ksq) * double(ksq);
          if (ksq == 0 || double(k4) > p.h_cut * denom * (1.0 + 1e-12)) {
            ++n_h; continue;
          }
        }
        out.push_back(k);
      }

      QDPIO::cout << "NPR_MOMFRAC: momenta: " << n_ball << " with k^2 <= "
                  << p.ksq_cut << (p.all_pos ? " (all_pos)" : "")
                  << ", " << n_h << " removed by h_cut <= " << p.h_cut
                  << ", " << out.size() << " kept" << std::endl;
    }

    if (out.empty()) {
      QDPIO::cerr << "NPR_MOMFRAC: the momentum selection is empty" << std::endl;
      QDP_abort(1);
    }

    multi1d<multi1d<int> > res(out.size());
    for (size_t i = 0; i < out.size(); ++i) res[i] = out[i];
    return res;
  }

  //! One line per entry:
  //!   <tag> k0 k1 k2 k3 spin_row spin_col colour_row colour_col re im
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

      // Resolve the source point before anything uses it, self-tests included.
      if (!params.have_tsrc) drawSourcePoint();
      QDPIO::cout << "NPR_MOMFRAC: source point tsrc = ("
                  << params.tsrc[0] << "," << params.tsrc[1] << ","
                  << params.tsrc[2] << "," << params.tsrc[3] << ")  "
                  << (params.have_tsrc_from_xml ? "[given in XML]"
                                                : "[drawn from the RNG seed above]")
                  << std::endl;

      if (params.dump_gamma)   { dumpGamma(); }
      if (params.test_project) { testProject(); }
      if (params.test_seqsrc)  { testSeqSource(); }
      if (params.test_fft)     { testFft(); }

      // The flags above are self-tests that run instead of the
      // measurement. With none of them set, do the real thing.
      if (!params.dump_gamma && !params.test_project && !params.test_seqsrc
          && !params.test_fft)
        measure(xml_out);
      push(xml_out, "NprMomfrac");
      write(xml_out, "update_no", update_no);
      pop(xml_out);
    }
  private:
    void dumpGamma() const {
      multi1d<int> orig(Nd);
      for (int mu = 0; mu < Nd; ++mu) orig[mu] = 0;

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
    void testProject() const {
      multi1d<int> k0(Nd); k0[0] = 1; k0[1] = 1; k0[2] = 1; k0[3] = 2;

      LatticeReal arg = zero;
      for (int mu = 0; mu < Nd; ++mu) {
        arg += LatticeReal(Layout::latticeCoordinate(mu) - params.tsrc[mu])
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
        DPropagator P = projectMomentum(F, ks, params.tsrc, params.bvec);
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
    //! Draw tsrc uniformly on the lattice from the XML-seeded RNG.
    void drawSourcePoint() {
      params.tsrc.resize(Nd);
      for (int mu = 0; mu < Nd; ++mu) {
        int L = Layout::lattSize()[mu];
        Real r;
        random(r);                                  // uniform in [0,1)
        int c = static_cast<int>(toDouble(r) * L);
        if (c >= L) c = L - 1;                      // guard against r rounding to 1
        QDPInternal::broadcast(c);
        params.tsrc[mu] = c;
      }
      params.have_tsrc = true;
    }

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
        pokeSite(src, one_site, params.tsrc);
      }

      if (params.output_type != "TEXT") {
        QDPIO::cerr << "NPR_MOMFRAC: output_type '" << params.output_type
                    << "' not implemented; only TEXT is" << std::endl;
        QDP_abort(1);
      }

      multi1d<multi1d<int> > moms = buildMomenta(params);
      QDPIO::cout << "NPR_MOMFRAC: will project at " << moms.size()
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

      swatch.reset(); swatch.start();
      {
        std::vector<const LatticePropagator*> fields(1, &S);
        multi2d<DPropagator> Sk = project(fields, moms, params.check_fft);
        for (int i = 0; i < moms.size(); ++i)
          writeEntry(fout, "prop", moms[i], Sk(0, i));
      }
      if (Layout::primaryNode()) fout.flush();
      swatch.stop();
      QDPIO::cout << "NPR_MOMFRAC: prop projected and written in "
                  << swatch.getTimeInSeconds() << " s -- comparable now"
                  << std::endl;

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

      // 5. Second pass: the four operators.
      swatch.reset(); swatch.start();
      {
        std::vector<const LatticePropagator*> fields;
        for (int mu = 0; mu < Nd; ++mu) fields.push_back(&M[mu]);
        multi2d<DPropagator> Mk = project(fields, moms, params.check_fft);
        for (int i = 0; i < moms.size(); ++i) {
          for (int mu = 0; mu < Nd; ++mu) {
            std::ostringstream tag; tag << "O" << (mu+1) << (mu+1);
            writeEntry(fout, tag.str(), moms[i], Mk(mu, i));
          }
        }
      }
      swatch.stop();

      if (Layout::primaryNode()) fout.close();

      QDPIO::cout << "NPR_MOMFRAC: operators projected in "
                  << swatch.getTimeInSeconds() << " s; wrote "
                  << params.output_file << std::endl;

      push(xml_out, "NprMomfracResults");
      write(xml_out, "projection", params.projection);
      write(xml_out, "num_momenta", moms.size());
      write(xml_out, "tsrc", params.tsrc);
      write(xml_out, "output_file", params.output_file);
      pop(xml_out);
    }

    //! Project each field at every momentum, by the method <projection> names.
    //! Returns [field, momentum]. PHASE_SUM builds one phase field per momentum
    //! and shares it across the fields; FFT does one transform per component.
    multi2d<DPropagator> project(const std::vector<const LatticePropagator*>& Fs,
                                 const multi1d<multi1d<int> >& moms,
                                 bool check) const
    {
      const int nf = Fs.size(), nmom = moms.size();
      multi2d<DPropagator> P(nf, nmom);

      if (params.projection == "PHASE_SUM" || check) {
        for (int i = 0; i < nmom; ++i) {
          LatticeComplex ph = momentumPhase(moms[i], params.tsrc, params.bvec);
          for (int f = 0; f < nf; ++f) P(f, i) = projectWithPhase(ph, *Fs[f]);
        }
        if (params.projection == "PHASE_SUM") return P;
      }

      StopWatch sw; sw.reset(); sw.start();
      double max_diff = 0.0, max_abs = 0.0;
      for (int f = 0; f < nf; ++f) {
        multi1d<DPropagator> Pf = fftProject(*Fs[f], moms, params.tsrc, params.bvec);
        for (int i = 0; i < nmom; ++i) {
          if (check) {
            // Largest componentwise |FFT - phase sum|, against largest |phase sum|.
            for (int s0 = 0; s0 < Ns; ++s0)
            for (int s1 = 0; s1 < Ns; ++s1)
            for (int c0 = 0; c0 < Nc; ++c0)
            for (int c1 = 0; c1 < Nc; ++c1) {
              const RComplex<REAL64>& a = Pf[i].elem().elem(s0,s1).elem(c0,c1);
              const RComplex<REAL64>& b = P(f, i).elem().elem(s0,s1).elem(c0,c1);
              max_diff = std::max(max_diff, std::hypot(a.real() - b.real(),
                                                       a.imag() - b.imag()));
              max_abs  = std::max(max_abs,  std::hypot(b.real(), b.imag()));
            }
          }
          P(f, i) = Pf[i];
        }
      }
      sw.stop();
      QDPIO::cout << "NPR_MOMFRAC: FFT projection of " << nf << " field(s), "
                  << sw.getTimeInSeconds() << " s" << std::endl;
      if (check) {
        std::ostringstream os;
        os << std::setprecision(3) << std::scientific
           << "NPR_MOMFRAC: check_fft: max |FFT - phase sum| = " << max_diff
           << ", max |phase sum| = " << max_abs;
        QDPIO::cout << os.str() << std::endl;
      }
      return P;
    }

    //! Self-test for fftProject: a Gaussian random propagator field has no
    //! symmetry to hide a misplaced site, sign or twist. Uses the XML momenta,
    //! tsrc and bvec; give all_pos false to exercise negative (wrapped) k.
    void testFft() const {
      LatticePropagator F;
      gaussian(F);
      multi1d<multi1d<int> > moms = buildMomenta(params);
      std::vector<const LatticePropagator*> fields(1, &F);
      Params p = params;
      p.projection = "FFT";
      InlineNprMomfrac(p).project(fields, moms, true);
    }

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
  {
    // Logged because a source point drawn from it is only reproducible with it.
    XMLBufferWriter sx;
    write(sx, "Seed", input.rng_seed);
    // Print just the integers, e.g. "11 11 11 0", in the order <Seed> takes.
    const std::string str = sx.str();
    std::string out;
    for (std::string::size_type a = str.find("<elem>"); a != std::string::npos;
         a = str.find("<elem>", a + 1)) {
      std::string::size_type b = str.find("</elem>", a);
      out += (out.empty() ? "" : " ") + str.substr(a + 6, b - a - 6);
    }
    QDPIO::cout << "NPR_MOMFRAC: RNG seed = " << out << std::endl;
  }

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

  MesPlq(xml_out, "Observables", u);
  {
    Double w_plaq, s_plaq, t_plaq, link;
    MesPlq(u, w_plaq, s_plaq, t_plaq, link);
    std::ostringstream os;
    os << std::setprecision(12)
       << "NPR_MOMFRAC: w_plaq = " << toDouble(w_plaq)
       << "  s_plaq = "            << toDouble(s_plaq)
       << "  t_plaq = "            << toDouble(t_plaq)
       << "  link = "              << toDouble(link);
    QDPIO::cout << os.str() << std::endl;
  }

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
