#include "run_0vbb_3pt.h"

CPS_START_NAMESPACE

void run_0vbb_exc_3pt(const PropContainer& lprops, const std::string& fn, PROP_TYPE ptype)
{
  const char* fname = "run_0vbb_exc_3pt()";
  const int Ndiag(16);

  const int t_scale = (ptype == PROP_PA) ? 2 : 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;
  const int lcl[4] = { GJP.XnodeSites(), GJP.YnodeSites(), GJP.ZnodeSites(), GJP.TnodeSites() };
  const int lcl_vol = lcl[0] * lcl[1] * lcl[2] * lcl[3];
  const int shift = GJP.TnodeSites() * GJP.TnodeCoor();

  // Compute wall sink propagators
  std::vector<std::vector<WilsonMatrix>> lwsnk;
  run_wall_snk(&lwsnk, lprops, ptype);

  // Compute pieces for gamma matrix insertions
  GammaContainer GC;

  FILE* fp = Fopen(fn.c_str(), "w");

  for(unsigned int sep=0; sep<t_size_ap; ++sep){
  for(unsigned int tm=0; tm<t_size; ++tm){

    const unsigned int t0 = (tm + sep) % t_size_ap;

    if(lprops.empty(tm, ptype)){ continue; }
    if(lwsnk[t0].empty()){ continue; }
    if(lprops.empty(t0, ptype)){ continue; }

    // 4 operators x 2 contractions x {V,A}
    std::vector<std::vector<Rcomplex>> C(Ndiag, std::vector<Rcomplex>(t_size_ap, Rcomplex(0,0)));

    #pragma omp parallel
    {
      // threaded results
      std::vector<std::vector<Rcomplex>> tmp(Ndiag, std::vector<Rcomplex>(t_size_ap, Rcomplex(0,0)));

      #pragma omp for
      for(int i=0; i<t_scale*lcl_vol; ++i)
      {
        int x[4];
        compute_coord_ap(x, lcl, i, t_size);
        int t_glb = x[3] + shift;

        const WilsonMatrix& S_tm_to_x  = lprops(i, tm, ptype);
        const WilsonMatrix& S_tm_to_t0 = lwsnk[tm][t0];
        const WilsonMatrix& S_t0_to_x  = lprops(i, t0, ptype);
        const WilsonMatrix& S_t0_to_tm = lwsnk[t0][tm];

        WilsonMatrix S_tm_to_x_dag = hconj(S_tm_to_x);
        WilsonMatrix S_t0_to_x_dag = hconj(S_t0_to_x);

        for(int mu=0; mu<4; mu++){
          tmp[mu][t_glb]    += Trace( GC.Gmu(mu) * S_tm_to_x * GC.G5() * S_t0_to_tm * S_t0_to_x_dag );            // V, contraction 1
          tmp[mu+4][t_glb]  += Trace( GC.Gmu(mu) * GC.G5() * S_tm_to_x * GC.G5() * S_t0_to_tm * S_t0_to_x_dag );  // A, contraction 1
          tmp[mu+8][t_glb]  += Trace( GC.Gmu(mu) * S_t0_to_x * GC.G5() * S_tm_to_t0 * S_tm_to_x_dag );            // V, contraction 2
          tmp[mu+12][t_glb] += Trace( GC.Gmu(mu) * GC.G5() * S_t0_to_x * GC.G5() * S_tm_to_t0 * S_tm_to_x_dag );  // A, contraction 2
        }
      } // sites

      #pragma omp critical
      for(int didx=0; didx<Ndiag; ++didx){
      for(int t=0; t<t_size_ap; ++t){
        C[didx][t] += tmp[didx][(t+tm)%t_size_ap];
      }} // critical, for
    }//omp

    assert(GJP.Snodes() == 1);
    for(int didx=0; didx<Ndiag; ++didx){ QMP_sum_double_array(reinterpret_cast<double*>(C[didx].data()), 2*t_size_ap); }

    for(unsigned int t=0; t<t_size_ap; ++t)
    {
      Fprintf(fp, "%3u %3u %3u", sep, tm, t);
      for(int didx=0; didx<Ndiag; ++didx){ Fprintf(fp, " %17.10e %17.10e", real(C[didx][t]), imag(C[didx][t])); }
      Fprintf(fp, "\n");
    }

  }} // tm, sep

  Fclose(fp);
}

void run_0vbb_4quark_3pt(const PropContainer& lprops, const std::string& fn, PROP_TYPE ptype)
{
  const char* fname = "run_0vbb_4quark_3pt()";
  const int Ndiag(24); // { SS, PP, VV, VA, AV, AA } x ( color mixed/unmixed ) x ( 2 contractions )

  const int t_scale = (ptype == PROP_PA) ? 2 : 1;    // 1 for our case of PROP_A
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;             // I'd assume this is just the T variable
  const int lcl[4] = { GJP.XnodeSites(), GJP.YnodeSites(), GJP.ZnodeSites(), GJP.TnodeSites() };
  const int lcl_vol = lcl[0] * lcl[1] * lcl[2] * lcl[3];
  const int shift = GJP.TnodeSites() * GJP.TnodeCoor();

  // Compute pieces for gamma matrix insertions
  GammaContainer GC;

  FILE* fp = Fopen(fn.c_str(), "w");

  for(unsigned int sep=0; sep<t_size_ap; ++sep){
  for(unsigned int tm=0; tm<t_size; ++tm){

    const unsigned int tp = (tm + sep) % t_size_ap;

    if(lprops.empty(tm, ptype)){ continue; }
    if(lprops.empty(tp, ptype)){ continue; }

    std::vector<std::vector<Rcomplex>> C(Ndiag, std::vector<Rcomplex>(t_size_ap, Rcomplex(0,0)));

    #pragma omp parallel
    {
      // threaded results
      std::vector<std::vector<Rcomplex>> tmp(Ndiag, std::vector<Rcomplex>(t_size_ap, Rcomplex(0,0)));

      #pragma omp for
      for(int i=0; i<t_scale*lcl_vol; ++i)    // I think i indexes the entire volume-- it's a vectorized index for every point on the lattice
      {
        int x[4];         // x here enumerates each point in the volume-- compute_coord_ap converts the vectorized i index into an actual 4-vector x
        compute_coord_ap(x, lcl, i, t_size);
        int t_glb = x[3] + shift;

        const WilsonMatrix& S_tm_to_x  = lprops(i, tm, ptype);    // Gets propagator computed with wall source (\vec p = \vec 0, t_-)
        const WilsonMatrix& S_tp_to_x  = lprops(i, tp, ptype);    // Gets propagator computed with wall source (\vec p = \vec 0, t_+)

        WilsonMatrix SSdag_tm_to_x = S_tm_to_x * hconj(S_tm_to_x);
        WilsonMatrix SSdag_tp_to_x = S_tp_to_x * hconj(S_tp_to_x);

        SpinMatrix SSdag_tm_to_x_s = ColorTrace(SSdag_tm_to_x);
        SpinMatrix SSdag_tp_to_x_s = ColorTrace(SSdag_tp_to_x);

        // SS
        // NOTE THAT THE SUM OVER EVERYTHING AT THE SAME t_glb-- THIS IS USING A POINT SINK AT T_X
        tmp[0][t_glb] += Trace( GC.G5() * SSdag_tm_to_x ) * Trace( GC.G5() * SSdag_tp_to_x );            // color unmixed, contraction 1
        tmp[1][t_glb] += Trace( GC.G5() * SSdag_tm_to_x * GC.G5() * SSdag_tp_to_x );                     // color unmixed, contraction 2
        tmp[2][t_glb] += Trace( SpinTr( GC.G5() * SSdag_tm_to_x ) * SpinTr( GC.G5() * SSdag_tp_to_x ) ); // color mixed, contraction 1
        tmp[3][t_glb] += Trace( ExtractColor(GC.G5(), 0, 0) * SSdag_tm_to_x_s *
                                  ExtractColor(GC.G5(), 0, 0) * SSdag_tp_to_x_s );                       // color mixed, contraction 2

        // PP
        tmp[4][t_glb] += Trace( SSdag_tm_to_x ) * Trace( SSdag_tp_to_x );
        tmp[5][t_glb] += Trace( SSdag_tm_to_x * SSdag_tp_to_x );
        tmp[6][t_glb] += Trace( SpinTr( SSdag_tm_to_x ) * SpinTr( SSdag_tp_to_x ) );
        tmp[7][t_glb] += Trace( SSdag_tm_to_x_s * SSdag_tp_to_x_s );

        for(int mu=0; mu<4; mu++)
        {
          WilsonMatrix GV = GC.G5() * GC.Gmu(mu);
          WilsonMatrix GA = GC.G5() * GC.Gmu(mu) * GC.G5();

          SpinMatrix GVs = ExtractColor(GV, 0, 0);
          SpinMatrix GAs = ExtractColor(GA, 0, 0);

          // VV
          tmp[8][t_glb]  += Trace( GV * SSdag_tm_to_x ) * Trace( GV * SSdag_tp_to_x );
          tmp[9][t_glb]  += Trace( GV * SSdag_tm_to_x * GV * SSdag_tp_to_x );
          tmp[10][t_glb] += Trace( SpinTr( GV * SSdag_tm_to_x ) * SpinTr( GV * SSdag_tp_to_x ) );
          tmp[11][t_glb] += Trace( GVs * SSdag_tm_to_x_s * GVs * SSdag_tp_to_x_s );

          // VA
          tmp[12][t_glb]  += Trace( GV * SSdag_tm_to_x ) * Trace( GA * SSdag_tp_to_x );
          tmp[13][t_glb]  += Trace( GV * SSdag_tm_to_x * GA * SSdag_tp_to_x );
          tmp[14][t_glb] += Trace( SpinTr( GV * SSdag_tm_to_x ) * SpinTr( GA * SSdag_tp_to_x ) );
          tmp[15][t_glb] += Trace( GVs * SSdag_tm_to_x_s * GAs * SSdag_tp_to_x_s );

          // AV
          tmp[16][t_glb]  += Trace( GA * SSdag_tm_to_x ) * Trace( GV * SSdag_tp_to_x );
          tmp[17][t_glb]  += Trace( GA * SSdag_tm_to_x * GV * SSdag_tp_to_x );
          tmp[18][t_glb] += Trace( SpinTr( GA * SSdag_tm_to_x ) * SpinTr( GV * SSdag_tp_to_x ) );
          tmp[19][t_glb] += Trace( GAs * SSdag_tm_to_x_s * GVs * SSdag_tp_to_x_s );

          // AA
          tmp[20][t_glb]  += Trace( GA * SSdag_tm_to_x ) * Trace( GA * SSdag_tp_to_x );
          tmp[21][t_glb]  += Trace( GA * SSdag_tm_to_x * GA * SSdag_tp_to_x );
          tmp[22][t_glb] += Trace( SpinTr( GA * SSdag_tm_to_x ) * SpinTr( GA * SSdag_tp_to_x ) );
          tmp[23][t_glb] += Trace( GAs * SSdag_tm_to_x_s * GAs * SSdag_tp_to_x_s );
        }
      } // sites

      #pragma omp critical
      for(int didx=0; didx<Ndiag; ++didx){
      for(int t=0; t<t_size_ap; ++t){
        C[didx][t] += tmp[didx][(t+tm)%t_size_ap];
      }} // critical, for
    }//omp

    assert(GJP.Snodes() == 1);
    for(int didx=0; didx<Ndiag; ++didx){ QMP_sum_double_array(reinterpret_cast<double*>(C[didx].data()), 2*t_size_ap); }

    for(unsigned int t=0; t<t_size_ap; ++t)
    {
      Fprintf(fp, "%3u %3u %3u", sep, tm, t);
      for(int didx=0; didx<Ndiag; ++didx){ Fprintf(fp, " %17.10e %17.10e", C[didx][t].real(), C[didx][t].imag()); }
      Fprintf(fp, "\n");
    }

  }} // tm, sep

  Fclose(fp);
}

CPS_END_NAMESPACE
