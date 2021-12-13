#include "run_meson_2pt.h"

CPS_START_NAMESPACE

static inline const WilsonMatrix& apply_op_src(WilsonMatrix& out, const WilsonMatrix& in, Operator src_op)
{
  WilsonMatrix tmp;
  switch(src_op){
    case GAMMA_0:
    case GAMMA_1:
    case GAMMA_2:
    case GAMMA_3:
      return out.glA(in, src_op - GAMMA_0);
    case GAMMA_5:
      return out = in;
    case ID:
      return out.glV(in, -5);
    case GAMMA_05:
    case GAMMA_15:
    case GAMMA_25:
    case GAMMA_35:
      return out.glV(in, src_op - GAMMA_05);
    case GAMMA_50:
    case GAMMA_51:
    case GAMMA_52:
    case GAMMA_53:
      tmp.glV(in, src_op - GAMMA_50);
      return out = -1*tmp;
    default:
      ERR.General("", "apply_op_src()", "Invalid op_id = %d\n", src_op);
      return out;
  }
}

static inline const WilsonMatrix& apply_op_snk(WilsonMatrix& out, const WilsonMatrix& in, Operator sink_op)
{
  WilsonMatrix tmp;
  switch(sink_op){
    case GAMMA_0:
    case GAMMA_1:
    case GAMMA_2:
    case GAMMA_3:
      tmp.glA(in, sink_op - GAMMA_0);
      return out = -1*tmp;
    case GAMMA_5:
      return out = in;
    case ID:
      return out.glV(in, -5);
    case GAMMA_05:
    case GAMMA_15:
    case GAMMA_25:
    case GAMMA_35:
      tmp.glV(in, sink_op - GAMMA_05);
      return out = -1*tmp;
    case GAMMA_50:
    case GAMMA_51:
    case GAMMA_52:
    case GAMMA_53:
      return out.glV(in, sink_op - GAMMA_50);
    default:
      ERR.General("", "apply_op_snk()", "Invalid op_id = %d\n", sink_op);
      return out;
  }
}

// Compute meson 2pt correlation functions with point sink
void run_meson_2pt_ptsnk(const PropContainer& propsA, const PropContainer& propsB,
    Operator sink_op, Operator src_op, const std::string& fn, PROP_TYPE ptype)
{
  const int t_scale = (ptype == PROP_PA) ? 2 : 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;
  const int lcl[4] = { GJP.XnodeSites(), GJP.YnodeSites(), GJP.ZnodeSites(), GJP.TnodeSites() };
  const int lcl_vol = lcl[0] * lcl[1] * lcl[2] * lcl[3];
  const int shift = GJP.TnodeSites() * GJP.TnodeCoor();

  FILE* fp = Fopen(fn.c_str(), "w");

  for(unsigned int k=0; k<t_size_ap; ++k) 
  {
    if(propsA.empty(k, ptype)){ continue; }
    if(propsB.empty(k, ptype)){ continue; }

    std::vector<Rcomplex> meson(t_size_ap, Rcomplex(0, 0));

    #pragma omp parallel
    {
      // threaded results
      std::vector<Rcomplex> tmp(t_size_ap, Rcomplex(0, 0));

      #pragma omp for
      for(int i=0; i<t_scale*lcl_vol; ++i) 
      {
        int x[4];
        compute_coord_ap(x, lcl, i, t_size);
        int t_glb = x[3] + shift;
            
        WilsonMatrix w = propsB(i, k, ptype);
        w.hconj();
        WilsonMatrix p[2];

        apply_op_snk(p[0], propsA(i, k, ptype), sink_op);
        apply_op_src(p[1], w, src_op);

        tmp[t_glb] += Trace(p[0], p[1]);
      } // sites

      #pragma omp critical
      for(int t=0; t<t_size_ap; ++t){
        meson[t] += tmp[(t+k)%t_size_ap];
      } // critical, for
    } // omp

    assert(GJP.Snodes() == 1);
    QMP_sum_double_array(reinterpret_cast<double*>(meson.data()), 2*t_size_ap);

    for(unsigned int t=0; t<t_size_ap; ++t){
      Fprintf(fp, "%3u %3u %17.10e %17.10e\n", k, t, meson[t].real(), meson[t].imag());
    }
  } // k

  Fclose(fp);
}

// Compute free meson 2pt correlation functions with point sink
void run_meson_2pt_ptsnk(const FreePropContainer& propsA, const FreePropContainer& propsB,
    Operator sink_op, Operator src_op, const std::string& fn)
{
  const int t_scale = 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;
  const int lcl[4] = { GJP.XnodeSites(), GJP.YnodeSites(), GJP.ZnodeSites(), GJP.TnodeSites() };
  const int lcl_vol = lcl[0] * lcl[1] * lcl[2] * lcl[3];
  const int shift = GJP.TnodeSites() * GJP.TnodeCoor();

  FILE* fp = Fopen(fn.c_str(), "w");

  std::vector<Rcomplex> meson(t_size_ap, Rcomplex(0, 0));

  #pragma omp parallel
  {
    // threaded results
    std::vector<Rcomplex> tmp(t_size_ap, Rcomplex(0, 0));

    #pragma omp for
    for(int i=0; i<t_scale*lcl_vol; ++i) 
    {
      int x[4];
      compute_coord_ap(x, lcl, i, t_size);
      int t_glb = x[3] + shift;
          
      WilsonMatrix w = propsB[i];
      w.hconj();
      WilsonMatrix p[2];

      apply_op_snk(p[0], propsA[i], sink_op);
      apply_op_src(p[1], w, src_op);

      tmp[t_glb] += Trace(p[0], p[1]);
    } // sites

    #pragma omp critical
    for(int t=0; t<t_size_ap; ++t){
      meson[t] += tmp[t];
    } // critical, for
  } // omp

  assert(GJP.Snodes() == 1);
  QMP_sum_double_array(reinterpret_cast<double*>(meson.data()), 2*t_size_ap);

  for(unsigned int t=0; t<t_size_ap; ++t){
    Fprintf(fp, "%3u %17.10e %17.10e\n", t, meson[t].real(), meson[t].imag());
  }

  Fclose(fp);
}

// Compute meson 2pt correlation functions with wall sink
void run_meson_2pt_wallsnk(const PropContainer& propsA, const PropContainer& propsB,
    Operator sink_op, Operator src_op, const std::string& fn, PROP_TYPE ptype)
{
  const int t_scale = (ptype == PROP_PA) ? 2 : 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;

  std::vector<std::vector<WilsonMatrix>> Awsnk, Bwsnk;
  run_wall_snk(&Awsnk, propsA, ptype);
  run_wall_snk(&Bwsnk, propsB, ptype);

  FILE* fp = Fopen(fn.c_str(), "w");

  for(unsigned int src=0; src<t_size_ap; ++src) 
  {
    if(Awsnk[src].empty()){ continue; }
    if(Bwsnk[src].empty()){ continue; }

    std::vector<Rcomplex> meson(t_size_ap, Rcomplex(0, 0));
    
    for(unsigned int dt=0; dt<t_size_ap; ++dt) 
    {
      unsigned int snk = (src + dt) % t_size_ap;

      WilsonMatrix w = Bwsnk[src][snk];
      w.hconj();

      WilsonMatrix p[2];
      apply_op_snk(p[0], Awsnk[src][snk], sink_op);
      apply_op_src(p[1], w, src_op);
            
      meson[dt] += Trace(p[0], p[1]);
    } // dt

    for(unsigned int t=0; t<t_size_ap; ++t){
      Fprintf(fp, "%3u %3u %17.10e %17.10e\n", src, t, meson[t].real(), meson[t].imag());
    }
  } // src
    
  Fclose(fp);
}

// Compute free meson 2pt correlation functions with wall sink
void run_meson_2pt_wallsnk(const FreePropContainer& propsA, const FreePropContainer& propsB,
    Operator sink_op, Operator src_op, const std::string& fn)
{
  const int t_scale = 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;

  std::vector<WilsonMatrix> Awsnk, Bwsnk;
  run_wall_snk(&Awsnk, propsA);
  run_wall_snk(&Bwsnk, propsB);

  FILE* fp = Fopen(fn.c_str(), "w");

  std::vector<Rcomplex> meson(t_size_ap, Rcomplex(0, 0));
  
  for(unsigned int dt=0; dt<t_size_ap; ++dt) 
  {
    WilsonMatrix w = Bwsnk[dt];
    w.hconj();

    WilsonMatrix p[2];
    apply_op_snk(p[0], Awsnk[dt], sink_op);
    apply_op_src(p[1], w, src_op);
          
    meson[dt] += Trace(p[0], p[1]);
  } // dt

  for(unsigned int t=0; t<t_size_ap; ++t){
    Fprintf(fp, "%3u %17.10e %17.10e\n", t, meson[t].real(), meson[t].imag());
  }
    
  Fclose(fp);
}

// Compute meson 2pt correlation functions with wall sink
void run_meson_2pt_wallsnk(const FreeDistributedOverlapPropContainer& propsA, 
    const FreeDistributedOverlapPropContainer& propsB,
    Operator sink_op, Operator src_op, const std::string& fn, PROP_TYPE ptype)
{
  const int t_scale = (ptype == PROP_PA) ? 2 : 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;

  std::vector<std::vector<WilsonMatrix>> Awsnk, Bwsnk;
  run_wall_snk(&Awsnk, propsA, ptype);
  run_wall_snk(&Bwsnk, propsB, ptype);

  FILE* fp = Fopen(fn.c_str(), "w");

  for(unsigned int src=0; src<t_size_ap; ++src) 
  {
    if(Awsnk[src].empty()){ continue; }
    if(Bwsnk[src].empty()){ continue; }

    std::vector<Rcomplex> meson(t_size_ap, Rcomplex(0, 0));
    
    for(unsigned int dt=0; dt<t_size_ap; ++dt) 
    {
      unsigned int snk = (src + dt) % t_size_ap;

      WilsonMatrix w = Bwsnk[src][snk];
      w.hconj();

      WilsonMatrix p[2];
      apply_op_snk(p[0], Awsnk[src][snk], sink_op);
      apply_op_src(p[1], w, src_op);
            
      meson[dt] += Trace(p[0], p[1]);
    } // dt

    for(unsigned int t=0; t<t_size_ap; ++t){
      Fprintf(fp, "%3u %3u %17.10e %17.10e\n", src, t, meson[t].real(), meson[t].imag());
    }
  } // src
    
  Fclose(fp);
}

CPS_END_NAMESPACE
