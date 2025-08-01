#include "run_2pion_2pt.h"

CPS_START_NAMESPACE

void run_2pionDC_2pt_wallsnk(const PropContainer& uprops, const PropContainer& dprops, const std::string& fn, PROP_TYPE ptype, const int pd[3])
{
  const int t_scale = (ptype == PROP_PA) ? 2 : 1;
  const int t_size = GJP.TnodeSites() * GJP.Tnodes();
  const int t_size_ap = t_scale * t_size;

  std::vector<std::vector<WilsonMatrix>> us, dsp, dsm;

  run_wall_snk(&us, uprops, ptype);
  run_wall_snk(&dsp, dprops, ptype, pd);

  int pdm[3] = { -pd[0], -pd[1], -pd[2] };
  run_wall_snk(&dsm, dprops, ptype, pdm);

  FILE* fp = Fopen(fn.c_str(), "w");

  for(unsigned int src=0; src<t_size; ++src) 
  {
    if( us[src].empty()){ continue; }
    if(dsp[src].empty()){ continue; }
    if(dsm[src].empty()){ continue; }

    std::vector<Rcomplex> ddiag(t_size_ap, Rcomplex(0, 0));
    std::vector<Rcomplex> cdiag(t_size_ap, Rcomplex(0, 0));

    for(unsigned int dt=0; dt<t_size_ap; ++dt) 
    {
      unsigned int snk = (src + dt) % t_size_ap;

      WilsonMatrix w[2];

      w[0] = dsp[src][snk];
      w[0].hconj();
      w[0] *= us[src][snk];

      w[1] = dsm[src][snk];
      w[1].hconj();
      w[1] *= us[src][snk];

      ddiag[dt] = w[0].Trace() * w[1].Trace();
      cdiag[dt] = Trace(w[0], w[1]);
    } // dt

    for(unsigned int t=0; t<t_size_ap; ++t){
      Fprintf(fp, "%3u %3u %17.10e %17.10e %17.10e %17.10e\n", src, t, real(ddiag[t]), imag(ddiag[t]), real(cdiag[t]), imag(cdiag[t]));
    }
  } // src
  
  Fclose(fp);
}

CPS_END_NAMESPACE
