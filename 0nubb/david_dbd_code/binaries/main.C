// C
#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// C++
#include <cassert>
#include <string>
#include <vector>

// QMP
#include <qmp.h>

// CPS
#include <alg/alg_fix_gauge.h>
#include <alg/alg_meas.h>
#include <alg/array_arg.h>
#include <alg/common_arg.h>
#include <alg/do_arg.h>
#include <alg/eigcg_arg.h>
#include <alg/meas_arg.h>
#include <alg/qpropw_arg.h>
#include <util/command_line.h>
#include <util/eigen_container.h>
#include <util/error.h>
#include <util/gjp.h>
#include <util/lat_cont.h>
#include <util/lattice.h>
#ifdef USE_BFM
#include <util/lattice/fbfm.h>
#endif
#include <util/qcdio.h>
#include <util/ReadLatticePar.h>
#include <util/time_cps.h>
#include <util/verbose.h>

// Measurement package
#include "eigcg.h"
#include "free_prop_container.h"
#include "prop_container.h"
#include "run_0vbb_3pt.h"
#include "run_0vbb_4pt.h"
#include "run_kl3_3pt.h"
#include "run_meson_2pt.h"
#include "run_mres_2pt.h"
#include "run_prop.h"
#include "twisted_bc.h"
#include "utils.h"

static const char* cname = "";

USING_NAMESPACE_CPS
using namespace std;

CommonArg common_arg;
DoArg do_arg;
DoArgExt do_ext;
// EigCGArg l_eigcg_arg;
FixGaugeArg fix_gauge_arg;
MeasArg meas_arg;
QPropWArg lqpropw_arg;
QPropWArg sqpropw_arg;
// EigCGArg s_eigcg_arg;

//--------------------------------------------------------------------
// DJM: Setup for QUDA interface

#ifdef USE_QUDA

std::vector<EigenCache*> cps::EigenCacheList(0);

// Search contents that match to arguments, return 0 if not found
EigenCache* cps::EigenCacheListSearch(char* fname_root_bc, int neig)
{
  EigenCache* ecache = 0;

  for(int i=0; i<EigenCacheList.size(); i++){
    if(EigenCacheList[i]->is_cached(fname_root_bc, neig)){
      ecache = EigenCacheList[i];
    }
  }

  return ecache;
}

// Cleanup list EigenCache.
// It also destroys contents pointed to by the elements.
void cps::EigenCacheListCleanup(void)
{
  for(size_t i=0; i<EigenCacheList.size(); i++){
    EigenCacheList[i] -> dealloc();
  }
  EigenCacheList.clear();
}

#endif

//--------------------------------------------------------------------

//--------------------------------------------------------------------
// DJM: Setup for BFM interface

#ifdef USE_BFM

void init_bfm(int* argc, char*** argv)
{
  const char* fname = "init_bfm()";

  QDP::QDP_initialize(argc, argv);
  multi1d<int> nrow(Nd);

  for(int i=0; i<Nd; ++i){ nrow[i] = GJP.Sites(i); }

  Layout::setLattSize(nrow);
  Layout::create();

  Fbfm::use_mixed_solver = true;

  bfmarg::Threads(GJP.Nthreads());
  bfmarg::Reproduce(0);
  bfmarg::ReproduceChecksum(0);
  bfmarg::ReproduceMasterCheck(0);
  bfmarg::Verbose(0);

  double alpha = do_ext.mobius_b_coeff + do_ext.mobius_c_coeff;
  Fbfm::arg_map[lqpropw_arg.cg.mass].ScaledShamirCayleyTanh(lqpropw_arg.cg.mass, 
      do_arg.dwf_height, do_arg.s_sites, alpha);
  Fbfm::arg_map[sqpropw_arg.cg.mass].ScaledShamirCayleyTanh(sqpropw_arg.cg.mass, 
      do_arg.dwf_height, do_arg.s_sites, alpha);

  VRB.Result(cname, fname, "init_bfm finished successfully\n");
}

#endif

//--------------------------------------------------------------------

//--------------------------------------------------------------------
// DJM: Setup routines common to both interfaces

#define decode_vml(arg_name)  do{                        \
  if ( ! arg_name.Decode(#arg_name".vml", #arg_name) )   \
  ERR.General(cname, fname, "Bad " #arg_name ".vml.\n"); \
} while(0)  

inline int Chdir(const char* dir)
{
  const char* fname = "Chdir(char*)";

  if(chdir(dir) != 0){
    ERR.General("", fname, "Changing to directory %s failed.\n", dir);
  }

  return 0;
}

void decode_vml_all(void)
{
  const char *fname = "decode_vml_all()";

  decode_vml(do_arg);
  decode_vml(meas_arg);
  decode_vml(lqpropw_arg);
  decode_vml(sqpropw_arg);
  decode_vml(fix_gauge_arg);
  // decode_vml(l_eigcg_arg);
  // decode_vml(s_eigcg_arg);
}

void load_checkpoint(int traj)
{
  const char *fname = "load_checkpoint()";

  char lat_file[256];
  GnoneFnone lat;

  sprintf(lat_file, "%s.%d", meas_arg.GaugeStem, traj);
  QioArg rd_arg(lat_file, 0.001);
  rd_arg.ConcurIONumber = meas_arg.IOconcurrency;
  ReadLatticeParallel rl;
  rl.read(lat,rd_arg);
  if(!rl.good()) ERR.General(cname,fname,"Failed read lattice %s\n",lat_file);
}

void setup(int argc, char** argv)
{
  const char *fname = "setup()";

  Start(&argc, &argv);

  CommandLine::is(argc, argv);
  Chdir(CommandLine::arg());

  #ifdef USE_QUDA
  if(!QudaParam.Decode(argv[2], "QudaParam")){ ERR.General("", fname, "Bum quda_arg\n"); }
  VRB.Result("", fname, "device %d\n", QudaParam.device);
  #endif

  decode_vml_all();
  VRB.Result("", fname, "Read VML files successfully.\n");

  GJP.Initialize(do_arg);
  GJP.InitializeExt(do_ext);
  LRG.setSerial();
  LRG.Initialize();

  int nthreads(64);
  if(getenv("OMP_NUM_THREADS")){ 
    nthreads = atoi(getenv("OMP_NUM_THREADS")); 
  } else {
    VRB.Result("", fname, "WARNING: using default %d OMP threads.\n", nthreads);
  }
  GJP.SetNthreads(nthreads);

  #ifdef USE_BFM 
  init_bfm(&argc, &argv);
  #endif
}

//--------------------------------------------------------------------

// stw: twisting angle of the strange quark (connecting the operator and the kaon).
// ltw: twisting angle of the light quark (connecting the operator and the pion).
int main(int argc, char** argv)
{
  const char* fname = "main()";

  // Seed rng
  srand48(time(NULL));

  setup(argc, argv);

  int traj = meas_arg.TrajStart;
  int ntraj = (meas_arg.TrajLessThanLimit - traj) / meas_arg.TrajIncrement + 1;

  for(int conf=0; conf<ntraj; conf++)
  {
    VRB.Result(cname, fname, "Beginning trajectory %d\n", traj);

    Lattice& lat = LatticeFactory::Create(meas_arg.Fermion, meas_arg.Gluon);

    Float dtime0 = dclock();

    //////////////////////////////////////////////////////////////////////
    // 1. Gauge fixing

    VRB.Result(cname, fname, "Starting gauge fixing:\n");
    
    AlgFixGauge fg(lat, &common_arg, &fix_gauge_arg);
    fg.run();

    Float dtime1 = dclock();
    
    //////////////////////////////////////////////////////////////////////
    // 2. Propagator generation

    VRB.Result(cname, fname, "Starting wall source propagator inversions:\n");

    // l untwisted
    VRB.Result(cname, fname, "Computing light untwisted wall source propagators...\n");
    
    PropContainer lprops(PropContainer::DOUBLE);
    cps::IntArray tmp; // Just to make run_wall_props() happy...
    run_wall_props(NULL, &lprops, tmp, lat, lqpropw_arg, lqpropw_arg, NULL, traj, true);
    
    Float dtime2 = dclock();

    // Neutrino propagators
    VRB.Result(cname, fname, "Computing neutrino propagators...\n");
    
    FreeMasslessScalarContinuumProp nu_prop(cutoff_type::gaussian);

    Float dtime3 = dclock();

    VRB.Result(cname, fname, "Starting contractions:\n");

    const string trajs = string(".") + toString(traj);

    //////////////////////////////////////////////////////////////////////
    // 3. Two-point functions

    VRB.Result(cname, fname, "Computing two-point functions...\n");
    
    run_meson_2pt_ptsnk  (lprops, lprops, GAMMA_5,  GAMMA_5, "../resultsA/pion-00WP" + trajs, PROP_A);
    run_meson_2pt_wallsnk(lprops, lprops, GAMMA_5,  GAMMA_5, "../resultsA/pion-00WW" + trajs, PROP_A);
    run_meson_2pt_ptsnk  (lprops, lprops, GAMMA_35, GAMMA_5, "../resultsA/fp-00WP"   + trajs, PROP_A);
    run_meson_2pt_wallsnk(lprops, lprops, GAMMA_35, GAMMA_5, "../resultsA/fp-00WW"   + trajs, PROP_A);
    run_kl3_3pt(lprops, lprops, lprops, "../resultsA/zpa-00" + trajs, PROP_A);

    Float dtime4 = dclock();

    //////////////////////////////////////////////////////////////////////
    // 4. Intermediate state three-point functions
    
    VRB.Result(cname, fname, "Computing three-point functions for 0vbb intermediate state matrix elements...\n");
    
    run_0vbb_exc_3pt(lprops, "../resultsA/pion_0vbb_exc" + trajs, PROP_A);

    Float dtime5 = dclock();
    
    //////////////////////////////////////////////////////////////////////
    // 5. Short distance three-point functions

    VRB.Result(cname, fname, "Computing three-point functions for short-distance 0vbb...\n");
    
    run_0vbb_4quark_3pt(lprops, "../resultsA/pion_0vbb_4quark" + trajs, PROP_A);

    Float dtime6 = dclock();

    //////////////////////////////////////////////////////////////////////
    // 6. Long distance four-point function

    VRB.Result(cname, fname, "Computing four-point function for long-distance 0vbb...\n");

    // Type 1 contractions
    double me(0.5109989461 / 1784.4); // electron mass in 24I lattice units
    const int dt_min_wall(6);
    const int sep_min(12);
    const int sep_max(24);
    const int dsep(1);
    run_0vbb_4pt_pion_ds_type1_double_prec(lprops, PROP_A, nu_prop, sep_min, sep_max, dsep, dt_min_wall, me, 0, 0, 
                                "../resultsA.tavg/pion_0vbb_ds_type1" + trajs, "../work/fftw_wisdom_ds_type1.dat");
    
    Float dtime7 = dclock();

    // Type 2 contractions
    const int N_time_translations(4);
    run_0vbb_4pt_pion_ds_type2_double_prec(lprops, PROP_A, nu_prop, N_time_translations, sep_min, sep_max, dsep, dt_min_wall, me, 0, 0, 
                                "../resultsA.tavg/pion_0vbb_ds_type2" + trajs, "../work/fftw_wisdom_ds_type2.dat");
    
    Float dtime8 = dclock();

    VRB.Result(cname, fname, "\n");
    VRB.Result(cname, fname, "---------------------------- Timing Summary ---------------------------\n");
    VRB.Result(cname, fname, "1. Gauge fixing:                                    %17.10e s\n", dtime1 - dtime0);
    VRB.Result(cname, fname, "2. Wall source light quark propagator inversions:   %17.10e s\n", dtime2 - dtime1);
    VRB.Result(cname, fname, "3. Neutrino propagators:                            %17.10e s\n", dtime3 - dtime2);
    VRB.Result(cname, fname, "4. Two-point functions:                             %17.10e s\n", dtime4 - dtime3);
    VRB.Result(cname, fname, "5. Intermediate state three-point functions:        %17.10e s\n", dtime5 - dtime4);
    VRB.Result(cname, fname, "6. Short-distance three-point functions:            %17.10e s\n", dtime6 - dtime5);
    VRB.Result(cname, fname, "7. Long-distance four-point function (type 1):      %17.10e s\n", dtime7 - dtime6);
    VRB.Result(cname, fname, "8. Long-distance four-point function (type 2):      %17.10e s\n", dtime8 - dtime7);
    VRB.Result(cname, fname, "-----------------------------------------------------------------------\n");
    VRB.Result(cname, fname, "Total:                                              %17.10e s\n", dtime8 - dtime0);
    VRB.Result(cname, fname, "-----------------------------------------------------------------------\n");
    VRB.Result(cname, fname, "\n");

    traj += meas_arg.TrajIncrement;
  }

  #ifdef USE_QUDA
  EigenCacheListCleanup();
  #endif

  VRB.Result(cname, fname, "Propgram ended normally.\n");
  End();
}
