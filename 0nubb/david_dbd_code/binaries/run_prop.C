#include "run_prop.h"

static const char* cname = "";

using namespace std;

CPS_START_NAMESPACE

static void run_mres_za(const QPropW& qpw, const QPropWArg& qpw_arg, const string& rdir, int traj)
{
  string mres_fn = rdir + "/mres_" + toString(qpw_arg.cg.mass) + "." + toString(traj);
  string za_fn   = rdir + "/za_" + toString(qpw_arg.cg.mass) + "." + toString(traj);

  run_mres(qpw, qpw_arg.t, mres_fn.c_str());
  run_za(qpw, qpw_arg.cg.mass, qpw_arg.t, za_fn.c_str());
}

// Temporary hack, solve a 4D volume source to collect low modes,
// useful for AMA.
//
// Note: How many times we solve the volume source depends on how many
// low modes we want to solve. Lattice properties also apply.
//
// For 300 low modes, 1 propagator using mixed solver will be good
// (depends on EigCGArg).
//
// On 48^3 2 solves are needed for 600 low modes.
static void collect_lowmodes(Lattice& lat, QPropWArg& qpw_arg, CommonArg& com_prop)
{
  const char* fname = "collect_lowmodes()"; 
  Float timer0 = dclock();

  double stop_rsd = qpw_arg.cg.stop_rsd;
  double true_rsd = qpw_arg.cg.true_rsd;

  qpw_arg.cg.stop_rsd = 1.0e-10;
  qpw_arg.cg.true_rsd = 1.0e-10;

  QPropW4DBoxArg vol_arg;
  for(int mu=0; mu<4; ++mu){
    vol_arg.box_start[mu] = 0;
    vol_arg.box_size[mu] = GJP.Sites(mu);
    vol_arg.mom[mu] = 0;
  }

  // 2 solves for 600 low modes.
  for(int i=0; i<2; ++i){
    // QPropWVolSrc(lat, &qpw_arg, &com_prop);
    QPropW4DBoxSrc qp_box(lat, &qpw_arg, &vol_arg, &com_prop);
  }

  qpw_arg.cg.stop_rsd = stop_rsd;
  qpw_arg.cg.true_rsd = true_rsd;

  Float timer1 = dclock();
  VRB.Result(cname, fname, "Finished collecting low modes: took %e seconds\n", timer1 - timer0);
}

// Point source:
// \phi(x) = \delta_{x,x0}
void run_pt_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs, 
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg, 
    int traj, bool mixed_solver, int x0[3], bool do_P, bool do_mres)
{
  const char* fname = "run_pt_props()";

  // Check boundary conditions
  if(GJP.Tbc() == BND_CND_APRD){ ERR.General(cname, fname, "Boundary condition does not match!\n"); }

  #ifdef USE_BFM
  Fbfm::use_mixed_solver = mixed_solver;
  #endif

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", exact_qpw_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  // Note: indexing is 0 --> P, 1 --> A
  for(int bc=0; bc<2; ++bc) 
  {
    // Don't bother to compute P propagators unless we explicitly say so
    if((bc == 0) && !do_P){ continue; }

    // Set boundary conditions
    GJP.Tbc( (bc == 0) ? BND_CND_PRD : BND_CND_APRD );
    lat.BondCond();

    EigCG* eig_cg = NULL;
    if(eigcg_arg) 
    {
      #ifdef USE_BFM
        eig_cg = new EigCG(eigcg_arg, mixed_solver);
        VRB.Result(cname, fname, "Collecting low modes...\n");
        collect_lowmodes(lat, exact_qpw_arg, com_prop);
        const string fn = string("../results") + (bc == 0 ? "EP" : "EA") + 
          "/eigH_" + (do_mres ? "wall_" : "twist_") + 
          toString(exact_qpw_arg.cg.mass) + "." + toString(traj);
        eig_cg -> printH(fn);
      #else
        ERR.General(cname, fname, "EigCG solver not implemented.\n");
      #endif
    }

    Float timer0 = dclock();

    // Compute exact propagators
    if(exact_props != NULL){
      for(unsigned int i=0; i<exact_locs.v.v_len; ++i){
        exact_qpw_arg.x = x0[0];
        exact_qpw_arg.y = x0[1];
        exact_qpw_arg.z = x0[2];
        exact_qpw_arg.t = exact_locs.v.v_val[i];
        VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", exact_qpw_arg.t, exact_qpw_arg.cg.stop_rsd);
        QPropWPointSrc qpw_pt(lat, &exact_qpw_arg, &com_prop);
        if(do_mres){ run_mres_za(qpw_pt, exact_qpw_arg, string("../results") + (bc == 0 ? "EP" : "EA"), traj); }
        exact_props -> add_prop(qpw_pt, exact_qpw_arg.t, bc == 0);
      }
    }

    Float timer1 = dclock();

    // Compute sloppy propagators
    if(sloppy_props != NULL)
    {
      for(int t=0; t< GJP.Sites(3); ++t){
        sloppy_qpw_arg.x = x0[0];
        sloppy_qpw_arg.y = x0[1];
        sloppy_qpw_arg.z = x0[2];
        sloppy_qpw_arg.t = t;
        VRB.Result(cname, fname, "Solving sloppy propagator at %d with stop_rsd = %e\n", sloppy_qpw_arg.t, sloppy_qpw_arg.cg.stop_rsd);
        QPropWPointSrc qpw_pt(lat, &sloppy_qpw_arg, &com_prop);
        if(do_mres){ run_mres_za(qpw_pt, sloppy_qpw_arg, string("../results") + (bc == 0 ? "P" : "A"), traj); }
        sloppy_props -> add_prop(qpw_pt, sloppy_qpw_arg.t, bc == 0);
      }
    }

    Float timer2 = dclock();

    VRB.Result(cname, fname, "Total time for  exact propagators = %e seconds\n", timer1 - timer0);
    VRB.Result(cname, fname, "Total time for sloppy propagators = %e seconds\n", timer2 - timer1);

    // Unset boundary conditions
    delete eig_cg;
    lat.BondCond();
  }

  GJP.Tbc(BND_CND_PRD);
}

// Wall source:
// \phi(x) = 1 for all spatial points on a single fixed time slice, i.e.
//    \phi(\vec{x},t) = 1 \forall \vec{x}, t = t_0
//    \phi(\vec{x},t) = 0, t \ne t_0
void run_wall_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs, 
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg, 
    int traj, bool mixed_solver, bool do_P, bool do_mres)
{
  const char* fname = "run_wall_props()";

  // Check boundary conditions
  if(GJP.Tbc() == BND_CND_APRD){ ERR.General(cname, fname, "Boundary condition does not match!\n"); }

  #ifdef USE_BFM
  Fbfm::use_mixed_solver = mixed_solver;
  #endif

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", exact_qpw_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  // Note: indexing is 0 --> P, 1 --> A
  for(int bc=0; bc<2; ++bc) 
  {
    // Don't bother to compute P propagators unless we explicitly say so
    if((bc == 0) && !do_P){ continue; }

    // Set boundary conditions
    GJP.Tbc( (bc == 0) ? BND_CND_PRD : BND_CND_APRD );
    lat.BondCond();

    EigCG* eig_cg = NULL;
    if(eigcg_arg) 
    {
      #ifdef USE_BFM
        eig_cg = new EigCG(eigcg_arg, mixed_solver);
        VRB.Result(cname, fname, "Collecting low modes...\n");
        collect_lowmodes(lat, exact_qpw_arg, com_prop);
        const string fn = string("../results") + (bc == 0 ? "EP" : "EA") + 
          "/eigH_" + (do_mres ? "wall_" : "twist_") + 
          toString(exact_qpw_arg.cg.mass) + "." + toString(traj);
        eig_cg -> printH(fn);
      #else
        ERR.General(cname, fname, "EigCG solver not implemented.\n");
      #endif
    }

    Float timer0 = dclock();

    // Compute exact propagators
    if(exact_props != NULL){
      for(unsigned int i=0; i<exact_locs.v.v_len; ++i){
        exact_qpw_arg.t = exact_locs.v.v_val[i];
        VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", exact_qpw_arg.t, exact_qpw_arg.cg.stop_rsd);
        QPropWWallSrc qpw_wall(lat, &exact_qpw_arg, &com_prop);
        if(do_mres){ run_mres_za(qpw_wall, exact_qpw_arg, string("../results") + (bc == 0 ? "EP" : "EA"), traj); }
        exact_props -> add_prop(qpw_wall, exact_qpw_arg.t, bc == 0);
      }
    }

    Float timer1 = dclock();

    // Compute sloppy propagators
    if(sloppy_props != NULL)
    {
      for(int t=0; t< GJP.Sites(3); ++t){
        sloppy_qpw_arg.t = t;
        VRB.Result(cname, fname, "Solving sloppy propagator at %d with stop_rsd = %e\n", sloppy_qpw_arg.t, sloppy_qpw_arg.cg.stop_rsd);
        QPropWWallSrc qpw_wall(lat, &sloppy_qpw_arg, &com_prop);
        if(do_mres){ run_mres_za(qpw_wall, sloppy_qpw_arg, string("../results") + (bc == 0 ? "P" : "A"), traj); }
        sloppy_props -> add_prop(qpw_wall, sloppy_qpw_arg.t, bc == 0);
      }
    }

    Float timer2 = dclock();

    VRB.Result(cname, fname, "Total time for  exact propagators = %e seconds\n", timer1 - timer0);
    VRB.Result(cname, fname, "Total time for sloppy propagators = %e seconds\n", timer2 - timer1);

    // Unset boundary conditions
    delete eig_cg;
    lat.BondCond();
  }

  GJP.Tbc(BND_CND_PRD);
}

// Momentum projected source:
// phi(x) = e^{i \vec{p} * \vec{x} } for all spatial points on a single fixed time slice
// Note: explicitly assumes we are *not* using twisted boundary conditions!!
void run_mom_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs,
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg,
    int traj, bool mixed_solver, int mom[3], bool do_P)
{
  const char* fname = "run_mom_props()";

  // Check boundary conditions in all four directions
  for(int mu=0; mu<4; ++mu) 
  {
    if(GJP.Bc(mu) == BND_CND_APRD){ ERR.General(cname, fname, "Boundary condition does not match!\n"); }
    if((mu < 3) && mom[mu]){ GJP.Bc(mu, BND_CND_APRD); }
  }

  #ifdef USE_BFM
  Fbfm::use_mixed_solver = mixed_solver;
  #endif

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", exact_qpw_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  for(int bc=0; bc<2; ++bc)
  {
    if((bc == 0) && !do_P){ continue; }

    GJP.Tbc( (bc == 0) ? BND_CND_PRD : BND_CND_APRD );
    lat.BondCond();

    EigCG* eig_cg = NULL;
    if(eigcg_arg) 
    {
      #ifdef USE_BFM
        eig_cg = new EigCG(eigcg_arg, mixed_solver);
        VRB.Result(cname, fname, "Collecting low modes...\n");
        collect_lowmodes(lat, exact_qpw_arg, com_prop);
        const string fn = string("../results") + (bc == 0 ? "EP" : "EA") + 
          "/eigH_mom_" + toString(exact_qpw_arg.cg.mass) + "." + toString(traj);
        eig_cg -> printH(fn);
      #else
        ERR.General(cname, fname, "EigCG solver not implemented.\n");
      #endif
    }
    
    Float timer0 = dclock();
    
    // Compute exact propagators
    if(exact_props != NULL){
      for(unsigned int i=0; i < exact_locs.v.v_len; ++i){
        exact_qpw_arg.t = exact_locs.v.v_val[i];
        VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", exact_qpw_arg.t, exact_qpw_arg.cg.stop_rsd);
        QPropWMomSrc qpw_mom(lat, &exact_qpw_arg, mom, &com_prop);
        exact_props -> add_prop(qpw_mom, exact_qpw_arg.t, bc == 0);
      }
    }

    Float timer1 = dclock();

    // sloppy propagators
    if(sloppy_props != NULL){
      for(int t=0; t<GJP.Sites(3); ++t){
        sloppy_qpw_arg.t = t;
        VRB.Result(cname, fname, "Solving sloppy propagator at %d with stop_rsd = %e\n", sloppy_qpw_arg.t, sloppy_qpw_arg.cg.stop_rsd);
        QPropWMomSrc qpw_mom(lat, &sloppy_qpw_arg, mom, &com_prop);
        sloppy_props -> add_prop(qpw_mom, sloppy_qpw_arg.t, bc == 0);
      }
    }

    Float timer2 = dclock();

    VRB.Result(cname, fname, "Total time for  exact propagators = %e seconds\n", timer1 - timer0);
    VRB.Result(cname, fname, "Total time for sloppy propagators = %e seconds\n", timer2 - timer1);

    delete eig_cg;
    lat.BondCond();
  }

  for(int mu=0; mu<4; ++mu){ GJP.Bc(mu, BND_CND_PRD); }
}

// Cosine momentum projected source:
// phi(x) = cos(px*x) * cos(py*y) * cos(pz*z) for all spatial points on a single fixed time slice
// Note: explicitly assumes we are *not* using twisted boundary conditions!!
void run_cos_mom_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs,
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg,
    int traj, bool mixed_solver, int mom[3], bool do_P)
{
  const char* fname = "run_cos_mom_props()";

  // Check boundary conditions in all four directions
  for(int mu=0; mu<4; ++mu) 
  {
    if(GJP.Bc(mu) == BND_CND_APRD){ ERR.General(cname, fname, "Boundary condition does not match!\n"); }
    if((mu < 3) && mom[mu]){ GJP.Bc(mu, BND_CND_APRD); }
  }

  #ifdef USE_BFM
  Fbfm::use_mixed_solver = mixed_solver;
  #endif

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", exact_qpw_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  for(int bc=0; bc<2; ++bc)
  {
    if((bc == 0) && !do_P){ continue; }

    GJP.Tbc( (bc == 0) ? BND_CND_PRD : BND_CND_APRD );
    lat.BondCond();

    EigCG* eig_cg = NULL;
    if(eigcg_arg) 
    {
      #ifdef USE_BFM
        eig_cg = new EigCG(eigcg_arg, mixed_solver);
        VRB.Result(cname, fname, "Collecting low modes...\n");
        collect_lowmodes(lat, exact_qpw_arg, com_prop);
        const string fn = string("../results") + (bc == 0 ? "EP" : "EA") + 
          "/eigH_mom_" + toString(exact_qpw_arg.cg.mass) + "." + toString(traj);
        eig_cg -> printH(fn);
      #else
        ERR.General(cname, fname, "EigCG solver not implemented.\n");
      #endif
    }
    
    Float timer0 = dclock();
    
    // Compute exact propagators
    if(exact_props != NULL){
      for(unsigned int i=0; i < exact_locs.v.v_len; ++i){
        exact_qpw_arg.t = exact_locs.v.v_val[i];
        VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", exact_qpw_arg.t, exact_qpw_arg.cg.stop_rsd);
        QPropWMomCosSrc qpw_mom(lat, &exact_qpw_arg, mom, &com_prop);
        exact_props -> add_prop(qpw_mom, exact_qpw_arg.t, bc == 0);
      }
    }

    Float timer1 = dclock();

    // sloppy propagators
    if(sloppy_props != NULL){
      for(int t=0; t<GJP.Sites(3); ++t){
        sloppy_qpw_arg.t = t;
        VRB.Result(cname, fname, "Solving sloppy propagator at %d with stop_rsd = %e\n", sloppy_qpw_arg.t, sloppy_qpw_arg.cg.stop_rsd);
        QPropWMomCosSrc qpw_mom(lat, &sloppy_qpw_arg, mom, &com_prop);
        sloppy_props -> add_prop(qpw_mom, sloppy_qpw_arg.t, bc == 0);
      }
    }

    Float timer2 = dclock();

    VRB.Result(cname, fname, "Total time for  exact propagators = %e seconds\n", timer1 - timer0);
    VRB.Result(cname, fname, "Total time for sloppy propagators = %e seconds\n", timer2 - timer1);

    delete eig_cg;
    lat.BondCond();
  }

  for(int mu=0; mu<4; ++mu){ GJP.Bc(mu, BND_CND_PRD); }
}

// Twisted cosine momentum projected source:
// phi(x) = cos(px*x) * cos(py*y) * cos(pz*z) for all spatial points on a single fixed time slice
void run_twist_cos_mom_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs,
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg,
    int traj, bool mixed_solver, int mom[3], bool do_P)
{
  const char* fname = "run_twist_cos_mom_props()";

  // Check boundary conditions in all four directions
  for(int mu=0; mu<4; ++mu) 
  {
    if(GJP.Bc(mu) == BND_CND_APRD){ ERR.General(cname, fname, "Boundary condition does not match!\n"); }
    if((mu < 3) && mom[mu]){ GJP.Bc(mu, BND_CND_APRD); }
  }

  #ifdef USE_BFM
  Fbfm::use_mixed_solver = mixed_solver;
  #endif

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", exact_qpw_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  for(int bc=0; bc<2; ++bc)
  {
    if((bc == 0) && !do_P){ continue; }

    GJP.Tbc( (bc == 0) ? BND_CND_PRD : BND_CND_APRD );
    lat.BondCond();

    EigCG* eig_cg = NULL;
    if(eigcg_arg) 
    {
      #ifdef USE_BFM
        eig_cg = new EigCG(eigcg_arg, mixed_solver);
        VRB.Result(cname, fname, "Collecting low modes...\n");
        collect_lowmodes(lat, exact_qpw_arg, com_prop);
        const string fn = string("../results") + (bc == 0 ? "EP" : "EA") + 
          "/eigH_mom_" + toString(exact_qpw_arg.cg.mass) + "." + toString(traj);
        eig_cg -> printH(fn);
      #else
        ERR.General(cname, fname, "EigCG solver not implemented.\n");
      #endif
    }
    
    Float timer0 = dclock();
    
    // Compute exact propagators
    if(exact_props != NULL){
      for(unsigned int i=0; i < exact_locs.v.v_len; ++i){
        exact_qpw_arg.t = exact_locs.v.v_val[i];
        VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", exact_qpw_arg.t, exact_qpw_arg.cg.stop_rsd);
        QPropWMomCosTwistSrc qpw_mom(lat, &exact_qpw_arg, mom, &com_prop);
        exact_props -> add_prop(qpw_mom, exact_qpw_arg.t, bc == 0);
      }
    }

    Float timer1 = dclock();

    // sloppy propagators
    if(sloppy_props != NULL){
      for(int t=0; t<GJP.Sites(3); ++t){
        sloppy_qpw_arg.t = t;
        VRB.Result(cname, fname, "Solving sloppy propagator at %d with stop_rsd = %e\n", sloppy_qpw_arg.t, sloppy_qpw_arg.cg.stop_rsd);
        QPropWMomCosTwistSrc qpw_mom(lat, &sloppy_qpw_arg, mom, &com_prop);
        sloppy_props -> add_prop(qpw_mom, sloppy_qpw_arg.t, bc == 0);
      }
    }

    Float timer2 = dclock();

    VRB.Result(cname, fname, "Total time for  exact propagators = %e seconds\n", timer1 - timer0);
    VRB.Result(cname, fname, "Total time for sloppy propagators = %e seconds\n", timer2 - timer1);

    delete eig_cg;
    lat.BondCond();
  }

  for(int mu=0; mu<4; ++mu){ GJP.Bc(mu, BND_CND_PRD); }
}

void run_box_props(PropContainer* props, cps::Lattice &lat, cps::QPropWArg &qpw_arg,
    cps::QPropW4DBoxArg& box_arg, int traj, bool mixed_solver, bool do_P)
{
  const char* fname = "run_box_props()";

  // Check boundary condition. We need this to ensure that we are
  // doing P + A and P - A, not A + P and A - P (I think it's OK to
  // skip this check, though).
  if(GJP.Tbc() == BND_CND_APRD){ ERR.General(cname, fname, "Boundary condition does not match!\n"); }

  #ifdef USE_BFM
  Fbfm::use_mixed_solver = mixed_solver;
  #endif

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", qpw_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  for(int bc=0; bc<2; ++bc) 
  {
    if((bc == 0) && !do_P){ continue; }
        
    GJP.Tbc( (bc == 0) ? BND_CND_PRD : BND_CND_APRD );
    lat.BondCond();

    Float timer0 = dclock();

    for(int t=0; t<GJP.Sites(3); ++t) 
    {
      box_arg.box_start[3] = qpw_arg.t = t;
      QPropWZ3BWallSrc qpw_z3(lat, &qpw_arg, &box_arg, &com_prop);
      props -> add_prop(qpw_z3, box_arg.box_start[3], bc == 0);
    }

    Float timer1 = dclock();

    VRB.Result(cname, fname, "Total time for box propagators: %e seconds\n", timer1 - timer0);
    
    lat.BondCond();
  }

  GJP.Tbc(BND_CND_PRD);
}

CPS_END_NAMESPACE
