#ifndef __RUN_PROP_H_INCLUDED__
#define __RUN_PROP_H_INCLUDED__

// C
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// C++
#include <cassert>
#include <string>
#include <vector>

// CPS
#include <alg/alg_int.h>
#include <alg/array_arg.h>
#include <alg/eigcg_arg.h>
#include <alg/qpropw.h>
#include <alg/qpropw_arg.h>
#include <util/error.h>
#include <util/gjp.h>
#include <util/lattice.h>
#ifdef USE_BFM
#include <util/lattice/fbfm.h>
#endif
#include <util/qcdio.h>
#include <util/time_cps.h>
#include <util/verbose.h>

#include "eigcg.h"
#include "fft.h"
#include "free_prop_container.h"
#include "prop_container.h"
#include "run_mres_2pt.h"
#include "utils.h"

namespace cps {
  class Lattice;
};

CPS_START_NAMESPACE

void run_pt_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs, 
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg, 
    int traj, bool mixed_solver, int x0[3], bool do_P = false, bool do_mres = true);

void run_wall_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs, 
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg, 
    int traj, bool mixed_solver, bool do_P = false, bool do_mres = true);

void run_mom_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs,
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg,
    int traj, bool mixed_solver, int mom[3], bool do_P = false);

void run_cos_mom_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs,
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg,
    int traj, bool mixed_solver, int mom[3], bool do_P = false);

void run_twist_cos_mom_props(PropContainer* exact_props, PropContainer* sloppy_props, cps::IntArray& exact_locs,
    cps::Lattice& lat, cps::QPropWArg& exact_qpw_arg, cps::QPropWArg& sloppy_qpw_arg, cps::EigCGArg* eigcg_arg,
    int traj, bool mixed_solver, int mom[3], bool do_P = false);

void run_box_props(PropContainer* props, cps::Lattice &lat, cps::QPropWArg &qpw_arg,
    cps::QPropW4DBoxArg& box_arg, int traj, bool mixed_solver, bool do_P = false);

void run_free_prop(FreePropContainer& props, const int* x0 = NULL);

CPS_END_NAMESPACE

#endif
