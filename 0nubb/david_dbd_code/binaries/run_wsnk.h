#ifndef __RUN_WSNK_H_INCLUDED__
#define __RUN_WSNK_H_INCLUDED__

// C++
#include <vector>

// CPS
#include <alg/qpropw.h>

#include "free_prop_container.h"
#include "prop_container.h"

CPS_START_NAMESPACE

// compute wall sink propagator, with (optional) momentum
void run_wall_snk(std::vector<std::vector<WilsonMatrix>>* wsnk, const PropContainer& props, PROP_TYPE ptype, const int* p = NULL);
void run_wall_snk(std::vector<std::vector<WilsonMatrix>>* wsnk, const FreeDistributedOverlapPropContainer& props, PROP_TYPE ptype, const int* p = NULL);
void run_wall_snk(std::vector<WilsonMatrix>* wsnk, const FreePropContainer& props, const int* p = NULL);

CPS_END_NAMESPACE

#endif
