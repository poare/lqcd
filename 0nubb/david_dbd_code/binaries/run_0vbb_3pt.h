#ifndef __RUN_0VBB_3PT_H_INCLUDED__
#define __RUN_0VBB_3PT_H_INCLUDED__

// C++
#include <string>
#include <vector>

// CPS
#include <alg/qpropw.h>
#include <util/qcdio.h>

#include "color_matrix.h"
#include "gamma_container.h"
#include "prop_container.h"
#include "run_wsnk.h"
#include "utils.h"

CPS_START_NAMESPACE

// compute first excited state ME for \pi^{-} --> \pi^{+} e e
void run_0vbb_exc_3pt(const PropContainer& lprops, const std::string& fn, PROP_TYPE ptype);

// Compute ME's of local four-quark operators for renormalization
void run_0vbb_4quark_3pt(const PropContainer& lprops, const std::string& fn, PROP_TYPE ptype);

CPS_END_NAMESPACE

#endif
