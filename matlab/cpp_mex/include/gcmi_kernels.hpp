#pragma once

#include <vector>

#include "gcmi_mex_utils.hpp"

namespace gcmi {

std::vector<double> copnorm_slice_kernel(
    const double* x,
    mwSize nTrials,
    mwSize nPages,
    mwSize threadCount);

void info_cc_slice(
    const double* x,
    mwSize nTrials,
    mwSize xDim,
    mwSize nPages,
    const double* y,
    mwSize yDim,
    mwSize threadCount,
    double* output);

std::vector<double> info_cd_slice(
    const double* x,
    mwSize xDim,
    mwSize nTrials,
    mwSize nPages,
    const std::vector<mwSignedIndex>& labels,
    mwSize nClasses,
    mwSize threadCount);

}  // namespace gcmi
