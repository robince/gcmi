#include "gcmi_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <omp.h>

namespace gcmi {

namespace {

inline double ndtri(double p) {
    // Peter J. Acklam's inverse-normal approximation, mirrored from the
    // Python/Numba batch kernel for consistent batch copula normalization.
    const double a0 = -3.969683028665376e01;
    const double a1 = 2.209460984245205e02;
    const double a2 = -2.759285104469687e02;
    const double a3 = 1.383577518672690e02;
    const double a4 = -3.066479806614716e01;
    const double a5 = 2.506628277459239e00;

    const double b0 = -5.447609879822406e01;
    const double b1 = 1.615858368580409e02;
    const double b2 = -1.556989798598866e02;
    const double b3 = 6.680131188771972e01;
    const double b4 = -1.328068155288572e01;

    const double c0 = -7.784894002430293e-03;
    const double c1 = -3.223964580411365e-01;
    const double c2 = -2.400758277161838e00;
    const double c3 = -2.549732539343734e00;
    const double c4 = 4.374664141464968e00;
    const double c5 = 2.938163982698783e00;

    const double d0 = 7.784695709041462e-03;
    const double d1 = 3.224671290700398e-01;
    const double d2 = 2.445134137142996e00;
    const double d3 = 3.754408661907416e00;

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;

    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
            ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
    }
    if (phigh < p) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
            ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
    }

    const double q = p - 0.5;
    const double r = q * q;
    return ((((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q) /
        ((((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0));
}

}  // namespace

void copnorm_slice_kernel(
    const double* x,
    mwSize nTrials,
    mwSize nPages,
    mwSize threadCount,
    double* output) {
    if (nTrials == 0 || nPages == 0) {
        return;
    }

    const double denom = static_cast<double>(nTrials + 1);

    #pragma omp parallel num_threads(static_cast<int>(threadCount)) default(shared)
    {
        std::vector<std::size_t> order(nTrials);

        #pragma omp for schedule(static)
        for (mwSignedIndex page = 0; page < static_cast<mwSignedIndex>(nPages); ++page) {
            std::iota(order.begin(), order.end(), std::size_t{0});
            const double* pageData = x + static_cast<std::size_t>(page) * nTrials;
            std::sort(order.begin(), order.end(), [pageData](std::size_t a, std::size_t b) {
                const double av = pageData[a];
                const double bv = pageData[b];
                if (av < bv) {
                    return true;
                }
                if (bv < av) {
                    return false;
                }
                return a < b;
            });

            double* outPage = output + static_cast<std::size_t>(page) * nTrials;
            for (mwSize rank = 0; rank < nTrials; ++rank) {
                outPage[order[rank]] = ndtri((static_cast<double>(rank) + 1.0) / denom);
            }
        }
    }
}

}  // namespace gcmi
