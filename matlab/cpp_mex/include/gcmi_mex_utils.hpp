#pragma once

#include "blas.h"
#include "lapack.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gcmi {

using mwSize = std::size_t;
using mwSignedIndex = std::ptrdiff_t;
using BlasInt = ptrdiff_t;

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline BlasInt to_blas_int(mwSize value, const char* name) {
    constexpr auto maxValue = static_cast<mwSize>(std::numeric_limits<BlasInt>::max());
    if (value > maxValue) {
        throw std::runtime_error(std::string(name) + " is too large for BLAS/LAPACK");
    }
    return static_cast<BlasInt>(value);
}

inline std::vector<mwSize> count_labels(const std::vector<mwSignedIndex>& labels, mwSize nClasses) {
    std::vector<mwSize> counts(nClasses, 0);
    for (mwSignedIndex label : labels) {
        counts[static_cast<std::size_t>(label)] += 1;
    }
    return counts;
}

inline double nan_value() {
    return std::numeric_limits<double>::quiet_NaN();
}

inline double logdet_from_cholesky_upper(const double* matrix, BlasInt dim) {
    double sum = 0.0;
    for (BlasInt i = 0; i < dim; ++i) {
        sum += std::log(matrix[i + i * dim]);
    }
    return sum;
}

inline bool cholesky_upper_in_place(double* matrix, BlasInt dim) {
    const char uplo = 'U';
    BlasInt info = 0;
    dpotrf(&uplo, &dim, matrix, &dim, &info);
    return info == 0;
}

}  // namespace gcmi
