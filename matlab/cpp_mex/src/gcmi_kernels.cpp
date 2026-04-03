#include "gcmi_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

namespace gcmi {

namespace {

template <int Dim>
inline bool chol_logdet_upper(double* upper, double& logdet) {
    logdet = 0.0;
    for (int col = 0; col < Dim; ++col) {
        for (int row = 0; row < col; ++row) {
            double value = upper[row + col * Dim];
            for (int k = 0; k < row; ++k) {
                value -= upper[k + row * Dim] * upper[k + col * Dim];
            }
            value /= upper[row + row * Dim];
            upper[row + col * Dim] = value;
        }
        double diag = upper[col + col * Dim];
        for (int k = 0; k < col; ++k) {
            const double value = upper[k + col * Dim];
            diag -= value * value;
        }
        if (!(diag > 0.0)) {
            return false;
        }
        const double root = std::sqrt(diag);
        upper[col + col * Dim] = root;
        logdet += std::log(root);
    }
    return true;
}

template <int XDim, int YDim>
inline void info_cc_slice_small(
    const double* x,
    mwSize nTrials,
    mwSize nPages,
    const double* y,
    const double* cy,
    double hy,
    mwSize threadCount,
    double* output) {
    const double denom = 1.0 / static_cast<double>(nTrials - 1);
    const double ln2 = std::log(2.0);
    constexpr int JointDim = XDim + YDim;
    std::array<const double*, YDim> yCols{};
    for (int yi = 0; yi < YDim; ++yi) {
        yCols[yi] = y + static_cast<std::size_t>(yi) * nTrials;
    }

    #pragma omp parallel for num_threads(static_cast<int>(threadCount)) schedule(static) default(shared)
    for (mwSignedIndex page = 0; page < static_cast<mwSignedIndex>(nPages); ++page) {
        const double* xPage = x + static_cast<std::size_t>(page) * nTrials * XDim;
        std::array<const double*, XDim> xCols{};
        for (int xi = 0; xi < XDim; ++xi) {
            xCols[xi] = xPage + static_cast<std::size_t>(xi) * nTrials;
        }

        double sxx[XDim * XDim] = {};
        double cxy[XDim * YDim] = {};

        for (mwSize trial = 0; trial < nTrials; ++trial) {
            double xv[XDim];
            double yv[YDim];
            for (int xi = 0; xi < XDim; ++xi) {
                xv[xi] = xCols[xi][trial];
            }
            for (int yi = 0; yi < YDim; ++yi) {
                yv[yi] = yCols[yi][trial];
            }
            for (int col = 0; col < XDim; ++col) {
                for (int row = 0; row <= col; ++row) {
                    sxx[row + col * XDim] += xv[row] * xv[col];
                }
            }
            for (int col = 0; col < YDim; ++col) {
                for (int row = 0; row < XDim; ++row) {
                    cxy[row + col * XDim] += xv[row] * yv[col];
                }
            }
        }

        for (int col = 0; col < XDim; ++col) {
            for (int row = 0; row <= col; ++row) {
                sxx[row + col * XDim] *= denom;
            }
        }
        for (int col = 0; col < YDim; ++col) {
            for (int row = 0; row < XDim; ++row) {
                cxy[row + col * XDim] *= denom;
            }
        }

        double joint[JointDim * JointDim] = {};
        for (int col = 0; col < XDim; ++col) {
            for (int row = 0; row <= col; ++row) {
                joint[row + col * JointDim] = sxx[row + col * XDim];
            }
        }
        for (int col = 0; col < YDim; ++col) {
            for (int row = 0; row < XDim; ++row) {
                joint[row + (XDim + col) * JointDim] = cxy[row + col * XDim];
            }
        }
        for (int col = 0; col < YDim; ++col) {
            for (int row = 0; row <= col; ++row) {
                joint[(XDim + row) + (XDim + col) * JointDim] = cy[row + col * YDim];
            }
        }

        double hx = 0.0;
        if (!chol_logdet_upper<XDim>(sxx, hx)) {
            output[page] = nan_value();
            continue;
        }
        double hxy = 0.0;
        if (!chol_logdet_upper<JointDim>(joint, hxy)) {
            output[page] = nan_value();
            continue;
        }
        output[page] = (hx + hy - hxy) / ln2;
    }
}

inline bool dispatch_info_cc_slice_small(
    const double* x,
    mwSize nTrials,
    mwSize xDim,
    mwSize nPages,
    const double* y,
    mwSize yDim,
    const double* cy,
    double hy,
    mwSize threadCount,
    double* output) {
#define GCMI_INFO_CC_SMALL_CASE(XDIM, YDIM) \
    case YDIM: \
        info_cc_slice_small<XDIM, YDIM>(x, nTrials, nPages, y, cy, hy, threadCount, output); \
        return true

    switch (xDim) {
        case 1:
            switch (yDim) {
                GCMI_INFO_CC_SMALL_CASE(1, 1);
                GCMI_INFO_CC_SMALL_CASE(1, 2);
                GCMI_INFO_CC_SMALL_CASE(1, 3);
                GCMI_INFO_CC_SMALL_CASE(1, 4);
                default:
                    return false;
            }
        case 2:
            switch (yDim) {
                GCMI_INFO_CC_SMALL_CASE(2, 1);
                GCMI_INFO_CC_SMALL_CASE(2, 2);
                GCMI_INFO_CC_SMALL_CASE(2, 3);
                GCMI_INFO_CC_SMALL_CASE(2, 4);
                default:
                    return false;
            }
        case 3:
            switch (yDim) {
                GCMI_INFO_CC_SMALL_CASE(3, 1);
                GCMI_INFO_CC_SMALL_CASE(3, 2);
                GCMI_INFO_CC_SMALL_CASE(3, 3);
                GCMI_INFO_CC_SMALL_CASE(3, 4);
                default:
                    return false;
            }
        case 4:
            switch (yDim) {
                GCMI_INFO_CC_SMALL_CASE(4, 1);
                GCMI_INFO_CC_SMALL_CASE(4, 2);
                GCMI_INFO_CC_SMALL_CASE(4, 3);
                GCMI_INFO_CC_SMALL_CASE(4, 4);
                default:
                    return false;
            }
        default:
            return false;
    }

#undef GCMI_INFO_CC_SMALL_CASE
}

}  // namespace

void info_cc_slice(
    const double* x,
    mwSize nTrials,
    mwSize xDim,
    mwSize nPages,
    const double* y,
    mwSize yDim,
    mwSize threadCount,
    double* output) {
    require(nTrials > (xDim + yDim), "info_cc_slice_cpp requires Ntrl > Xdim + Ydim");

    const BlasInt nTrialsBlas = to_blas_int(nTrials, "Ntrl");
    const BlasInt xDimBlas = to_blas_int(xDim, "Xdim");
    const BlasInt yDimBlas = to_blas_int(yDim, "Ydim");
    const BlasInt xyDimBlas = to_blas_int(xDim + yDim, "Xdim + Ydim");
    const double denom = 1.0 / static_cast<double>(nTrials - 1);
    const double zero = 0.0;
    const double ln2 = std::log(2.0);
    const char uplo = 'U';
    const char trans = 'T';
    const char transb = 'N';

    std::vector<double> cy(yDim * yDim, 0.0);
    dsyrk(&uplo, &trans, &yDimBlas, &nTrialsBlas, &denom, y, &nTrialsBlas, &zero, cy.data(), &yDimBlas);

    std::vector<double> cyChol = cy;
    require(cholesky_upper_in_place(cyChol.data(), yDimBlas), "info_cc_slice_cpp failed Cholesky factorization for shared Y");
    const double hy = logdet_from_cholesky_upper(cyChol.data(), yDimBlas);

    if (dispatch_info_cc_slice_small(x, nTrials, xDim, nPages, y, yDim, cy.data(), hy, threadCount, output)) {
        return;
    }

    #pragma omp parallel num_threads(static_cast<int>(threadCount)) default(shared)
    {
        std::vector<double> cxCov(xDim * xDim, 0.0);
        std::vector<double> cxy(xDim * yDim, 0.0);
        std::vector<double> joint((xDim + yDim) * (xDim + yDim), 0.0);

        #pragma omp for schedule(static)
        for (mwSignedIndex page = 0; page < static_cast<mwSignedIndex>(nPages); ++page) {
            std::fill(cxCov.begin(), cxCov.end(), 0.0);
            std::fill(cxy.begin(), cxy.end(), 0.0);
            const double* xPage = x + static_cast<std::size_t>(page) * nTrials * xDim;

            dsyrk(&uplo, &trans, &xDimBlas, &nTrialsBlas, &denom, xPage, &nTrialsBlas, &zero, cxCov.data(), &xDimBlas);
            for (mwSize col = 0; col < xDim; ++col) {
                for (mwSize row = 0; row <= col; ++row) {
                    joint[row + col * (xDim + yDim)] = cxCov[row + col * xDim];
                }
            }

            dgemm(&trans, &transb, &xDimBlas, &yDimBlas, &nTrialsBlas, &denom, xPage, &nTrialsBlas, y, &nTrialsBlas, &zero, cxy.data(), &xDimBlas);
            for (mwSize col = 0; col < yDim; ++col) {
                for (mwSize row = 0; row < xDim; ++row) {
                    joint[row + (xDim + col) * (xDim + yDim)] = cxy[row + col * xDim];
                }
            }
            for (mwSize col = 0; col < yDim; ++col) {
                for (mwSize row = 0; row <= col; ++row) {
                    joint[(xDim + row) + (xDim + col) * (xDim + yDim)] = cy[row + col * yDim];
                }
            }

            if (!cholesky_upper_in_place(cxCov.data(), xDimBlas)) {
                output[page] = nan_value();
                continue;
            }
            const double hx = logdet_from_cholesky_upper(cxCov.data(), xDimBlas);

            if (!cholesky_upper_in_place(joint.data(), xyDimBlas)) {
                output[page] = nan_value();
                continue;
            }
            const double hxy = logdet_from_cholesky_upper(joint.data(), xyDimBlas);
            output[page] = (hx + hy - hxy) / ln2;
        }
    }
}

std::vector<double> info_cd_slice(
    const double* x,
    mwSize xDim,
    mwSize nTrials,
    mwSize nPages,
    const std::vector<mwSignedIndex>& labels,
    mwSize nClasses,
    mwSize threadCount) {
    require(labels.size() == nTrials, "label count does not match Ntrl");
    auto counts = count_labels(labels, nClasses);
    for (mwSize group = 0; group < nClasses; ++group) {
        require(counts[group] > 0, "info_cd_slice_cpp does not support empty classes");
        require(counts[group] > xDim, "info_cd_slice_cpp requires each class to have more than Xdim samples");
    }
    require(nTrials > xDim, "info_cd_slice_cpp requires Ntrl > Xdim");

    const BlasInt xDimBlas = to_blas_int(xDim, "Xdim");
    const double ln2 = std::log(2.0);
    const double alpha = -1.0 / static_cast<double>(nTrials);
    const double alpha1 = 1.0 / static_cast<double>(nTrials - 1);

    std::vector<double> output(nPages, nan_value());
    #pragma omp parallel num_threads(static_cast<int>(threadCount)) default(shared)
    {
        std::vector<double> sumX(xDim, 0.0);
        std::vector<double> sumXX(xDim * xDim, 0.0);
        std::vector<double> sumXg(xDim * nClasses, 0.0);
        std::vector<double> sumXXg(xDim * xDim * nClasses, 0.0);

        #pragma omp for schedule(static)
        for (mwSignedIndex page = 0; page < static_cast<mwSignedIndex>(nPages); ++page) {
            std::fill(sumX.begin(), sumX.end(), 0.0);
            std::fill(sumXX.begin(), sumXX.end(), 0.0);
            std::fill(sumXg.begin(), sumXg.end(), 0.0);
            std::fill(sumXXg.begin(), sumXXg.end(), 0.0);

        const double* xPage = x + static_cast<std::size_t>(page) * nTrials * xDim;

            for (mwSize trial = 0; trial < nTrials; ++trial) {
                const mwSize group = static_cast<mwSize>(labels[trial]);
                for (mwSize ii = 0; ii < xDim; ++ii) {
                    const double xi = xPage[ii + trial * xDim];
                    sumX[ii] += xi;
                    sumXg[ii + group * xDim] += xi;
                    for (mwSize jj = 0; jj <= ii; ++jj) {
                        const double xj = xPage[jj + trial * xDim];
                        sumXX[jj + ii * xDim] += xi * xj;
                        sumXXg[jj + ii * xDim + group * xDim * xDim] += xi * xj;
                    }
                }
            }

            for (mwSize ii = 0; ii < xDim; ++ii) {
                for (mwSize jj = 0; jj <= ii; ++jj) {
                    sumXX[jj + ii * xDim] = alpha1 * (sumXX[jj + ii * xDim] + alpha * sumX[ii] * sumX[jj]);
                }
            }

            if (!cholesky_upper_in_place(sumXX.data(), xDimBlas)) {
                output[static_cast<std::size_t>(page)] = nan_value();
                continue;
            }
            double hUnc = logdet_from_cholesky_upper(sumXX.data(), xDimBlas);

            double weightedHCond = 0.0;
            bool failed = false;
            for (mwSize group = 0; group < nClasses; ++group) {
                double* cov = sumXXg.data() + group * xDim * xDim;
                const double groupAlpha = -1.0 / static_cast<double>(counts[group]);
                const double groupAlpha1 = 1.0 / static_cast<double>(counts[group] - 1);
                const double* sumGroup = sumXg.data() + group * xDim;
                for (mwSize ii = 0; ii < xDim; ++ii) {
                    for (mwSize jj = 0; jj <= ii; ++jj) {
                        cov[jj + ii * xDim] = groupAlpha1 * (cov[jj + ii * xDim] + groupAlpha * sumGroup[ii] * sumGroup[jj]);
                    }
                }
                if (!cholesky_upper_in_place(cov, xDimBlas)) {
                    failed = true;
                    break;
                }
                const double hCond = logdet_from_cholesky_upper(cov, xDimBlas);
                weightedHCond += (static_cast<double>(counts[group]) / static_cast<double>(nTrials)) * hCond;
            }

            if (failed) {
                output[static_cast<std::size_t>(page)] = nan_value();
                continue;
            }
            output[static_cast<std::size_t>(page)] = (hUnc - weightedHCond) / ln2;
        }
    }

    return output;
}

}  // namespace gcmi
