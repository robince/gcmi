#include "gcmi_mex_adapter_utils.hpp"

#include <omp.h>

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 2 || outputs.size() > 1) {
            fail("gcmi_cpp_runtime_probe:usage", "gcmi_cpp_runtime_probe(X, Nthread) expects a double matrix and thread count.");
        }
        const auto x = require_double_array(inputs[0], "X");
        const auto dims = x.getDimensions();
        if (dims.size() != 2) {
            fail("gcmi_cpp_runtime_probe:shape", "X must be a 2-D double matrix");
        }
        const auto nTrials = dims[0];
        const auto nPages = dims[1];
        const auto threads = std::max<gcmi::mwSize>(1, scalar_to_size(inputs[1], "Nthread"));

        auto out = factory_.createArray<double>({nPages, 1});
        const double* xData = raw_data(x);
        double* outData = raw_data(out);
        #pragma omp parallel for num_threads(static_cast<int>(threads)) default(shared)
        for (gcmi::mwSignedIndex page = 0; page < static_cast<gcmi::mwSignedIndex>(nPages); ++page) {
            const double* pageData = xData + static_cast<std::size_t>(page) * nTrials;
            double sumSquares = 0.0;
            for (gcmi::mwSize trial = 0; trial < nTrials; ++trial) {
                sumSquares += pageData[trial] * pageData[trial];
            }
            outData[static_cast<std::size_t>(page)] = sumSquares;
        }
        outputs[0] = out;
    }
};
