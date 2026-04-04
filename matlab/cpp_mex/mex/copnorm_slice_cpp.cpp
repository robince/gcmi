#include "gcmi_mex_adapter_utils.hpp"

#include "gcmi_kernels.hpp"

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 2 || outputs.size() > 1) {
            fail("copnorm_slice_cpp:usage", "copnorm_slice_cpp(X, Nthread) expects a double matrix and thread count.");
        }

        const auto x = require_double_array(inputs[0], "X");
        const auto threads = std::max<gcmi::mwSize>(1, scalar_to_size(inputs[1], "Nthread"));
        const auto dims = x.getDimensions();
        if (dims.size() != 2) {
            fail("copnorm_slice_cpp:shape", "X must have shape [Ntrl Npage]");
        }

        const auto nTrials = dims[0];
        const auto nPages = dims[1];
        auto out = factory_.createArray<double>({nTrials, nPages});
        auto normalized = gcmi::copnorm_slice_kernel(raw_data(x), nTrials, nPages, threads);
        if (!normalized.empty()) {
            std::copy(normalized.begin(), normalized.end(), raw_data(out));
        }
        outputs[0] = out;
    }
};
