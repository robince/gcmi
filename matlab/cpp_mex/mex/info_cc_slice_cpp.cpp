#include "gcmi_mex_adapter_utils.hpp"

#include "gcmi_kernels.hpp"

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 5 || outputs.size() > 1) {
            fail("info_cc_slice_cpp:usage", "info_cc_slice_cpp(X, Xdim, Y, Ntrl, Nthread) expects five inputs.");
        }

        const auto x = require_double_array(inputs[0], "X");
        const auto xdim = scalar_to_size(inputs[1], "Xdim");
        const auto y = require_double_array(inputs[2], "Y");
        const auto ntrl = scalar_to_size(inputs[3], "Ntrl");
        const auto threads = std::max<gcmi::mwSize>(1, scalar_to_size(inputs[4], "Nthread"));

        const auto xDims = x.getDimensions();
        const auto yDims = y.getDimensions();
        if ((xDims.size() != 2 && xDims.size() != 3) || xDims[0] != ntrl || xDims[1] != xdim) {
            fail("info_cc_slice_cpp:shape", "X must have shape [Ntrl Xdim Npage] or [Ntrl Xdim] for a single page");
        }
        if (yDims.size() != 2 || yDims[0] != ntrl) {
            fail("info_cc_slice_cpp:shape", "Y must have shape [Ntrl Ydim]");
        }

        const auto npage = xDims.size() == 2 ? 1 : xDims[2];
        const auto ydim = yDims[1];
        auto out = factory_.createArray<double>({1, npage});
        gcmi::info_cc_slice(raw_data(x), ntrl, xdim, npage, raw_data(y), ydim, threads, raw_data(out));
        outputs[0] = out;
    }
};
