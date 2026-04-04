#include "gcmi_mex_adapter_utils.hpp"

#include "gcmi_kernels.hpp"

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 6 || outputs.size() > 1) {
            fail("info_cd_slice_cpp:usage", "info_cd_slice_cpp(X, Xdim, Y, Ym, Ntrl, Nthread) expects six inputs.");
        }

        const auto x = require_double_array(inputs[0], "X");
        const auto xdim = scalar_to_size(inputs[1], "Xdim");
        const auto ym = scalar_to_size(inputs[3], "Ym");
        const auto ntrl = scalar_to_size(inputs[4], "Ntrl");
        const auto threads = std::max<gcmi::mwSize>(1, scalar_to_size(inputs[5], "Nthread"));

        const auto xDims = x.getDimensions();
        if ((xDims.size() != 2 && xDims.size() != 3) || xDims[0] != xdim || xDims[1] != ntrl) {
            fail("info_cd_slice_cpp:shape", "X must have shape [Xdim Ntrl Npage] or [Xdim Ntrl] for a single page");
        }

        const auto labels = parse_zero_based_labels(inputs[2], ntrl, ym, "Y");
        const auto npage = xDims.size() == 2 ? 1 : xDims[2];
        auto info = gcmi::info_cd_slice(raw_data(x), xdim, ntrl, npage, labels, ym, threads);
        auto out = factory_.createArray<double>({1, npage});
        std::copy(info.begin(), info.end(), raw_data(out));
        outputs[0] = out;
    }
};
