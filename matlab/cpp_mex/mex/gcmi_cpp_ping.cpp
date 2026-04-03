#include "gcmi_mex_adapter_utils.hpp"

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (!inputs.empty() || outputs.size() > 1) {
            fail("gcmi_cpp_ping:usage", "gcmi_cpp_ping takes no inputs and returns one struct.");
        }

        matlab::data::StructArray out = factory_.createStructArray({1, 1}, {"release", "arch", "mexext"});
        out[0]["release"] = factory_.createCharArray(GCMI_CPP_MATLAB_RELEASE);
        out[0]["arch"] = factory_.createCharArray(GCMI_CPP_ARCH);
        out[0]["mexext"] = factory_.createCharArray(GCMI_CPP_MEXEXT);
        outputs[0] = out;
    }
};
