#include "gcmi_mex_adapter_utils.hpp"

#include <omp.h>

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 2 || outputs.size() > 1) {
            fail("gcmi_cpp_omp_probe:usage", "gcmi_cpp_omp_probe(N, Nthread) expects two scalar inputs.");
        }
        const auto count = scalar_to_size(inputs[0], "N");
        const auto threads = std::max<gcmi::mwSize>(1, scalar_to_size(inputs[1], "Nthread"));
        auto out = factory_.createArray<double>({count, 1});
        double* data = raw_data(out);
        #pragma omp parallel for num_threads(static_cast<int>(threads)) default(shared)
        for (gcmi::mwSignedIndex i = 0; i < static_cast<gcmi::mwSignedIndex>(count); ++i) {
            data[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);
        }
        outputs[0] = out;
    }
};
