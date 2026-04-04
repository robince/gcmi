#include "gcmi_mex_adapter_utils.hpp"

#include <vector>

class MexFunction : public gcmi::MexAdapterBase {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override {
        if (inputs.size() != 1 || outputs.size() > 1) {
            fail("gcmi_cpp_blas_probe:usage", "gcmi_cpp_blas_probe(A) expects one square double matrix.");
        }

        const auto a = require_double_array(inputs[0], "A");
        const auto dims = a.getDimensions();
        if (dims.size() != 2 || dims[0] != dims[1]) {
            fail("gcmi_cpp_blas_probe:shape", "A must be a square double matrix");
        }

        const auto n = dims[0];
        const gcmi::BlasInt nBlas = gcmi::to_blas_int(n, "A");
        const double one = 1.0;
        const double zero = 0.0;
        const char uplo = 'U';
        const char trans = 'T';
        std::vector<double> gram(n * n, 0.0);
        const double* data = raw_data(a);
        dsyrk(&uplo, &trans, &nBlas, &nBlas, &one, data, &nBlas, &zero, gram.data(), &nBlas);
        if (!gcmi::cholesky_upper_in_place(gram.data(), nBlas)) {
            fail("gcmi_cpp_blas_probe:chol", "Cholesky factorization failed");
        }
        outputs[0] = factory_.createScalar(gcmi::logdet_from_cholesky_upper(gram.data(), nBlas));
    }
};
