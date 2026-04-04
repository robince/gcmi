#pragma once

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gcmi_mex_utils.hpp"

namespace gcmi {

class MexAdapterBase : public matlab::mex::Function {
protected:
    matlab::data::ArrayFactory factory_;

    [[noreturn]] void fail(const char* id, const char* message) {
        auto engine = getEngine();
        engine->feval(u"error", 0, std::vector<matlab::data::Array>({
            factory_.createCharArray(id),
            factory_.createCharArray(message)}));
        throw std::runtime_error(message);
    }

    [[noreturn]] void fail(const char* id, const std::string& message) {
        fail(id, message.c_str());
    }

    matlab::data::TypedArray<double> require_double_array(const matlab::data::Array& array, const char* name) {
        if (array.getType() != matlab::data::ArrayType::DOUBLE) {
            fail("gcmi_cpp_mex:type", std::string(name) + " must have class double");
        }
        return array;
    }

    gcmi::mwSize scalar_to_size(const matlab::data::Array& array, const char* name) {
        if (array.getNumberOfElements() != 1 || array.getType() != matlab::data::ArrayType::DOUBLE) {
            fail("gcmi_cpp_mex:scalar", std::string(name) + " must be a scalar double");
        }
        const auto typed = matlab::data::TypedArray<double>(array);
        const double value = typed[0];
        if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
            fail("gcmi_cpp_mex:scalar", std::string(name) + " must be a non-negative integer scalar");
        }
        return static_cast<gcmi::mwSize>(value);
    }

    const double* raw_data(const matlab::data::TypedArray<double>& array) const {
        return array.getNumberOfElements() == 0 ? nullptr : &(*array.cbegin());
    }

    double* raw_data(matlab::data::TypedArray<double>& array) const {
        return array.getNumberOfElements() == 0 ? nullptr : &(*array.begin());
    }

    std::vector<gcmi::mwSignedIndex> parse_zero_based_labels(const matlab::data::Array& array, gcmi::mwSize expectedCount, gcmi::mwSize nClasses, const char* name) {
        if (array.getNumberOfElements() != expectedCount) {
            fail("gcmi_cpp_mex:labels", std::string(name) + " length does not match trial count");
        }
        std::vector<gcmi::mwSignedIndex> labels;
        switch (array.getType()) {
            case matlab::data::ArrayType::DOUBLE: {
                const auto typed = matlab::data::TypedArray<double>(array);
                labels.reserve(expectedCount);
                for (double value : typed) {
                    if (!std::isfinite(value) || std::floor(value) != value || value < 0.0 || value >= static_cast<double>(nClasses)) {
                        fail("gcmi_cpp_mex:labels", "discrete labels must be integers in the range 0..M-1");
                    }
                    labels.push_back(static_cast<gcmi::mwSignedIndex>(value));
                }
                return labels;
            }
            case matlab::data::ArrayType::INT32: {
                const auto typed = matlab::data::TypedArray<int32_t>(array);
                labels.reserve(expectedCount);
                for (int32_t value : typed) {
                    if (value < 0 || value >= static_cast<int32_t>(nClasses)) {
                        fail("gcmi_cpp_mex:labels", "discrete labels must be integers in the range 0..M-1");
                    }
                    labels.push_back(static_cast<gcmi::mwSignedIndex>(value));
                }
                return labels;
            }
            default:
                fail("gcmi_cpp_mex:labels", std::string(name) + " must have class double or int32");
        }
    }
};

}  // namespace gcmi
