#include "gcmi_kernels.hpp"

#include <cstddef>
#include <exception>
#include <string>

extern "C" void gcmi_info_cc_slice_bridge(
    const double* x,
    std::size_t ntrl,
    std::size_t xdim,
    std::size_t npage,
    const double* y,
    std::size_t ydim,
    std::size_t threads,
    double* output,
    const char** error_message) {
    static thread_local std::string error_storage;
    try {
        gcmi::info_cc_slice(
            x,
            static_cast<gcmi::mwSize>(ntrl),
            static_cast<gcmi::mwSize>(xdim),
            static_cast<gcmi::mwSize>(npage),
            y,
            static_cast<gcmi::mwSize>(ydim),
            static_cast<gcmi::mwSize>(threads),
            output);
        error_storage.clear();
        *error_message = nullptr;
    } catch (const std::exception& ex) {
        error_storage = ex.what();
        *error_message = error_storage.c_str();
    } catch (...) {
        *error_message = "unknown error";
    }
}
