#include "mex.h"

#include <math.h>

void gcmi_info_cc_slice_bridge(
    const double* x,
    mwSize ntrl,
    mwSize xdim,
    mwSize npage,
    const double* y,
    mwSize ydim,
    mwSize threads,
    double* output,
    const char** error_message);

static mwSize scalar_to_size(const mxArray* value, const char* name) {
    double scalar;
    if (!mxIsDouble(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != 1) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:type", "%s must be a real double scalar.", name);
    }
    scalar = mxGetScalar(value);
    if (scalar < 0.0 || scalar != floor(scalar)) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:value", "%s must be a non-negative integer scalar.", name);
    }
    return (mwSize)scalar;
}

static const double* require_double_array(const mxArray* value, const char* name) {
    const double* data;
    if (!mxIsDouble(value) || mxIsComplex(value)) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:type", "%s must be a real double array.", name);
    }
    data = mxGetPr(value);
    if (data == NULL) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:null", "%s data pointer was null.", name);
    }
    return data;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    const double* x;
    const double* y;
    double* out;
    mwSize xdim;
    mwSize ntrl;
    mwSize threads;
    mwSize npage;
    mwSize ydim;
    const mwSize* xDims;
    const mwSize* yDims;
    const char* error_message = NULL;

    if (nrhs != 5 || nlhs > 1) {
        mexErrMsgIdAndTxt(
            "info_cc_slice_cpp_capi:usage",
            "info_cc_slice_cpp_capi(X, Xdim, Y, Ntrl, Nthread) expects five inputs.");
    }

    x = require_double_array(prhs[0], "X");
    xdim = scalar_to_size(prhs[1], "Xdim");
    y = require_double_array(prhs[2], "Y");
    ntrl = scalar_to_size(prhs[3], "Ntrl");
    threads = scalar_to_size(prhs[4], "Nthread");
    if (threads < 1) {
        threads = 1;
    }

    if (mxGetNumberOfDimensions(prhs[0]) != 3) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:shape", "X must have shape [Ntrl Xdim Npage].");
    }
    if (mxGetNumberOfDimensions(prhs[2]) != 2) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:shape", "Y must have shape [Ntrl Ydim].");
    }

    xDims = mxGetDimensions(prhs[0]);
    yDims = mxGetDimensions(prhs[2]);
    if (xDims[0] != ntrl || xDims[1] != xdim) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:shape", "X must have shape [Ntrl Xdim Npage].");
    }
    if (yDims[0] != ntrl) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:shape", "Y must have shape [Ntrl Ydim].");
    }

    npage = xDims[2];
    ydim = yDims[1];
    plhs[0] = mxCreateDoubleMatrix(1, npage, mxREAL);
    out = mxGetPr(plhs[0]);
    if (out == NULL) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:null", "Output allocation failed.");
    }

    gcmi_info_cc_slice_bridge(x, ntrl, xdim, npage, y, ydim, threads, out, &error_message);
    if (error_message != NULL) {
        mexErrMsgIdAndTxt("info_cc_slice_cpp_capi:runtime", "%s", error_message);
    }
}
