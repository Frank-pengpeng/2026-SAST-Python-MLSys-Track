#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Everything is row-major, so
    //     X[i, j]     -> X[i * n + j]
    //     theta[j, c] -> theta[j * k + c]
    //     Z[i, c]     -> Z[i * k + c]
    // We only ever touch raw pointers here, so every matrix product has to be
    // written out by hand (no external BLAS allowed by the assignment).

    // logits / gradient buffer, reused across minibatches (batch*k floats)
    float *Z = new float[batch * k];

    for (size_t start = 0; start < m; start += batch) {
        const size_t b = std::min(batch, m - start);   // final batch may be short
        const float *Xb = X + start * n;
        const unsigned char *yb = y + start;

        // ---- forward pass: Z = Xb @ theta -------------------------------
        for (size_t i = 0; i < b; ++i) {
            for (size_t c = 0; c < k; ++c) {
                float logit = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    logit += Xb[i * n + j] * theta[j * k + c];
                }
                Z[i * k + c] = logit;
            }
        }

        // ---- softmax(Z) - one_hot(yb), computed in place ----------------
        // The max shift keeps expf() from overflowing; it cancels out in the
        // normalisation, so the result is mathematically identical.
        for (size_t i = 0; i < b; ++i) {
            float max_logit = Z[i * k];
            for (size_t c = 1; c < k; ++c) {
                if (Z[i * k + c] > max_logit) max_logit = Z[i * k + c];
            }

            float normalizer = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                float value = std::exp(Z[i * k + c] - max_logit);
                Z[i * k + c] = value;
                normalizer += value;
            }

            for (size_t c = 0; c < k; ++c) {
                Z[i * k + c] /= normalizer;
            }
            Z[i * k + yb[i]] -= 1.0f;
        }

        // ---- backward pass: theta -= lr / b * Xb^T @ G ------------------
        for (size_t j = 0; j < n; ++j) {
            for (size_t c = 0; c < k; ++c) {
                float grad = 0.0f;
                for (size_t i = 0; i < b; ++i) {
                    grad += Xb[i * n + j] * Z[i * k + c];
                }
                theta[j * k + c] -= lr * grad / static_cast<float>(b);
            }
        }
    }

    delete[] Z;

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
