#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(error));                 \
        }                                                                       \
    } while (0)


__global__ void softmax_gradient_kernel(const float *X,
                                        const unsigned char *y,
                                        const float *theta,
                                        float *logits_grad,
                                        size_t batch_size,
                                        size_t n,
                                        size_t k) {
    size_t example = blockIdx.x * blockDim.x + threadIdx.x;
    if (example >= batch_size) {
        return;
    }

    float max_logit = -INFINITY;

    for (size_t class_id = 0; class_id < k; ++class_id) {
        float logit = 0.0f;

        /// BEGIN YOUR CODE
        // Compute one element of X @ theta. Both arrays use row-major layout.
        //   X[example, j]     -> X[example * n + j]
        //   theta[j, class_id] -> theta[j * k + class_id]
        // This thread owns the whole row `example`, so it walks the k columns
        // and accumulates the dot product for each class in turn.
        for (size_t j = 0; j < n; ++j) {
            logit += X[example * n + j] * theta[j * k + class_id];
        }
        /// END YOUR CODE

        logits_grad[example * k + class_id] = logit;
        max_logit = fmaxf(max_logit, logit);
    }

    float normalizer = 0.0f;
    for (size_t class_id = 0; class_id < k; ++class_id) {
        float value = expf(logits_grad[example * k + class_id] - max_logit);
        logits_grad[example * k + class_id] = value;
        normalizer += value;
    }

    /// BEGIN YOUR CODE
    // Convert the stored exponentials into dL/dZ = softmax(Z) - one_hot(y).
    // Remember that logits_grad contains batch_size rows and k columns.
    unsigned char label = y[example];
    for (size_t class_id = 0; class_id < k; ++class_id) {
        float softmax = logits_grad[example * k + class_id] / normalizer;
        logits_grad[example * k + class_id] =
            softmax - (class_id == static_cast<size_t>(label) ? 1.0f : 0.0f);
    }
    /// END YOUR CODE
}


__global__ void update_theta_kernel(const float *X,
                                    const float *logits_grad,
                                    float *theta,
                                    size_t batch_size,
                                    size_t n,
                                    size_t k,
                                    float lr) {
    size_t parameter = blockIdx.x * blockDim.x + threadIdx.x;
    if (parameter >= n * k) {
        return;
    }

    size_t feature = parameter / k;
    size_t class_id = parameter % k;

    /// BEGIN YOUR CODE
    // Accumulate the minibatch gradient for theta[feature, class_id], then
    // perform the SGD update in place. Divide the gradient by batch_size.
    //   grad = sum_i X[i, feature] * G[i, class_id]
    //        = sum_i X[i * n + feature] * logits_grad[i * k + class_id]
    // Each thread handles exactly one element of the (n x k) theta matrix,
    // which is why `parameter` decomposes cleanly into (feature, class_id).
    float grad = 0.0f;
    for (size_t i = 0; i < batch_size; ++i) {
        grad += X[i * n + feature] * logits_grad[i * k + class_id];
    }
    theta[parameter] -= lr * grad / static_cast<float>(batch_size);
    /// END YOUR CODE
}


void softmax_regression_epoch_cuda(const float *X,
                                   const unsigned char *y,
                                   float *theta,
                                   size_t m,
                                   size_t n,
                                   size_t k,
                                   float lr,
                                   size_t batch) {
    if (batch == 0) {
        throw std::invalid_argument("batch must be positive");
    }
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    float *device_X = nullptr;
    unsigned char *device_y = nullptr;
    float *device_theta = nullptr;
    float *device_logits_grad = nullptr;

    CUDA_CHECK(cudaMalloc(&device_X, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_y, m * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_theta, n * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_logits_grad, batch * k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(device_X, X, m * n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, y, m * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_theta, theta, n * k * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr size_t examples_per_block = 128;
    constexpr size_t parameters_per_block = 256;

    for (size_t start = 0; start < m; start += batch) {
        size_t batch_size = (start + batch <= m) ? batch : (m - start);
        size_t example_blocks =
            (batch_size + examples_per_block - 1) / examples_per_block;
        size_t parameter_blocks =
            (n * k + parameters_per_block - 1) / parameters_per_block;

        softmax_gradient_kernel<<<example_blocks, examples_per_block>>>(
            device_X + start * n,
            device_y + start,
            device_theta,
            device_logits_grad,
            batch_size,
            n,
            k);
        CUDA_CHECK(cudaGetLastError());

        update_theta_kernel<<<parameter_blocks, parameters_per_block>>>(
            device_X + start * n,
            device_logits_grad,
            device_theta,
            batch_size,
            n,
            k,
            lr);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(theta, device_theta, n * k * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_logits_grad));
    CUDA_CHECK(cudaFree(device_theta));
    CUDA_CHECK(cudaFree(device_y));
    CUDA_CHECK(cudaFree(device_X));
}


bool cuda_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}


PYBIND11_MODULE(simple_ml_cuda, module) {
    module.def("cuda_available", &cuda_available);
    module.def(
        "softmax_regression_epoch_cuda",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
            if (batch <= 0) {
                throw std::invalid_argument("batch must be positive");
            }
            if (X.ndim() != 2 || y.ndim() != 1 || theta.ndim() != 2) {
                throw std::invalid_argument("X and theta must be 2D; y must be 1D");
            }
            if (X.shape(0) != y.shape(0) || X.shape(1) != theta.shape(0)) {
                throw std::invalid_argument("incompatible X, y, and theta shapes");
            }

            softmax_regression_epoch_cuda(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.shape(0),
                X.shape(1),
                theta.shape(1),
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
        py::arg("batch"));
}
