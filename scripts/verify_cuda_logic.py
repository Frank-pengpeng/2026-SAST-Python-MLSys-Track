"""Offline correctness check for the two CUDA kernels in src/simple_ml_cuda.cu.

The assignment's CUDA test is skipped unless the machine has an NVIDIA GPU, so
without this script the kernel bodies would be completely unverified.

This script re-implements the *exact* loop structure of

    softmax_gradient_kernel   (one thread == one example)
    update_theta_kernel       (one thread == one theta element)

using float32 scalar accumulation on the CPU, and compares the result against
the same pure-numpy reference used by `test_softmax_regression_epoch_cuda`.
It validates the index arithmetic -- the part that is easiest to get wrong --
not the GPU scheduling.

Run with:  python scripts/verify_cuda_logic.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def emulate_softmax_gradient_kernel(X, y, theta, logits_grad,
                                    batch_start, batch_size, n, k):
    """One thread per example; mirrors softmax_gradient_kernel."""
    for example in range(batch_size):
        row = batch_start + example
        max_logit = np.float32(-np.inf)

        for class_id in range(k):
            logit = np.float32(0.0)
            for j in range(n):
                logit = np.float32(logit + np.float32(
                    X[row * n + j] * theta[j * k + class_id]))
            logits_grad[example * k + class_id] = logit
            max_logit = np.float32(max(max_logit, logit))

        normalizer = np.float32(0.0)
        for class_id in range(k):
            value = np.float32(np.exp(np.float32(
                logits_grad[example * k + class_id] - max_logit)))
            logits_grad[example * k + class_id] = value
            normalizer = np.float32(normalizer + value)

        label = int(y[row])
        for class_id in range(k):
            softmax = np.float32(logits_grad[example * k + class_id] / normalizer)
            logits_grad[example * k + class_id] = np.float32(
                softmax - (1.0 if class_id == label else 0.0))


def emulate_update_theta_kernel(X, logits_grad, theta,
                               batch_start, batch_size, n, k, lr):
    """One thread per theta element; mirrors update_theta_kernel."""
    for parameter in range(n * k):
        feature = parameter // k
        class_id = parameter % k

        grad = np.float32(0.0)
        for i in range(batch_size):
            grad = np.float32(grad + np.float32(
                X[(batch_start + i) * n + feature] * logits_grad[i * k + class_id]))

        theta[parameter] = np.float32(
            theta[parameter] - np.float32(lr * grad / np.float32(batch_size)))


def emulate_cuda_epoch(X, y, theta, lr, batch, n=None, k=None):
    """Mirror of softmax_regression_epoch_cuda's host-side batch loop.

    X, y and theta are flat row-major buffers, exactly like the raw pointers
    the host function hands to the kernels.
    """
    if n is None:
        n = theta.shape[0] // theta.shape[1]
        k = theta.shape[1]
    m = X.shape[0] // n
    logits_grad = np.zeros(batch * k, dtype=np.float32)

    for start in range(0, m, batch):
        batch_size = min(batch, m - start)
        emulate_softmax_gradient_kernel(
            X, y, theta, logits_grad, start, batch_size, n, k)
        emulate_update_theta_kernel(
            X, logits_grad, theta, start, batch_size, n, k, lr)


def main():
    # Same fixture as tests/test_simple_ml.py::test_softmax_regression_epoch_cuda
    np.random.seed(0)
    X = np.random.randn(53, 7).astype(np.float32)
    y = np.random.randint(4, size=(53,)).astype(np.uint8)
    theta_reference = np.random.randn(7, 4).astype(np.float32) * 0.1
    theta_cuda = theta_reference.copy()

    lr, batch = 0.2, 16

    for start in range(0, X.shape[0], batch):
        X_batch = X[start:start + batch]
        y_batch = y[start:start + batch]
        logits = X_batch @ theta_reference
        logits -= logits.max(axis=1, keepdims=True)
        logits_grad = np.exp(logits)
        logits_grad /= logits_grad.sum(axis=1, keepdims=True)
        logits_grad[np.arange(y_batch.size), y_batch] -= 1
        theta_reference -= lr * (X_batch.T @ logits_grad) / y_batch.size

    # The kernels only ever see flat row-major buffers (raw pointers), so the
    # emulation has to work on 1D views too.  ravel() on C-contiguous arrays
    # returns a view, so writes to `theta_flat` show up in `theta_cuda`.
    emulate_cuda_epoch(X.ravel(), y.ravel(), theta_cuda.ravel(),
                       lr=lr, batch=batch, n=X.shape[1], k=theta_reference.shape[1])

    diff = np.abs(theta_cuda - theta_reference)
    print("theta (emulated CUDA kernels):")
    print(theta_cuda)
    print("\nmax |diff| vs numpy reference = {:.3e}".format(diff.max()))
    assert np.allclose(theta_cuda, theta_reference, rtol=1e-4, atol=1e-4), \
        "emulated kernels disagree with the reference implementation"
    print("PASS: kernel index arithmetic matches the reference implementation.")


if __name__ == "__main__":
    main()
