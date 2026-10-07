"""End-to-end MNIST demo: the three training runs suggested by the notebook.

    1. numpy softmax regression        (10 epochs, lr=0.2, batch=100)
    2. the same epoch loop in C++      (same hyper-parameters, for the timing
                                        comparison discussed in problem 6)
    3. two-layer NN, 400 hidden units  (20 epochs, lr=0.2, batch=100)

Run with:  python scripts/train_mnist.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)  # parse_mnist uses repo-relative data paths

from simple_ml import (loss_err, nn_epoch, parse_mnist,
                       softmax_regression_epoch, train_nn, train_softmax)

try:
    from simple_ml_ext import softmax_regression_epoch_cpp
    HAVE_CPP = True
except ImportError:
    HAVE_CPP = False


def time_softmax(X_tr, y_tr, X_te, y_te, epochs, lr, batch, use_cpp):
    """train_softmax() without the printing, plus wall-clock timing."""
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    start = time.perf_counter()
    for _ in range(epochs):
        if use_cpp:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
    elapsed = time.perf_counter() - start
    test_loss, test_err = loss_err(X_te @ theta, y_te)
    return elapsed, test_loss, test_err


def main():
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")
    print(f"train X {X_tr.shape} {X_tr.dtype}   test X {X_te.shape}\n")

    print("=" * 62)
    print("1) numpy softmax regression, 10 epochs, lr=0.2, batch=100")
    print("=" * 62)
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.2, batch=100)
    np_time, np_loss, np_err = time_softmax(
        X_tr, y_tr, X_te, y_te, epochs=10, lr=0.2, batch=100, use_cpp=False)
    print(f"\n>>> numpy : {np_time:6.2f}s   test loss {np_loss:.5f}   "
          f"test error {np_err * 100:.2f}%")

    if HAVE_CPP:
        print("\n" + "=" * 62)
        print("2) C++ softmax regression, same hyper-parameters")
        print("=" * 62)
        cpp_time, cpp_loss, cpp_err = time_softmax(
            X_tr, y_tr, X_te, y_te, epochs=10, lr=0.2, batch=100, use_cpp=True)
        print(f">>> c++   : {cpp_time:6.2f}s   test loss {cpp_loss:.5f}   "
              f"test error {cpp_err * 100:.2f}%")
        print(f">>> ratio : C++ is {cpp_time / np_time:.2f}x the numpy time "
              f"(hand-written matmul vs. BLAS)")
        print(f">>> max |numpy - c++| theta-independent check: "
              f"loss diff {abs(cpp_loss - np_loss):.2e}, "
              f"error diff {abs(cpp_err - np_err):.2e}")
    else:
        print("\n[skip] simple_ml_ext not compiled; run the C++ build first.")

    print("\n" + "=" * 62)
    print("3) two-layer NN, 400 hidden units, 20 epochs, lr=0.2, batch=100")
    print("=" * 62)
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, 400).astype(np.float32) / np.sqrt(400)
    W2 = np.random.randn(400, k).astype(np.float32) / np.sqrt(k)
    start = time.perf_counter()
    for epoch in range(20):
        nn_epoch(X_tr, y_tr, W1, W2, lr=0.2, batch=100)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print(f"| {epoch:>4} | {train_loss:.5f} | {train_err:.5f} | "
              f"{test_loss:.5f} | {test_err:.5f} |")
    nn_time = time.perf_counter() - start
    print(f"\n>>> nn    : {nn_time:6.2f}s   final test error {test_err * 100:.2f}%")
    print(f">>> norm(W1) = {np.linalg.norm(W1):.6f}, "
          f"norm(W2) = {np.linalg.norm(W2):.6f}")


if __name__ == "__main__":
    main()
