import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # The MNIST files use the IDX binary format.  Integers are stored
    # big-endian ("  >  "), and both files are gzip compressed:
    #
    #   images: magic(4) | num_images(4) | rows(4) | cols(4) | uint8 pixels
    #   labels: magic(4) | num_labels(4)                          | uint8 labels
    #
    # We read the header with `struct` so that the pixel payload can be slurped
    # straight into a numpy array with zero per-pixel Python loops.
    with gzip.open(image_filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)

    with gzip.open(label_filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_labels)

    # Normalize with respect to the *whole dataset* (255 is the max possible
    # pixel value), not per image.  astype() also makes X writable, since
    # frombuffer returns a read-only view onto the gzip buffer.
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.uint8).copy()

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # loss = -1/b * sum_i log( exp(Z[i, y_i]) / sum_j exp(Z[i, j]) )
    #      =  1/b * sum_i ( logsumexp(Z[i]) - Z[i, y_i] )
    # Fully vectorized: one gather for the true-class logits, one row-wise
    # log-sum-exp.  The max-shift keeps exp() in range and does not change the
    # mathematical result.
    Z = Z - Z.max(axis=1, keepdims=True)
    log_partition = np.log(np.sum(np.exp(Z), axis=1))
    log_prob_true = log_partition - Z[np.arange(Z.shape[0]), y]
    return np.mean(log_prob_true)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify
    the theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # For a minibatch of size b the gradient of the *mean* loss is
    #     dL/dTheta = X_b^T (softmax(X_b Theta) - one_hot(y_b)) / b
    # so the in-place SGD update is Theta -= lr * that.  Every step below is a
    # single BLAS-backed numpy call; there is no Python loop over examples.
    m = X.shape[0]
    for start in range(0, m, batch):
        X_b = X[start:start + batch]
        y_b = y[start:start + batch]
        b = X_b.shape[0]

        Z = X_b @ theta                                   # (b, k) logits
        Z = Z - Z.max(axis=1, keepdims=True)              # numerical stability
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)                 # (b, k) softmax
        P[np.arange(b), y_b] -= 1.0                       # softmax - one_hot(y)

        theta -= lr * (X_b.T @ P) / b
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # Backprop through  logits = ReLU(X W1) W2  with softmax cross-entropy:
    #   Z2     = H W2                          (H = ReLU(Z1), Z1 = X W1)
    #   dZ2    = (softmax(Z2) - one_hot(y)) / b
    #   dW2    = H^T dZ2
    #   dZ1    = (dZ2 W2^T) * 1[Z1 > 0]        (ReLU subgradient)
    #   dW1    = X^T dZ1
    m = X.shape[0]
    for start in range(0, m, batch):
        X_b = X[start:start + batch]
        y_b = y[start:start + batch]
        b = X_b.shape[0]

        Z1 = X_b @ W1                                     # (b, d) pre-activation
        H = np.maximum(Z1, 0)                             # (b, d) ReLU
        Z2 = H @ W2                                       # (b, k) logits

        Z2 = Z2 - Z2.max(axis=1, keepdims=True)
        P = np.exp(Z2)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(b), y_b] -= 1.0                       # (b, k) dZ2 * b

        dW2 = H.T @ P
        dH = P @ W2.T
        dZ1 = dH * (Z1 > 0)                               # (b, d)
        dW1 = X_b.T @ dZ1

        W2 -= lr * dW2 / b
        W1 -= lr * dW1 / b
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
