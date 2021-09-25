from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    C = W.shape[1]
    N = X.shape[0]
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores = np.exp(scores)  # Exp
        scores /= np.sum(scores)  # Normalize
        loss += - np.log(scores[y[i]])

        for j in range(C):
            indicator = (y[i] == j)
            dW[:, j] += indicator * (X[i] * (scores[j] - 1)) + (1 - indicator) * scores[j] * X[i]

    loss /= N
    dW /= N

    loss += reg * np.square(W).sum()
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    n_range = np.arange(N)
    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape(N, 1)    # Trick to avoid numerical instabilty
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1).reshape(scores.shape[0], 1)
    loss = np.mean(-np.log(scores[n_range, y])) + reg * np.square(W).sum()

    # dW:

    z = np.zeros_like(scores)
    z[n_range, y] = scores[n_range, y] - 1
    c_range = np.arange(W.shape[1]).reshape(1, -1).repeat(N, axis=0)
    c_range = np.array(np.nonzero(c_range != y.reshape(-1, 1)))
    z[c_range[0], c_range[1]] = scores[c_range[0], c_range[1]]
    dW = X.T.dot(z)


    # dW = z.T.dot(scores)
    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

# W = np.random.randn(3, 2) * 0.01
x = np.array([[1, 1], [2, 2], [3, 3], [6, 6], [7, 7], [8, 8]])
# x = np.hstack((x, np.ones(x.shape[0]).reshape(-1, 1)))
# y = np.array([0,0,0,1,1,1])
# reg = 1e-5
#
# l1, g1 = softmax_loss_naive(W, x, y, reg)
# l2, g2 = softmax_loss_vectorized(W, x, y, reg)
#
# y = np.sum(x, axis=0)

def f(*argv):
    res = np.inf

    return np.min(argv)

print(f(1,2,3))