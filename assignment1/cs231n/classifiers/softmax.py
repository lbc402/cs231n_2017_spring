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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for k in xrange(num_train):
    origin_scores = X[k].dot(W)
    probabilities = np.zeros(origin_scores.shape)
    logc = -np.max(origin_scores)
    total_sum = np.sum(np.exp(origin_scores - logc))

    for i in xrange(num_classes):
      probabilities[i] = np.exp(origin_scores[i] - logc) / total_sum

    for i in xrange(num_classes):
      if i == y[k]:
        dW[:, i] += -X[k] * (1 - probabilities[i])
      else:
        dW[:, i] += X[k] * probabilities[i]

    loss += -np.log(probabilities[y[k]])

  loss /= num_train
  dW /= num_train
  dW += reg * W
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X, W)
  scores -= np.max(scores, axis=1, keepdims=True) # subtract the maximum value to prevent the number too large
  correct_class_scores = np.sum(scores[range(num_train), y])
  scores = np.exp(scores)
  exp_sum = np.sum(scores, axis=1, keepdims=True)

  loss = -correct_class_scores + np.sum(np.log(exp_sum))
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  prob = scores / exp_sum
  prob[range(num_train), y] -= 1    #把 -Xi 项“分配”进梯度的公式里
  dW = np.dot(X.T, prob)
  dW = dW / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

