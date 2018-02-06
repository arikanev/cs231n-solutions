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
    
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i].dot(W)
        
        # lets avoid exploding nums from exp(large) by shifting the scores.
      shift_scores = scores - max(scores)
        
        # cross entropy loss
      loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
      loss += loss_i
      for j in xrange(num_classes):
          softmax = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
          if j==y[i]:
                # take grad for correct current class prob
              dW[:,j] += (-1 + softmax) * X[i]
          else:
              dW[:,j] += softmax *X[i]
                
  #get mean loss across all samples
  loss /= num_train
  #add regularizer and L2 ridge loss
  loss += 0.5 * reg * np.sum(W * W)
  # mean gradient across all samples
  # add regularizer and L1 lasso loss
  dW /= num_train + reg * W
                            
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

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # (500,3073) dot (3073,10) = (500,10)
  scores = X.dot(W)

  # avoid large numbers feeding into softmax b/c e^large is laaarge. We want to shift inputs so when e is
  # is raised by them we dont get too large numbers
  shift_scores = scores - np.amax(scores,axis=1).reshape(-1,1)

  #calculate softmax
  softmax = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1).reshape(-1,1)
    
  #lets calculate the cross entropy loss
  # find the predicted probability for each y label
  # take logs for all 500 predicted probabilities corresponding to each 500 sample
  # take sum the 500 logs, then negate
  loss = -np.sum(np.log(softmax[range(num_train), list(y)]))
  #take mean loss across all training samples
  loss /= num_train
  #add regularizer to loss as well as L2 ridge loss (sum of weight matrix squared)
  loss += 0.5* reg * np.sum(W * W)
  
  #get softmax gradient
  dSoftmax = softmax.copy()
  
  # add (-1) to all 'correct' predicted class probabilities
  dSoftmax[range(num_train), list(y)] += -1

  #get weight matrix gradient
  dW = (X.T).dot(dSoftmax)
  dW = dW/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

