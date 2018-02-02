import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
    # for i in 500 X
  for i in xrange(num_train):
    # return array will be 1 x 10 (1 X times times Number of classes) ( [1,3073] times [3073,10])
    # we look at this array as probabilities for each class y(0-9)
    scores = X[i].dot(W)
    # y[i] denotes the correct class label for this current X (X[i]).
    # We use the current correct class label y[i] as an index for the score array of 10 probabilities (1 for each class)
    # This gives us the specific predicted class probability that matters for the current correct class label y
    correct_class_score = scores[y[i]]
    
    num_diff = 0
    
    # for j 0,...,9
    for j in xrange(num_classes):
      if j == y[i]: # ignore rest of for loop code when
        continue    # j is the value of the current correct class label y[i]
      
      # for all j except j==y[i]
      # we subtract the correct predicted class probability from all other predicted class probabilities, and add 1
      # this is the margin
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # if the margin is greater than zero, we add to the loss
      # if the correct predicted class probability is 1, and all other class probabilities are zero, we will add nothing to our         # loss, as this is a perfect prediction
      if margin > 0:
        num_diff += 1
        # If we have a non-perfect prediction, we add the features from current X to the gradient weight matrix at all the  
        # incorrect class probabilities
        # all rows in gradient weight matrix will be filled except for one (the correct class probability row)
        dW[:,j] += X[i] # 3073 dW (initially zero) + 3073 features (representation of current X)
        loss += margin
    
    # add features of current X to the correct class probability row in the grad matrix *-(number of classes between 0 and  
    # this one (correct class probability))
    dW[:,y[i]] += -num_diff * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  # we make the loss an avg specific to each X, rather than compute one loss for all the X (we want to know how accurate we are
  # per example, not per chunk of examples)
  loss /= num_train
  #average the gradient weight matrix per example, same as the loss
  dW /= num_train
  # multiply weights by regularization term
  dW += reg * W

  # Add regularization to the loss.
  # here we use the L2 regularization. 
  # L1 would use only one weight matrix (L1 vs L2) == (magnitude of coefficients vs squared magnitude) or (lasso vs ridge)
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]
  delta = 1.0
    
  # calculate scores vectorized (dot product of all X and weights)
  # scores shape = (500,10)
  scores = X.dot(W)
  # returns an array of shape (500,) correct class scores chosen by index consisting   # of: (X sample number, y correct class label for X)
  correct_class_score = scores[np.arange(num_train), y]
  # we subtract all class probabilities shape (500,10) (10 in each row), by the         # scalar correct class probability for the row. We also shift by 1
  # we take the max of score probabilities and 0
  margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + delta)
  #We take index of margins that are (X row probabilities, y label) and set this  
  # margin to 0. This is where the correct probability is and should not change the     # loss function
  margins[np.arange(num_train), y] = 0
  # sum the loss
  loss = np.sum(margins)
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  # we make the loss an avg specific to each X, rather than compute one loss for all
  # the X (we want to know how accurate we are
  # per example, not per chunk of examples)
  loss /= num_train
  #average the gradient weight matrix per example, same as the loss
  # multiply weights by regularization term

  # Add regularization to the loss.
  # here we use the L2 regularization. 
  # L1 would use only one weight matrix (L1 vs L2) == (magnitude of coefficients vs squared magnitude) or (lasso vs ridge)
  loss += 0.5 * reg * np.sum(W * W)
    
  # (500,10) shape
  X_index_mask = np.zeros(margins.shape)
  
  # find classes where margins are greater than zero
  X_mask[margins > 0] = 1
  incorrect_counts = np.sum(X_mask, axis=1)
  #vectorized version of calculating gradient matrix we take the places where there
  # is a correct probability, and we set equal to the negation of values in 
  # incorrect counts. There is one incorrect_count number per row, (X sample).
  X_mask[np.arange(num_train), y] = -incorrect_counts
  # take the transpose of the dot product to calculate dW
  dW = X.T.dot(X_mask)
    
  #average the grad weight matrix and regularize it
  dW /= num_train
  dW += reg*W


  return loss, dW
