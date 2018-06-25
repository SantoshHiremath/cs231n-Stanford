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
    num_train = X.shape[0]
    num_classes = W.shape[1]    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in np.arange(num_train):
        # compute cross-entropy loss
        scores = X[i].dot(W)
        shifted_scores = scores - scores.max()
        normalizing_constant = np.sum(np.exp(shifted_scores))
        p_yi = np.exp(shifted_scores[y[i]])/normalizing_constant
        loss += -np.log(p_yi)

        # compute gradient
        for k in np.arange(num_classes):
            p_k = np.exp(shifted_scores[k])/normalizing_constant
            dW[:, k] += (p_k - (y[i] == k))*X[i]
    
    # Average loss and gradient
    loss /= num_train
    dW /= num_train
    
    # regularized loss and gradient
    loss += 0.5*reg*np.sum(W * W)
    dW += reg*W 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]    
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W) #NxC
    shifted_scores = scores - np.max(scores, axis = 1, keepdims = True)
    normalizing_constants = np.sum(np.exp(shifted_scores), axis = 1, keepdims = True)
    p = np.exp(shifted_scores)/normalizing_constants
    p_yi = p[np.arange(num_train), y]
    loss = -np.sum(np.log(p_yi))
    loss /= num_train
    loss += 0.5*reg*np.sum(W * W)
    
    # Compute gradient
    scale_factor = p.copy() # scale factor for each datapoint
    scale_factor[np.arange(num_train), y] -= 1
    dW = X.T.dot(scale_factor)
    '''
    indicator = np.zeros_like(p)
    indicator(np.arange(num_train), y) = 1
    dw = np.dot(X.T, (p-ind))
    '''
    
    #indicator = np.zeros_like(scores)
    #indicator[np.arange(num_train), y] = 1    
    #dW = X.T.dot(p - indicator)
    dW /= num_train
    dW += reg*W 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

