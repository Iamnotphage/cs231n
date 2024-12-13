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

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      scores = X[i].dot(W)
      # 对于计算loss，因为要exp，同时减去最大值计算值不变，但是能避免溢出或者数值的不稳定
      scores = scores - max(scores)
      
      exp_sum = np.sum(np.exp(scores))

      # loss_i = -log(exp(s_yi)/sum(s_j)) = -s_yi + log(sum(s_j))
      # 可以避免一次log产生的误差
      loss += -scores[y[i]] + np.log(exp_sum)

      # 求导数，这里推导公式即可
      for j in range(num_classes):
        temp = np.exp(scores[j]) / exp_sum
        if j == y[i]:
          dW[:,j] += X[i].T * (temp - 1)
        else:
          dW[:,j] += X[i].T * temp
          
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W
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

    num_classes = W.shape[1]
    num_train = X.shape[0]
    # 点乘得到一个(N, C)的得分矩阵，一行代表第i个图像的得分情况
    scores = X.dot(W)
    scores = scores - np.max(scores, axis = 1).reshape(-1,1) # shift消减误差积累
    
    # 计算出所有 e^s_ij / sum(e^s_ij)
    portion = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(-1,1)
    # 挑出 y_i
    loss = -np.sum(np.log(portion[range(num_train), list(y)]))
    loss /= num_train
    loss += reg * np.sum(W * W)

    portion[range(num_train), list(y)] += -1
    dW = (X.T).dot(portion)
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
