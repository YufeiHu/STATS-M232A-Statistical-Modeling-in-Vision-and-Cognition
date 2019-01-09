from builtins import range
import numpy as np


def fc_forward(x, w, b):
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[cache < 0] = 0
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N, C, H, W = x.shape[:]
    F, _, HH, WW = w.shape[:]
    S = conv_param['stride']
    P = conv_param['pad']
  
    H_hat = int(1 + (H + 2*P - HH) / S)
    W_hat = int(1 + (W + 2*P - WW) / S)
  
    npad = ((0, 0), (0, 0), (P, P), (P, P))
    x_p = np.pad(x, npad, 'constant', constant_values=(0))
  
    out = np.zeros((N, F, H_hat, W_hat))
    for i in range(0, H_hat):
        x_i = S * i
        for j in range(0, W_hat):
            x_j = S * j
            for n in range(N):
                for f in range(F):
                    out[n, f, i, j] = np.sum(x_p[n, :, x_i:x_i+HH, x_j:x_j+WW] * w[f, :, :, :]) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    x, w, b, conv_param = cache
  
    N, C, H, W = x.shape[:]
    F, _, HH, WW = w.shape[:]
    S = conv_param['stride']
    P = conv_param['pad']
  
    H_hat = int(1 + (H + 2*P - HH) / S)
    W_hat = int(1 + (W + 2*P - WW) / S)
  
    npad = ((0, 0), (0, 0), (P, P), (P, P))
    x_p = np.pad(x, npad, 'constant', constant_values=(0))
  
    dx_p = np.zeros(x_p.shape)
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    for i in range(0, H_hat):
        x_i = S*i
        for j in range(0, W_hat):
            x_j = S*j
            for n in range(N):
                for f in range(F):
                    dx_p[n, :, x_i:x_i+HH, x_j:x_j+WW] += dout[n, f, i, j] * w[f, :, :, :]
                    dw[f, :, :, :] += dout[n, f, i, j] * x_p[n, :, x_i:x_i+HH, x_j:x_j+WW]
                    db[f] += dout[n, f, i, j]
    dx = dx_p[:, :, P:H+P, P:W+P]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    
    N, C, H, W = x.shape[:]
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    S = pool_param['stride']
  
    H_hat = int(1 + (H - HH) / S)
    W_hat = int(1 + (W - WW) / S)
  
    out = np.zeros((N, C, H_hat, W_hat))
    for i in range(0, H_hat):
        x_i = S*i
        for j in range(0, W_hat):
            x_j = S*j
            for c in range(C):
                for n in range(N):
                    out[n, c, i, j] = np.max(x[n, c, x_i:x_i+HH, x_j:x_j+WW])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    
    x, pool_param = cache
  
    N, C, H, W = x.shape[:]
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    S = pool_param['stride']
  
    H_hat = int(1 + (H - HH) / S)
    W_hat = int(1 + (W - WW) / S)
  
    dx = np.zeros(x.shape)
    for i in range(0, H_hat):
        x_i = S*i
        for j in range(0, W_hat):
            x_j = S*j
            for c in range(C):
                for n in range(N):
                    idx = np.unravel_index(x[n, c, x_i:x_i+HH, x_j:x_j+WW].argmax(), x[n, c, x_i:x_i+HH, x_j:x_j+WW].shape)
                    dx[n, c, x_i+idx[0], x_j+idx[1]] += dout[n, c, i, j]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
