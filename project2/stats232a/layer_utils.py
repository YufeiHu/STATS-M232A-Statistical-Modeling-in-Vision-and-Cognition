pass
from stats232a.layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement fc-relu forward pass.                                   #
    ###########################################################################
    a, fc_cache = fc_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return out, cache


def fc_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the fc-relu backward pass.                              #
    ###########################################################################
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = fc_backward(da, fc_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return dx, dw, db


def fc_BN_relu_forward(x, w, b, gamma, beta, bn_params):
  r_1, fc_cache = fc_forward(x, w, b)
  r_2, BN_cache = batchnorm_forward(r_1, gamma, beta, bn_param=bn_params)
  out, relu_cache = relu_forward(r_2)
  cache = (fc_cache, BN_cache, relu_cache)
  return out, cache


def fc_BN_relu_backward(dout, cache):
  fc_cache, BN_cache, relu_cache = cache
  r_1 = relu_backward(dout, relu_cache)
  r_2, dgamma, dbeta = batchnorm_backward(r_1, BN_cache)
  dx, dw, db = fc_backward(r_2, fc_cache)
  return dx, dw, db, dgamma, dbeta