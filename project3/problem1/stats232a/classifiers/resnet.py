from builtins import object
import numpy as np

from stats232a.layers import *
from stats232a.fast_layers import *
from stats232a.layer_utils import *

class OneBlockResnet(object):
    """
    A convolutional network with a residual block:
    conv - relu - 2x2 max pool - residual block - relu - fc - relu - fc - softmax
    The residual block has the following structure:
           ______________________      
          |                      |
    input - conv - relu - conv - + - output
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[64, 64, 64], filter_size=[7, 3, 3], 
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in each convolutional layer.
        - filter_size: Size of filters to use in each convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final fc layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the one block residual network.  #
        # Weights should be initialized from a Gaussian with standard deviation    #
        # equal to weight_scale; biases should be initialized to zero.             #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2', and keys 'W3' and 'b3' for the two     #
        # convolutional layers in residual block; use keys 'W4' and 'b4' for the   #
        # hidden fc layer; use keys 'W5' and 'b5' for the output fc layer.         #
        ############################################################################
        
        D, H, W = input_dim[0], input_dim[1], input_dim[2]
        self.params['W1'] = weight_scale * np.random.randn(num_filters[0], D, filter_size[0], filter_size[0])
        self.params['b1'] = np.zeros(num_filters[0])
        self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1])
        self.params['b2'] = np.zeros(num_filters[1])
        self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2])
        self.params['b3'] = np.zeros(num_filters[2])
        self.params['W4'] = weight_scale * np.random.randn(num_filters[2]*H*W/4, hidden_dim)
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b5'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param1 = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        filter_size = W2.shape[2]
        conv_param2 = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        filter_size = W3.shape[2]
        conv_param3 = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one block residual net,         #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        margin1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)

        margin2, cache2 = conv_relu_forward(margin1, W2, b2, conv_param2)
        margin3, cache3 = conv_forward_fast(margin2, W3, b3, conv_param3)
        margin4, cache4 = relu_forward(margin1 + margin3)

        margin5, cache5 = fc_relu_forward(margin4, W4, b4)
        scores, cache6 = fc_forward(margin5, W5, b5)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one block residual net,        #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1**2) + 0.5 * self.reg * np.sum(W2**2) + 0.5 * self.reg * np.sum(W3**2) + 0.5 * self.reg * np.sum(W4**2) + 0.5 * self.reg * np.sum(W5**2)
        
        dmargin5, dW5, db5 = fc_backward(dscores, cache6)
        dmargin4, dW4, db4 = fc_relu_backward(dmargin5, cache5)

        dmargin1and3 = relu_backward(dmargin4, cache4)
        dmargin2, dW3, db3 = conv_backward_fast(dmargin1and3, cache3)
        dmargin1, dW2, db2 = conv_relu_backward(dmargin2, cache2)
        dmargin1 += dmargin1and3

        dx, dW1, db1 = conv_relu_pool_backward(dmargin1, cache1)
        
        dW5 += self.reg * W5
        dW4 += self.reg * W4
        dW3 += self.reg * W3
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        
        grads['W5'] = dW5
        grads['b5'] = db5
        grads['W4'] = dW4
        grads['b4'] = db4
        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
