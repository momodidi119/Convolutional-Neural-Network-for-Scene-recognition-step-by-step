import numpy as np

from main.layers import *
from main.fast_layers import *
from main.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    #########################################################
    # pass
    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params['W2'] = weight_scale * np.random.randn(F*H/2*W/2, hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros(F)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)
    #################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ###############################################################
    pass
    #conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
    #relu_out, relu_cache = relu_forward(conv_out)
    #pool_out, pool_cache = max_pool_forward_naive(relu_out, pool_param)
    #pool_out = pool_out.reshape(pool_out.shape[0], -1)
    pool_out, pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine_out, affine_cache = affine_relu_forward(pool_out, W2, b2)
    scores, cache = affine_forward(affine_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    #############################################################
    pass
    loss, dscore = softmax_loss(scores, y)
    daffine, grads['W3'], grads['b3'] = affine_backward(dscore, cache)
    dpool, grads['W2'], grads['b2'] = affine_relu_backward(daffine, affine_cache)
    #dpool = dpool.reshape(X.shape[0], W1.shape[0], X.shape[2]/2, X.shape[3]/2)
    #drelu = max_pool_backward_naive(dpool, pool_cache)
    #dconv = relu_backward(drelu, relu_cache)
    #dx, grads['W1'], grads['b1'] = conv_backward_naive(dconv, conv_cache)
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dpool, pool_cache)

    loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3
    ##################################################################

    return loss, grads


pass
