from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 因为隐藏层的层数是任意的，这里需要组合
        dims = [input_dim] + hidden_dims + [num_classes]
        
        for l in range(len(dims) - 1):
            # 每一层的输入size和输出size
            input_size = dims[l]
            output_size = dims[l + 1]

            # 初始化权重和biase
            self.params[f"W{l+1}"] = np.random.randn(input_size, output_size) * weight_scale
            self.params[f"b{l+1}"] = np.zeros(output_size)

            # 初始化batchnorm的参数
            if normalization == "batchnorm" and l < len(dims) - 2:
                self.params[f'gamma{l+1}'] = np.ones(output_size)  # 初始化gamma为1
                self.params[f'beta{l+1}'] = np.zeros(output_size)  # 初始化beta为0

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache = {}
        layer_in = X
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        for k in range(1, self.num_layers):
            W = self.params[f"W{k}"]
            b = self.params[f"b{k}"]
            
            affine_out, affine_cache = affine_forward(layer_in, W, b)

            if self.normalization == "batchnorm":
                bn_out, bn_cache = batchnorm_forward(affine_out, 
                    self.params[f"gamma{k}"], self.params[f"beta{k}"], self.bn_params[k - 1])
                cache[f"bn_cache{k}"] = bn_cache

                affine_out = bn_out # 兼容 normalization == None 的情况

            relu_out, relu_cache = relu_forward(affine_out)
            layer_in = relu_out

            cache[f"affine_cache{k}"] = affine_cache
            
            cache[f"relu_cache{k}"] = relu_cache
        
        # 最后的affine层
        W = self.params["W" + str(self.num_layers)]
        b = self.params["b" + str(self.num_layers)]
        scores, affine_cache = affine_forward(layer_in, W, b)

        cache["affine_cache" + str(self.num_layers)] = affine_cache


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, d_affine_out = softmax_loss(scores, y)
        
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        # 这里affine_cache恰好是最后一层affine的参数
        W = self.params["W" + str(self.num_layers)]
        d_relu_out, dw, db = affine_backward(d_affine_out, affine_cache)
        # 因为loss包含了1/2 * self.reg * W^2 所以dw = self.reg * W
        dw += self.reg * W
        # make sure that grads[k] holds the gradients for self.params[k]
        grads["W" + str(self.num_layers)] = dw
        grads["b" + str(self.num_layers)] = db

        # 反向传播(除了最后一层)
        for k in range(self.num_layers - 1, 0, -1):
            relu_cache = cache[f"relu_cache{k}"]
            affine_cache = cache[f"affine_cache{k}"]

            W = self.params[f"W{k}"]

            d_affine_out = relu_backward(d_relu_out, relu_cache)

            if self.normalization == "batchnorm":
                bn_cache = cache[f"bn_cache{k}"]
                d_bn_out = d_affine_out # 如果是batchnrom，实际上upstream是d_bn_out

                d_affine_out, dgamma, dbeta = batchnorm_backward_alt(d_bn_out, bn_cache)
                grads[f"gamma{k}"] = dgamma
                grads[f"beta{k}"] = dbeta
                
            
            d_relu_out, dw, db = affine_backward(d_affine_out, affine_cache)

            dw += self.reg * W
            grads[f"W{k}"] = dw
            grads[f"b{k}"] = db

        # 最后别忘记L2正则化
        for k in range(1, self.num_layers + 1):
            W = self.params[f"W{k}"]
            loss += 0.5 * self.reg * np.sum(W * W)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
