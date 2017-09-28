import numpy as np
import Utils.maths as um


def diff_actFun(z, type):
    if type == 'tanh':
        return um.diff_tanh(z)
    if type == 'sigmoid':
        return um.diff_sigmoid(z)
    if type == 'relu':
        return um.diff_relu(z)

    return None


class Layer(object):
    # W is weights that are incident on this layer
    def __init__(self, dims, prev_layer_dims, isOutput, prevLayer, nextLayer, isFirstLayerAfterInput,
                 actFun_type='tanh', reg_lambda=0.01, epsilon=0.01):
        self.prev_layer_dims = prev_layer_dims
        self.dims = dims
        self.isOutput = isOutput
        self.actFun_type = actFun_type
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        self.isFirstLayerAfterInput = isFirstLayerAfterInput
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon

        self.W = np.random.randn(self.prev_layer_dims, self.dims) / np.sqrt(self.prev_layer_dims)
        self.b = np.zeros((1, self.dims))
        self.dW = None
        self.db = None

        # this layer calculates : z = W * a_prev + b; a = actFun(z)
        self.a_prev = None
        self.z = None
        self.a = None
        # self.nextLayer.a_prev = self.a  - feedforward responsibility

        # calc & send it back to previous layer
        self.da_prev = None  # backprop responsibility
        # self.da = self.nextLayer.da_prev  - backprop benefit

        # For last layer
        self.probs = None

    # def setNeighbors(self, prevLayer, nextLayer):
    #     self.prevLayer = prevLayer
    #     self.nextLayer = nextLayer
    #     self.nextLayer.a_prev = self.a
    #     self.da = self.nextLayer.da_prev

    def feedforward(self, X, actFun):
        if self.isFirstLayerAfterInput:
            self.z = X.dot(self.W) + self.b
        else:
            # self.z = self.a_prev.dot(self.W) + self.b - if playing with pointers
            self.a_prev = X
            self.z = X.dot(self.W) + self.b

        self.a = actFun(self.z)

        if self.isOutput:
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return self.probs

        return self.a

    def backprop(self, X, y, da):

        if self.isOutput:
            num_examples = len(X)
            delta3 = self.probs
            delta3[range(num_examples), y] -= 1
            self.dW = (self.a_prev.transpose()).dot(delta3)
            self.db = np.sum(delta3, axis=0, keepdims=True)
            self.da_prev = delta3.dot(self.W.transpose())

        elif self.isFirstLayerAfterInput:
            self.da = da
            dz = self.da * diff_actFun(self.a, self.actFun_type)
            dz_w = X.transpose()
            self.dW = np.dot(dz_w, dz)
            self.db = np.sum(dz, axis=0)

        else:
            self.da = da
            dz = np.array(self.da * diff_actFun(self.a, self.actFun_type))
            dz_w = X.transpose()
            self.dW = np.dot(dz_w, dz)
            self.db = np.sum(dz, axis=0)

            dz_a_prev = self.W
            self.da_prev = np.dot(dz_a_prev, dz.transpose()).transpose()

        # Add regularization terms (b doesn't have regularization terms)
        self.dW += self.reg_lambda * self.W

        # Gradient descent parameter update
        self.W += -self.epsilon * self.dW
        self.b += -self.epsilon * self.db

        return self.da_prev


