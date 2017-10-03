from Utils.Plots.decisionBoundary import plot_decision_boundary
from Utils.data import generate_moons_data, generate_blob_data

import matplotlib.pyplot as plt
from Assignment1.Layer import Layer
import Utils.maths as um
import numpy as np


class DeepNeuralNet(object):
    def __init__(self, dimArray, inputSize, actFun_type, reg_lambda=0.01):
        self.dimArray = dimArray
        self.inputSize = inputSize
        self.layers = []
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.probs = None

        for i in range(len(dimArray)):
            if i == 0:
                layerObj = Layer(i, dimArray[i], inputSize, False, True)
            elif i == len(dimArray) - 1:
                layerObj = Layer(i, dimArray[i], dimArray[i - 1], True, False)
            else:
                layerObj = Layer(i, dimArray[i], dimArray[i - 1], False, False)

            self.layers.append(layerObj)

    def feedforward(self, X, actFun):
        for i in range(len(self.dimArray)):
            if i == 0:
                nextLayerX = self.layers[i].feedforward(X, actFun)
            elif i < len(self.dimArray):
                nextLayerX = self.layers[i].feedforward(nextLayerX, actFun)
                self.probs = nextLayerX

    def backprop(self, X, y, diff_actFun):
        for i in range(len(self.dimArray) - 1, -1, -1):
            if i == len(self.dimArray) - 1:
                prevLayer_da = self.layers[i].backprop(X, y, None, diff_actFun)
            else:
                prevLayer_da = self.layers[i].backprop(X, y, prevLayer_da, diff_actFun)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.backprop(X, y, lambda x: self.diff_actFun(x, type=self.actFun_type))

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def actFun(self, z, type):
        if type == 'tanh':
            return um.tanh(z)
        if type == 'sigmoid':
            return um.sigmoid(z)
        if type == 'relu':
            return um.relu(z)

        return None

    def diff_actFun(self, z, type):
        if type == 'tanh':
            return um.diff_tanh(z)
        if type == 'sigmoid':
            return um.diff_sigmoid(z)
        if type == 'relu':
            return um.diff_relu(z)

        return None

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        logProbs = np.log(self.probs)
        data_loss = sum(logProbs[:, 0] * y) + sum(logProbs[:, 1] * (1 - y))

        return (1. / num_examples) * data_loss

    def predict(self, X):
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    X, y = generate_blob_data()
    X, y = generate_moons_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    model = DeepNeuralNet([2, 3, 2], 2, actFun_type='sigmoid')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
