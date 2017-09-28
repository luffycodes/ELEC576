from Utils.Plots.decisionBoundary import plot_decision_boundary
from Utils.data import generate_data

import matplotlib.pyplot as plt
from Assignment1.Layer import Layer
import Utils.maths as um
import numpy as np


class DeepNeuralNet(object):
    def __init__(self, dimArray, inputSize, actFun_type='tanh', reg_lambda=0.01):
        self.dimArray = dimArray
        self.inputSize = inputSize
        self.layers = []
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.probs = None

        for i in range(len(dimArray)):
            if i == 0:
                layerObj = Layer(dimArray[i], inputSize, False, None, None, True, 'tanh')
            elif i == len(dimArray) - 1:
                layerObj = Layer(dimArray[i], dimArray[i - 1], True, None, None, False, 'tanh')
            else:
                layerObj = Layer(dimArray[i], dimArray[i - 1], False, None, None, False, 'tanh')

            self.layers.append(layerObj)

            # for i in range(len(dimArray)):
            #     if i == 0:
            #         self.layers[i].setNeighbors(None, self.layers[i + 1])
            #     elif i == len(dimArray) - 1:
            #         self.layers[i].setNeighbors(self.layers[i - 1], None)
            #     else:
            #         self.layers[i].setNeighbors(self.layers[i - 1], self.layers[i + 1])

    def feedforward(self, X, actFun):
        for i in range(len(self.dimArray)):
            if i == 0:
                nextLayerX = self.layers[i].feedforward(X, actFun)
            elif i < len(self.dimArray):
                nextLayerX = self.layers[i].feedforward(nextLayerX, actFun)
                self.probs = nextLayerX

    def backprop(self, X, y):
        for i in range(len(self.dimArray) - 1, -1, -1):
            if i == len(self.dimArray) - 1:
                prevLayer_da = self.layers[i].backprop(X, y, None)
            else:
                prevLayer_da = self.layers[i].backprop(X, y, prevLayer_da)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.backprop(X, y)

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
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    model = DeepNeuralNet([3, 2], 2)
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
