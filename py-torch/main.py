import py_torch
from py_torch.LinearRegression import LinearRegression
from py_torch.LogisticRegression import LogisticRegression
from py_torch.SoftmaxRegression import SoftmaxRegression
from py_torch.Perceptron import Perceptron
from py_torch.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork


def main():
    linear_regression = LinearRegression()
    # linear_regression.run()

    logistic_regression = LogisticRegression()
    # logistic_regression.run()

    softmax_regression = SoftmaxRegression()
    # softmax_regression.run()

    perceptron = Perceptron()
    #perceptron.run()

    convolutional_neural_network = ConvolutionalNeuralNetwork()
    convolutional_neural_network.run()

    return


main()
