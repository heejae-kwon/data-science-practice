import py_torch
from py_torch.LinearRegression  import LinearRegression
from py_torch.LogisticRegression import LogisticRegression
from py_torch.SoftmaxRegression import SoftmaxRegression

def main():
    linear_regression = LinearRegression()
    #linear_regression.run()

    logistic_regression = LogisticRegression()
    #logistic_regression.run()

    softmax_regression = SoftmaxRegression()
    softmax_regression.run()

    return


main()
