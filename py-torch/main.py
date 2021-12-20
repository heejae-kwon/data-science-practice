import py_torch
from py_torch.LinearRegression  import LinearRegression
from py_torch.LogisticRegression import LogisticRegression

def main():
    linear_regression = LinearRegression()
    #linear_regression.run()

    logistic_regression = LogisticRegression()
    logistic_regression.run()

    return


main()
