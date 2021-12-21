import numpy as np  # 넘파이 사용
import matplotlib.pyplot as plt  # 맷플롯립사용
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더


# https://wikidocs.net/57810
# 이진 분류(Binary Classification)라고 합니다.
# 그리고 이진 분류를 풀기 위한 로지스틱 회귀(Logistic Regression)
# sigmoid = H(x) = sigmoid(Wx+b) S자 형태로 그래프 그려주는것
class LogisticRegression:
    def __init__(self):
        return

    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __sigmoid_funtion(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y1 = self.__sigmoid(0.5*x)
        y2 = self.__sigmoid(x)
        y3 = self.__sigmoid(2*x)

        plt.plot(x, y1, 'r', linestyle='--')  # W의 값이 0.5일때
        plt.plot(x, y2, 'g')  # W의 값이 1일때
        plt.plot(x, y3, 'b', linestyle='--')  # W의 값이 2일때
        plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
        plt.title('Sigmoid Function')
        plt.show()

    def __logistic_regression(self):
        x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
        y_data = [[0], [0], [0], [1], [1], [1]]
        x_train = torch.FloatTensor(x_data)
        y_train = torch.FloatTensor(y_data)

        # 모델 초기화
        W = torch.zeros((2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        # optimizer 설정
        optimizer = optim.SGD([W, b], lr=1)

        nb_epochs = 1000
        for epoch in range(nb_epochs + 1):

            # Cost 계산
            hypothesis = torch.sigmoid(x_train.matmul(W) + b)
            # -[y*logH(x) + (1-y)*log(1-H(x))]
            losses = -(y_train * torch.log(hypothesis) +
                       (1 - y_train) * torch.log(1 - hypothesis))
            cost = losses.mean()

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, cost.item()
                ))

        hypothesis = torch.sigmoid(x_train.matmul(W) + b)
        print(hypothesis)
        prediction = hypothesis >= torch.FloatTensor([0.5])
        print(prediction)
        print(W)
        print(b)

    def __nnModule_logistic_regression(self):
        x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
        y_data = [[0], [0], [0], [1], [1], [1]]
        x_train = torch.FloatTensor(x_data)
        y_train = torch.FloatTensor(y_data)

        model = nn.Sequential(
            nn.Linear(2, 1),  # input_dim = 2, output_dim = 1
            nn.Sigmoid()  # 출력은 시그모이드 함수를 거친다
        )
        model(x_train)
        # optimizer 설정
        optimizer = optim.SGD(model.parameters(), lr=1)

        nb_epochs = 1000
        for epoch in range(nb_epochs + 1):

            # H(x) 계산
            hypothesis = model(x_train)

            # cost 계산
            cost = F.binary_cross_entropy(hypothesis, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 20번마다 로그 출력
            if epoch % 10 == 0:
                prediction = hypothesis >= torch.FloatTensor(
                    [0.5])  # 예측값이 0.5를 넘으면 True로 간주
                correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True로 간주
                accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
                print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
                    epoch, nb_epochs, cost.item(), accuracy * 100,
                ))

        model(x_train)
        print(list(model.parameters()))

        return

    def run(self):
        torch.manual_seed(1)
        #self.__sigmoid_funtion()
        #self.__logistic_regression()
        self.__nnModule_logistic_regression()

        return
