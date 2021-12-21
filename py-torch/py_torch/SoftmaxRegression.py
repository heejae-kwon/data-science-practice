import numpy as np  # 넘파이 사용
import matplotlib.pyplot as plt  # 맷플롯립사용
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더


# https://wikidocs.net/59678
# 각 클래스 간의 오차는 균등한 것이 옳다. 제곱 오차가 오류 남 ex) 1 2 3 4
# 원-핫 벡터의 관계의 무작위성은 때로는 단어의 유사성을 구할 수 없다는 단점으로 언급되기도 합니다.
# Multi-Class Classification
class SoftmaxRegression:
    def __init__(self):
        self.x_train = [[1, 2, 1, 1],
                        [2, 1, 3, 2],
                        [3, 1, 3, 4],
                        [4, 1, 5, 5],
                        [1, 7, 5, 5],
                        [1, 2, 5, 6],
                        [1, 6, 6, 6],
                        [1, 7, 7, 7]]
        self.y_train = [2, 2, 2, 1, 1, 1, 0, 0]
        return

    # 복습~
    def __cost_function(self):
        z = torch.FloatTensor([1, 2, 3])
        hypothesis = F.softmax(z, dim=0)
        print(hypothesis)
        hypothesis.sum()

        # random tensor 3*5
        z = torch.rand(3, 5, requires_grad=True)
        hypothesis = F.softmax(z, dim=1)
        print(hypothesis)
        # 임의의 레이블
        y = torch.randint(5, (3,)).long()
        print(y)
        # 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
        y_one_hot = torch.zeros_like(hypothesis)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)

        cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
        print(cost)

        # Low level
        # 첫번째 수식
        (y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

        # 두번째 수식
        (y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()

        # High level
        # 세번째 수식
        F.nll_loss(F.log_softmax(z, dim=1), y)

        # 네번째 수식
        F.cross_entropy(z, y)

        return

    def __softmax_regression_low_level(self):

        x_train = torch.FloatTensor(self.x_train)
        y_train = torch.LongTensor(self.y_train)

        # 8 * 3 one_hot
        y_one_hot = torch.zeros(8, 3)
        y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
        print(y_one_hot.shape)

        # 모델 초기화
        W = torch.zeros((4, 3), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        # optimizer 설정
        optimizer = optim.SGD([W, b], lr=0.1)

        nb_epochs = 1000
        for epoch in range(nb_epochs + 1):

            # 가설
            hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

            # 비용 함수
            cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, cost.item()
                ))
        return

    def __softmax_regression_high_level(self):

        x_train = torch.FloatTensor(self.x_train)
        y_train = torch.LongTensor(self.y_train)

        # 8 * 3 one_hot
        y_one_hot = torch.zeros(8, 3)
        y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
        print(y_one_hot.shape)

        # 모델 초기화
        W = torch.zeros((4, 3), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        # optimizer 설정
        optimizer = optim.SGD([W, b], lr=0.1)

        nb_epochs = 1000
        for epoch in range(nb_epochs + 1):

            # Cost 계산
            z = x_train.matmul(W) + b
            cost = F.cross_entropy(z, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, cost.item()
                ))

        return

    def __nnModule_softmax_regression(self):

        x_train = torch.FloatTensor(self.x_train)
        y_train = torch.LongTensor(self.y_train)

        # 8 * 3 one_hot
        y_one_hot = torch.zeros(8, 3)
        y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
        print(y_one_hot.shape)

        # 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3.
        model = nn.Linear(4, 3)
        # optimizer 설정
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        nb_epochs = 1000
        for epoch in range(nb_epochs + 1):

            # H(x) 계산
            prediction = model(x_train)

            # cost 계산
            cost = F.cross_entropy(prediction, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 20번마다 로그 출력
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, cost.item()
                ))

        return

    def run(self):
        torch.manual_seed(1)
        self.__cost_function()
        self.__softmax_regression_low_level()
        self.__softmax_regression_high_level()
        self.__nnModule_softmax_regression()

        return
