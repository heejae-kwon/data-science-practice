import numpy as np  # 넘파이 사용
import matplotlib.pyplot as plt  # 맷플롯립사용
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# https://wikidocs.net/60680
# 퍼셉트론(Perceptron)은 프랑크 로젠블라트(Frank Rosenblatt)가
# 1957년에 제안한 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘
# 퍼셉트론(Perceptron)은 프랑크 로젠블라트(Frank Rosenblatt)가 1957년에 제안한 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘
class Perceptron:
    def __init__(self):
        return

    # 단층 perceptron 은 XOR 을 못만듬. 단층은 선형인데 XOR은 비선형이 필요해서
    def __AND_gate(self, x1, x2):
        w1 = 0.5
        w2 = 0.5
        b = -0.7
        result = x1*w1 + x2*w2 + b
        if result <= 0:
            return 0
        else:
            return 1

    def __NAND_gate(self, x1, x2):
        w1 = -0.5
        w2 = -0.5
        b = 0.7
        result = x1*w1 + x2*w2 + b
        if result <= 0:
            return 0
        else:
            return 1

    def __OR_gate(self, x1, x2):
        w1 = 0.6
        w2 = 0.6
        b = -0.5
        result = x1*w1 + x2*w2 + b
        if result <= 0:
            return 0
        else:
            return 1

    # 단층에선 XOR 불가능
    def __single_layer_perceptron_XOR_gate(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(777)
        if device == 'cuda':
            torch.cuda.manual_seed_all(777)

        X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
        Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
        linear = nn.Linear(2, 1, bias=True)
        sigmoid = nn.Sigmoid()
        model = nn.Sequential(linear, sigmoid).to(device)

        # 비용 함수와 옵티마이저 정의
        criterion = torch.nn.BCELoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)

        # 10,001번의 에포크 수행. 0번 에포크부터 10,000번 에포크까지.
        for step in range(10001):
            optimizer.zero_grad()
            hypothesis = model(X)

            # 비용 함수
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            if step % 100 == 0:  # 100번째 에포크마다 비용 출력
                print(step, cost.item())

        with torch.no_grad():
            hypothesis = model(X)
            predicted = (hypothesis > 0.5).float()
            accuracy = (predicted == Y).float().mean()
            print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
            print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
            print('실제값(Y): ', Y.cpu().numpy())
            print('정확도(Accuracy): ', accuracy.item())

            return

    def __multi_layer_perceptron_XOR_gate(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # for reproducibility
        torch.manual_seed(777)
        if device == 'cuda':
            torch.cuda.manual_seed_all(777)

        X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
        Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

        model = nn.Sequential(
            # input_layer = 2, hidden_layer1 = 10
            nn.Linear(2, 10, bias=True),
            nn.Sigmoid(),
            # hidden_layer1 = 10, hidden_layer2 = 10
            nn.Linear(10, 10, bias=True),
            nn.Sigmoid(),
            # hidden_layer2 = 10, hidden_layer3 = 10
            nn.Linear(10, 10, bias=True),
            nn.Sigmoid(),
            # hidden_layer3 = 10, output_layer = 1
            nn.Linear(10, 1, bias=True),
            nn.Sigmoid()
        ).to(device)

        criterion = torch.nn.BCELoss().to(device)
        # modified learning rate from 0.1 to 1
        optimizer = torch.optim.SGD(model.parameters(), lr=1)

        for epoch in range(10001):
            optimizer.zero_grad()
            # forward 연산
            hypothesis = model(X)

            # 비용 함수
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            # 100의 배수에 해당되는 에포크마다 비용을 출력
            if epoch % 100 == 0:
                print(epoch, cost.item())

        with torch.no_grad():
            hypothesis = model(X)
            predicted = (hypothesis > 0.5).float()
            accuracy = (predicted == Y).float().mean()
            print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
            print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
            print('실제값(Y): ', Y.cpu().numpy())
            print('정확도(Accuracy): ', accuracy.item())

        return

    def __multi_layer_perceptron_labelling(self):
        digits = load_digits()  # 1,979개의 이미지 데이터 로드
        print(digits.images[0])
        print(digits.target[0])
        print('전체 샘플의 수 : {}'.format(len(digits.images)))
        images_and_labels = list(zip(digits.images, digits.target))
        # 5개의 샘플만 출력
        for index, (image, label) in enumerate(images_and_labels[:5]):
            plt.subplot(2, 5, index + 1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('sample: %i' % label)

        for i in range(5):
            print(i, '번 인덱스 샘플의 레이블 : ', digits.target[i])

        print(digits.data[0])
        X = digits.data  # 이미지. 즉, 특성 행렬
        Y = digits.target  # 각 이미지에 대한 레이블

        model = nn.Sequential(
            nn.Linear(64, 32),  # input_layer = 64, hidden_layer1 = 32
            nn.ReLU(),
            nn.Linear(32, 16),  # hidden_layer2 = 32, hidden_layer3 = 16
            nn.ReLU(),
            nn.Linear(16, 10)  # hidden_layer3 = 16, output_layer = 10
        )

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.int64)

        loss_fn = nn.CrossEntropyLoss()  # 이 비용 함수는 소프트맥스 함수를 포함하고 있음.
        optimizer = optim.Adam(model.parameters())
        losses = []

        for epoch in range(100):
            optimizer.zero_grad()
            y_pred = model(X)  # forwar 연산
            loss = loss_fn(y_pred, Y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, 100, loss.item()
                ))

            losses.append(loss.item())

        plt.plot(losses)

        return

    def __multi_layer_perceptron_MNIST(self):
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        print((mnist.data.to_numpy())[0])
        print((mnist.target.to_numpy())[0])
        X = mnist.data.to_numpy() / 255  # 0-255값을 [0,1] 구간으로 정규화
        y = mnist.target.to_numpy()
        X[0]
        y[0]
        plt.imshow(X[0].reshape(28, 28), cmap='gray')
        print(f"이 이미지 데이터의 레이블은 {y[0]}이다")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1/7, random_state=0)

        print(X_train)
        print(y_train)

        X_train = torch.Tensor(X_train)
        X_test = torch.Tensor(X_test)
        y_train = torch.LongTensor(y_train.astype(np.float64))
        y_test = torch.LongTensor(y_test.astype(np.float64))

        ds_train = TensorDataset(X_train, y_train)
        ds_test = TensorDataset(X_test, y_test)

        loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
        loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(28*28*1, 100))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(100, 100))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc3', nn.Linear(100, 10))

        print(model)

        # 오차함수 선택
        loss_fn = nn.CrossEntropyLoss()

        # 가중치를 학습하기 위한 최적화 기법 선택
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        def train(epoch):
            model.train()  # 신경망을 학습 모드로 전환

           # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
            for data, targets in loader_train:

                optimizer.zero_grad()  # 경사를 0으로 초기화
                outputs = model(data)  # 데이터를 입력하고 출력을 계산
                loss = loss_fn(outputs, targets)  # 출력과 훈련 데이터 정답 간의 오차를 계산
                loss.backward()  # 오차를 역전파 계산
                optimizer.step()  # 역전파 계산한 값으로 가중치를 수정

                print("epoch{}：완료\n".format(epoch))

        def test():
            model.eval()  # 신경망을 추론 모드로 전환
            correct = 0

            # 데이터로더에서 미니배치를 하나씩 꺼내 추론을 수행
            with torch.no_grad():  # 추론 과정에는 미분이 필요없음
                for data, targets in loader_test:

                    outputs = model(data)  # 데이터를 입력하고 출력을 계산

                    # 추론 계산
                    # 확률이 가장 높은 레이블이 무엇인지 계산
                    _, predicted = torch.max(outputs.data, 1)
                    # 정답과 일치한 경우 정답 카운트를 증가
                    correct += predicted.eq(targets.data.view_as(predicted)).sum()

            # 정확도 출력
            data_num = len(loader_test.dataset)  # 데이터 총 건수
            print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
                                                                 data_num, 100. * correct / data_num))

        test()
        for epoch in range(3):
            train(epoch)

        test()

        index = 2018

        model.eval()  # 신경망을 추론 모드로 전환
        data = X_test[index]
        output = model(data)  # 데이터를 입력하고 출력을 계산
        _, predicted = torch.max(output.data, 0)  # 확률이 가장 높은 레이블이 무엇인지 계산

        print("예측 결과 : {}".format(predicted))

        X_test_show = (X_test[index]).numpy()
        plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
        print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다".format(y_test[index]))

        return

    # 활성화 함수의 특징은 선형 함수가 아닌 비선형 함수여야 한다는 점입니다.
    # 즉, 선형 함수로는 은닉층을 여러번 추가하더라도 1회 추가한 것과 차이를 줄 수 없습니다
    # 주황색 부분은 기울기를 계산하면 0에 가까운 아주 작은 값이 나오게 됩니다.
    # 그런데 역전파 과정에서 0에 가까운 아주 작은 기울기가 곱해지게 되면, 앞단에는 기울기가 잘 전달되지 않게 됩니다.
    # 이러한 현상을 기울기 소실(Vanishing Gradient) 문제라고 합니다.
    # 비선형활성화함수 : https://wikidocs.net/60683

    def run(self):
        # self.__single_layer_perceptron_XOR_gate()
        # self.__multi_layer_perceptron_XOR_gate()
        # self.__multi_layer_perceptron_labelling()
        self.__multi_layer_perceptron_MNIST()

        return
