import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더


class LinearRegression:
    def __init__(self):
        return

    # y = Wx + b
    def __linear_regression(self):
        # 데이터
        x_train = torch.FloatTensor([[1], [2], [3]])
        y_train = torch.FloatTensor([[2], [4], [6]])
        # 모델 초기화
        W = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        # optimizer 설정
        # SGD == Gradient Descent 의 한가지
        # lr == learning rate
        optimizer = optim.SGD([W, b], lr=0.01)

        nb_epochs = 1999  # 원하는만큼 경사 하강법을 반복
        for epoch in range(nb_epochs + 1):

            # H(x) 계산
            hypothesis = x_train * W + b

           # cost 계산
            cost = torch.mean((hypothesis - y_train) ** 2)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                    epoch, nb_epochs, W.item(), b.item(), cost.item()
                ))
        return

    def __autograd(self):
        # 2w^2+5
        w = torch.tensor(2.0, requires_grad=True)
        y = w**2
        z = 2*y + 5
        z.backward()
        print(f'수식을 w로 미분한 값 : {w.grad}')

        return

    def __multivariable_linear_regression(self):
        # H(x) = w1x1 + w2x2 + w3x3 + b
        # 훈련 데이터
        """ version1
        x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
        x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
        x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
        y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
        # 가중치 w와 편향 b 초기화
        w1 = torch.zeros(1, requires_grad=True)
        w2 = torch.zeros(1, requires_grad=True)
        w3 = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        # optimizer 설정
        optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

        nb_epochs = 1000
        for epoch in range(nb_epochs + 1):

           # H(x) 계산
            hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

            # cost 계산
            cost = torch.mean((hypothesis - y_train) ** 2)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 100번마다 로그 출력
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
                    epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
                ))
        """
        x_train = torch.FloatTensor([[73,  80,  75],
                                     [93,  88,  93],
                                     [89,  91,  80],
                                     [96,  98,  100],
                                     [73,  66,  70]])
        y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

        # 모델 초기화
        W = torch.zeros((3, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        # optimizer 설정
        optimizer = optim.SGD([W, b], lr=1e-5)
        nb_epochs = 20

        for epoch in range(nb_epochs + 1):

            # H(x) 계산
            # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
            hypothesis = x_train.matmul(W) + b

            # cost 계산
            cost = torch.mean((hypothesis - y_train) ** 2)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
                epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
            ))

        return

    def __nnModule_linear_regression(self):
        # 데이터
        x_train = torch.FloatTensor([[1], [2], [3]])
        y_train = torch.FloatTensor([[2], [4], [6]])
        # 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
        model = nn.Linear(1, 1)
        # parameter[0] = W, parameter[1] = b
        print(list(model.parameters()))
        # optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
        nb_epochs = 2000
        for epoch in range(nb_epochs+1):
            # H(x) 계산
            prediction = model(x_train)

            # cost 계산
            # <== 파이토치에서 제공하는 평균 제곱 오차 함수
            cost = F.mse_loss(prediction, y_train)

            # cost로 H(x) 개선하는 부분
            # gradient를 0으로 초기화
            optimizer.zero_grad()
            # 비용 함수를 미분하여 gradient 계산
            cost.backward()  # backward 연산
            # W와 b를 업데이트
            optimizer.step()

            if epoch % 100 == 0:
                # 100번마다 로그 출력
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, cost.item()
                ))

        # 임의의 입력 4를 선언
        new_var = torch.FloatTensor([[4.0]])
        # 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
        pred_y = model(new_var)  # forward 연산
        # y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
        print("훈련 후 입력이 4일 때의 예측값 :", pred_y)
        print(list(model.parameters()))

        return

    def __nnModule_multivariable_linear_regression(self):
        x_train = torch.FloatTensor([[73,  80,  75],
                                     [93,  88,  93],
                                     [89,  91,  80],
                                     [96,  98,  100],
                                     [73,  66,  70]])
        y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

        # 모델 초기화
        model = nn.Linear(3, 1)
        # optimizer 설정
        optimizer = optim.SGD(model.parameters(), lr=1e-5)

        nb_epochs = 20
        for epoch in range(nb_epochs + 1):
            # H(x) 계산
            prediction = model(x_train)
            # model(x_train)은 model.forward(x_train)와 동일함.

            # cost 계산
            cost = F.mse_loss(prediction, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if epoch % 100 == 0:
                # 100번마다 로그 출력
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, cost.item()
                ))

        new_var = torch.FloatTensor([73, 80, 75])
        pred_y = model(new_var)
        print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
        print(list(model.parameters()))

        return

    def __minibatch(self):
        x_train = torch.FloatTensor([[73,  80,  75],
                                     [93,  88,  93],
                                     [89,  91,  90],
                                     [96,  98,  100],
                                     [73,  66,  70]])
        y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

        nb_epochs = 20
        for epoch in range(nb_epochs + 1):
            for batch_idx, samples in enumerate(dataloader):
                # print(batch_idx)
                # print(samples)
                x_train, y_train = samples
                # H(x) 계산
                prediction = model(x_train)

                # cost 계산
                cost = F.mse_loss(prediction, y_train)

                # cost로 H(x) 계산
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, batch_idx+1, len(dataloader),
                    cost.item()
                ))

        # 임의의 입력 [73, 80, 75]를 선언
        new_var = torch.FloatTensor([[73, 80, 75]])
        # 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
        pred_y = model(new_var)
        print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)

        return

    def run(self):
        torch.manual_seed(1)
        self.__linear_regression()
        self.__autograd()
        self.__multivariable_linear_regression()
        self.__nnModule_linear_regression()
        self.__nnModule_multivariable_linear_regression()
        self.__minibatch()
        #custom dataset, custom model skip
        return
