import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import List
from pathlib import Path
from sklearn import metrics, svm, datasets, model_selection, tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans


def run_KNeighbor():
    names = ['sepal-length', 'sepal-width',
             'petal-length', 'petal-width', 'Class']

    data_path = Path(str(Path.cwd())+'/chap03/data/iris.data', names=names)
    dataset = pd.read_csv(data_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    s = StandardScaler()  # 평균 0, 표준편차 1이 되도록 변환
    s.fit(X_train)
    X_train = s.transform(X_train)
    X_test = s.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f"정확도: {accuracy_score(y_test, y_pred)}")

    # 최적의 K찾기
    k = 10
    acc_array = np.zeros(k)
    for k in np.arange(1, k+1, 1):
        classifier: KNeighborsClassifier = KNeighborsClassifier(
            n_neighbors=k).fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        acc_array[k-1] = acc

    max_acc = np.amax(acc_array)
    acc_list = list(acc_array)
    k = acc_list.index(max_acc)
    print(f"정확도 {max_acc}으로 최적의 k는 {k+1}입니다")

    return


def run_SVM():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        iris['data'], iris['target'], test_size=0.6, random_state=42)

    s = svm.SVC(kernel='linear', C=1, gamma=0.5)
    s.fit(X_train, y_train)
    pred = s.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print(f'정확도: {score}')

    return


def run_decision_tree():
    data_path = Path(str(Path.cwd())+'/chap03/data/titanic/train.csv')
    df = pd.read_csv(data_path, index_col="PassengerId")
    print(df.head())

    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df.dropna()
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    split_list = train_test_split(X, y, random_state=1)

    X_train: List[np.array] = split_list[0]
    X_test: List[np.array] = split_list[1]
    y_train: List[np.array] = split_list[2]
    y_test: List[np.array] = split_list[3]

    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    print(f'정확도: {score}')

    # 혼동 행렬을 이용한 성능 측정
    print(pd.DataFrame(
        confusion_matrix(y_test, y_predict),
        columns=['Predicted Not Survival', 'Predicted Survival'],
        index=['True Not Survival', 'True Survival']
    ))

    return

# 최대 우도법?


def run_logisticRegr():
    digits = datasets.load_digits()
    print("Image Data Shape", digits['data'].shape)
    print("Label Data Shape", digits['target'].shape)

    plt.figure(figsize=(20, 4))
    for index, (image, label) in enumerate(zip(digits['data'][0:5], digits['target'][0:5])):
        plt.subplot(1, 5, index+1)
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title(f'Training: {label}\n')

    plt.show()

    split_list = train_test_split(
        digits['data'], digits['target'], test_size=0.25, random_state=0)

    X_train: np.ndarray = split_list[0]
    X_test: np.ndarray = split_list[1]
    y_train: np.ndarray = split_list[2]
    y_test: np.ndarray = split_list[3]

    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)

    print(logisticRegr.predict(X_test[0].reshape(1, -1)))
    print(logisticRegr.predict(X_test[0:10]))

    predictions = logisticRegr.predict(X_test)
    score = logisticRegr.score(X_test, y_test)
    print(score)

    # 혼동 행렬
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt='0.3f', linewidths=0.5,
                square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = f'Accuracy Score: {score}'
    plt.title(all_sample_title, size=15)
    plt.show()

    return


def run_linearRegr():
    data_path = Path(str(Path.cwd())+'/chap03/data/weather.csv')
    dataset = pd.read_csv(data_path)
    dataset.plot(x='MinTemp', y='MaxTemp', style='o')
    plt.title('MinTemp vs MaxTemp')
    plt.xlabel('MinTemp')
    plt.ylabel('MaxTemp')
    plt.show()

    X = dataset['MinTemp'].values.reshape(-1, 1)
    y = dataset['MaxTemp'].values.reshape(-1, 1)

    split_list = train_test_split(
        X, y, test_size=0.2)

    X_train: np.ndarray = split_list[0]
    X_test: np.ndarray = split_list[1]
    y_train: np.ndarray = split_list[2]
    y_test: np.ndarray = split_list[3]

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred: np.ndarray = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(),
                      'Predicted': y_pred.flatten()})

    print(df)

    plt.scatter(X_test, y_test, color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()

    print(f'평균제곱법: {metrics.mean_squared_error(y_test,y_pred)}')
    print(f'루트 평균제곱법: {np.sqrt(metrics.mean_squared_error(y_test,y_pred))}')

    return


def run_KMean():
    data_path = Path(str(Path.cwd())+'/chap03/data/sales_data.csv')
    data = pd.read_csv(data_path)
    print(data.head())
    categorical_features = ['Channel', 'Region']
    continuous_features = ['Fresh', 'Milk', 'Grocery',
                           'Frozen', 'Detergents_Paper', 'Delicassen']

    for col in categorical_features:
        dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)

    print(data.head())
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    # 적당한 K 값 추출
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km: KMeans = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Optimal k')
    plt.show()

    return


def run():

    # 지도학습
    # run_KNeighbor()
    # run_SVM()
    # run_decision_tree()
    # run_logisticRegr()
    # run_linearRegr()

    # 비지도학습
    run_KMean()

    return
