from operator import mod
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics, svm, datasets, model_selection, tree
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
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


def run():

    run_KNeighbor()
    run_SVM()
    run_decision_tree()

    return
