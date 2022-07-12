import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from ml_pytorch_scikit_learn.chapter3.LogisticRegressionGD import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# decision trees, SVM, SGD, logistic, KNN, random forest

def kernelSVM():
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                            X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)
    plt.scatter(X_xor[y_xor == 1, 0],
                 X_xor[y_xor == 1, 1],
                 c='royalblue', marker='s',
                 label='Class 1')
    plt.scatter(X_xor[y_xor == 0, 0],
                 X_xor[y_xor == 0, 1],
                 c='tomato', marker='o',
                 label='Class 0')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    return

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')

def run():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))


    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std,
                        y=y_combined,
                        classifier=ppn,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


    # 직접만듬
    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    lrgd = LogisticRegressionGD(eta=0.3,
                                n_iter=1000,
                                random_state=1)
    lrgd.fit(X_train_01_subset,
            y_train_01_subset)
    plot_decision_regions(X=X_train_01_subset,
                        y=y_train_01_subset,
                        classifier=lrgd)
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    #scikit-learn
    lr = LogisticRegression(C=100.0, solver='lbfgs',
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
                        y_combined,
                        classifier=lr,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    print(lr.predict_proba(X_test_std[:3, :]))

    # support vector machine
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
                        y_combined,
                        classifier=svm,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    #SGD version
    ppn = SGDClassifier(loss='perceptron')
    lr = SGDClassifier(loss='log')
    svm = SGDClassifier(loss='hinge')

    kernelSVM()

    #apply an RBF kernelSVM to Iris flower
    svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
                           y_combined, classifier=svm,
                           test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    #gamma = 100
    svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
                        y_combined, classifier=svm,
                        test_idx=range(105,150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    #decision tree
    tree_model = DecisionTreeClassifier(criterion='gini',
                                        max_depth=4,
                                        random_state=1)
    tree_model.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined,
                        y_combined,
                        classifier=tree_model,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    feature_names = ['Sepal length', 'Sepal width',
                    'Petal length', 'Petal width']
    tree.plot_tree(tree_model,
                    feature_names=feature_names,
                    filled=True)
    plt.show()

    #random forest
    forest = RandomForestClassifier(n_estimators=25,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined,
                        classifier=forest, test_idx=range(105,150))
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
