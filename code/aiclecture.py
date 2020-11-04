"""Pythonを用いた初心者向けAI実践講座 ~中級編~ モジュール"""

import numpy as np

## 重回帰
class LinearRegression:
    def __init__(self):
        self.w_ = None

    def fit(self, X, t):  ## 学習用関数
        X = np.insert(X, 0, 1, axis=1)
        self.w_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X): ## 予測用関数
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w_)


## パーセプトロン
class Perceptron:
    def __init__(self, eta, epoch):
        self.w = [0.5, 0.5, 0.5]
        self.eta = eta
        self.epoch = epoch

    def __z_calc(self, X, i):
        return X[i].dot(self.w)

    def __step(self, z):
        if z > 0:
            return 1
        else:
            return 0

    def train(self, X, y):
        for i in range(len(y)):
            for _ in range(self.epoch):
                z = self.__z_calc(X, i)
                y_output = self.__step(z)
                self.w = self.w + self.eta * (y[i] - y_output) * X[i]

    def accuracy(self, X, y):
        y_pred = []
        count = 0
        for i in range(len(y)):
            z = self.__z_calc(X, i)
            y_output = self.__step(z)
            y_pred.append(y_output)
        for j in range(len(y)):
            if y[j] == y_pred[j]:
                count += 1
        acc = count / len(y)
        print("正解ラベル: ", y)
        print("予測ラベル: ", y_pred)
        print("精度: ", acc)


## ロジスティック回帰
class LogisticRegression:
    def __init__(self, eta=1e-3, epoch=10000, data=None):
        self.eta = eta
        self.epoch = epoch
        self.data = data

    def __preprocessing(self, X):  ## 切片追加処理
        app = X.tolist()
        for i in range(X.shape[0]):
            app[i].append(1.0)
        return np.array(app)

    def __sigmoid(self, X):  ## シグモイドの計算
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    def __binary_cross_entropy(self, p, target):  ## 損失関数の計算
        return target.dot(np.log(p)) + (1 - target).dot(np.log(1 - p))

    def fit(self, X, target, display_loss='off'):  ## 最適化
        try:
            X = self.__preprocessing(X)
            self.w = np.random.rand(len(self.data.feature_names) + 1)
            for i in range(self.epoch):
                p = self.__sigmoid(X)
                loss = self.__binary_cross_entropy(p, target)
                if display_loss == 'on' and i % 100 == 0:
                    print("Loss_value: ", loss)
                self.w = self.w - self.eta * (X.T.dot(p - target))
        except AttributeError as e:
            print('Catch AttributeError : ', e)
            print('Please set data to lr = LogisticRegression(data="XXX")')

    def predict(self, X):  ## 予測
        X = self.__preprocessing(X)
        y_pred = 1 / (1 + np.exp(-np.dot(X, self.w)))
        app = []
        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                app.append(1)
            else:
                app.append(0)
        return app

    def accuracy(self, y_pred, target):  ## 精度評価
        count = 0
        for i in range(len(target)):
            if y_pred[i] == target[i]:
                count += 1
        print('精度: ', count / len(target))


## サポートベクトルマシン(線形)
class LinearSVM():
    def __init__(self, X, t):
        N = len(t)
        self.X, self.t = X, t
        self.alpha = np.zeros(N)
        self.eta = 1.0e-4
        self.beta = 1
        self.epoch= 1000

    def _preprocessing(self):
        app = []
        for i in range(len(self.t)):
        if self.t[i] == 1:
            app.append(1)
        else:
            app.append(-1)
        self.T = np.array(app)

    def _H(self):
        app = [] ## H作成
        for j in range(len(self.T)):
        col = []
        for i in range(len(self.T)):
            dot = (self.T[j] * self.T[i]) * self.X[j].dot(self.X[i])
            col.append(dot)
        app.append(col)
        return np.array(app)

    def _t_t(self):
        app = [] ## t*t^T作成
        for j in range(len(self.T)):
        col = []
        for i in range(len(self.T)):
            dot = self.T[j] * self.T[i]
            col.append(dot)
        app.append(col)
        return np.array(app)

    def fit(self):
        self._preprocessing()
        self.H = self._H()
        self.t_t = self._t_t()
        for _ in range(self.epoch): ## 勾配降下法
        self.alpha = self.alpha + self.eta*(1 - self.H.dot(self.alpha) - self.beta*self.t_t.dot(self.alpha))
        ##print("alpha = ", self.alpha)

    def predict(self, X):
        w = np.array([0.0,0.0,0.0,0.0])
        for k in range(len(self.T)):
        w += self.alpha[k]*self.T[k]*self.X[k]
        b = 0
        for i in range(len(self.T)):
        b += self.T[i] - w.dot(self.X[i])
        b = b / len(self.T)
        y = X.dot(w) + b
        app = []
        for i in range(len(y)):
        if y[i] >= 0:
            app.append(1)
        else:
            app.append(0)
        return np.array(app)

    def accuracy(self, y_pred, target): ## 精度評価
        count = 0
        for i in range(len(target)):
        if y_pred[i] == target[i]:
            count += 1
        print('精度: ', count / len(target))