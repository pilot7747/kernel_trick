import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import itertools


class BestFeaturesSelection:
    def __init__(self, estimator, scoring, parameters=dict(),
                 test_size=0.3, random_state=17, minimize=True):
        """
        Отбор наилучших признаков
        estimator: конструктор класса, например, LinearRegression
        paramters: параметры, передаваемые конструктору estimator,
                    например dict(fit_intercept=False)
        scoring: функция риска, например, mean_squared_error
        minimize: минимизировать ли функционал качества
                    (иначе - максимизировать)
        """

        self.estimator = estimator
        self.parameters = parameters
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.minimize = minimize

    def fit(self, X, y):
        """
        Подбор лучшего подмножества признаков
        и обучение модели на нём
        """
        self._columns = np.array(list(X.columns))
        # разделение выборки на test и train. Не перепутайте порядок !
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.3, random_state=17)

        self.results_ = []  # список пар (вектор использованных признаков,
        # значение функции потерь)
        features_count = X.shape[1]

        for bitmask in tqdm_notebook(range(1, 2 ** features_count)):
            subset = [i == "1" for i in
                      np.binary_repr(bitmask, width=features_count)]
            # binary_repr возвращает строку длины width с двоичным
            # представлением числа и ведущими нулями

            estimator = self.estimator(**self.parameters)
            estimator.fit(X_train[:, subset], y_train)
            # вычисление качества модели
            score = self.scoring(estimator.predict(X_test[:, subset]), y_test)

            self.results_.append((subset, score))

        self.results_.sort(key=lambda pair: pair[1],
                           reverse=not self.minimize)
        # сортируем по второму элементу в нужном порядке

        self._best_subset = self.results_[0][0]
        self._best_estimator = self.estimator(**self.parameters)
        self._best_estimator.fit(X_train[:, self._best_subset], y_train)
        self._best_columns = list(self._columns[self._best_subset])

        return self._best_estimator

    def predict(self, X):
        """
        Предсказание модели,
        обученной на наилучшем подмножестве признаков.
        """

        return self._best_estimator.predict(X.values[:, self._best_subset])


class KernelTrick(object):
    def __init__(self, model=LinearRegression, metric=mean_squared_error, feature_selection=None,  *args):
        self.feature_selection = feature_selection
        self.model = model
        self.metric = mean_squared_error
        self.args = args

    def __fit_polynom2(self, data):
        result = data.copy()
        for column in list(data.columns):
            result['{}^2'.format(column)] = data[column].values ** 2
        for col1, col2 in itertools.combinations(list(data.columns), 2):
            result['{}*{}'.format(col1, col2)] = data[col1].values * data[col2].values
        return result

    def __fit_gaussian(self, data):
        return None

    def fit(self, data, y, mode='polynom2'):
        if mode == 'polynom2':
            X = self.__fit_polynom2(data)
            if self.feature_selection is not None:
                selector = SelectFromModel(self.model(self.args).fit(X, y))
                X = selector.transform(X)

            model = self.model(self.args)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model.fit(X_train, y_train)
            return model

        if mode == 'gaussian':
            return self.__fit_gaussian(data)
