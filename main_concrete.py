from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

import numpy
import matplotlib.pyplot as plt

import read_concrete
import vr_saga_reg
import vr_svrg_reg

dim, X, y = read_concrete.read_concrete()

kf = KFold(n_splits = 5)
for tv_index, test_index in kf.split(X):
    X_tv, X_test = X[tv_index], X[test_index]
    y_tv, y_test = y[tv_index], y[test_index]

    rnd = 3; size_tv = len(y_tv)

    saga = vr_saga_reg.saga_estimator(dim = dim, round = rnd)
    svrg = vr_svrg_reg.svrg_estimator(dim = dim, round = rnd)

    saga_params = {'step_size': [0.1, 0.2, 0.3, 0.4, 0.5]}
    svrg_params = {'step_size': [0.1, 0.2, 0.3, 0.4, 0.5]}

    cv_saga = GridSearchCV(estimator = saga, param_grid = saga_params, cv = 8)
    cv_svrg = GridSearchCV(estimator = svrg, param_grid = svrg_params, cv = 8)

    cv_saga.fit(X_tv, y_tv)
    cv_svrg.fit(X_tv, y_tv)

    saga_plot = cv_saga.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)
    svrg_plot = cv_svrg.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)

    plt.ylabel('Test MSE')
    plt.xlabel('Number of passes through data')

    times = [1.0 / size * i for i in range(rnd * size)]
    plt.plot(times, saga_plot, 'r-', 'SAGA',
             times, svrg_plot, 'b-', 'SVRG')
