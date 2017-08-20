from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

import numpy
import matplotlib.pyplot as plt

import read_concrete
import vr_saga_reg
import vr_svrg_reg
import vr_sgld_reg

dim, X, y = read_concrete.read_concrete()

kf = KFold(n_splits = 5)
for tv_index, test_index in kf.split(X):
    X_tv, X_test = X[tv_index], X[test_index]
    y_tv, y_test = y[tv_index], y[test_index]

    rnd = 3; size_tv = len(y_tv)

    #saga = vr_saga_reg.saga_estimator(dim = dim, round = rnd)
    #svrg = vr_svrg_reg.svrg_estimator(dim = dim, round = rnd)
    sgld = vr_sgld_reg.sgld_estimator(dim = dim, round = rnd)

    #saga_params = {'step_size': [0.00000625, 0.0000065, 0.000000675], \
    #                'temp': [6.75, 7, 7.25]}
    #svrg_params = {'step_size': [0.000003, 0.000004, 0.000005], \
    #                'temp': [5.5, 6, 6.5]}
    sgld_params = {'step_size': [0.0000000000001, 0.0000000000005, 0.000000000001, 0.000000000005, 0.00000000001, 0.00000000005]}

    #cv_saga = GridSearchCV(estimator = saga, param_grid = saga_params, cv = 8)
    #cv_svrg = GridSearchCV(estimator = svrg, param_grid = svrg_params, cv = 8)
    cv_sgld = GridSearchCV(estimator = sgld, param_grid = sgld_params, cv = 8)

    #cv_saga.fit(X_tv, y_tv)
    #cv_svrg.fit(X_tv, y_tv)
    cv_sgld.fit(X_tv, y_tv)

    print('saga params: ', cv_saga.best_params_)
    print('svrg params: ', cv_svrg.best_params_)
    print('sgld params: ', cv_svrg.best_params_)

    saga_plot = cv_saga.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)
    svrg_plot = cv_svrg.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)
    sgld_plot = cv_sgld.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)

    plt.ylabel('Test MSE')
    plt.xlabel('Number of passes through data')

    times = [10.0 / size_tv * i for i in range(len(saga_plot))]
    plt.semilogy(times, saga_plot, 'r-', label = 'SAGA')
    plt.semilogy(times, svrg_plot, 'b-', label = 'SVRG')
    plt.semilogy(times, sgld_plot, 'g-', label = 'SGLD')
    plt.show()
    plt.savefig('mse.png')
