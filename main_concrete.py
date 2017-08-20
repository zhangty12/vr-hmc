from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

import numpy
import matplotlib.pyplot as plt

import read_concrete
import vr_saga_reg
import vr_svrg_reg
import vr_sgld_reg
import vr_sald_reg

dim, X, y = read_concrete.read_concrete()

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size = 1./5)

rnd = 3; size_tv = len(y_tv)

saga = vr_saga_reg.saga_estimator(dim = dim, round = rnd)
svrg = vr_svrg_reg.svrg_estimator(dim = dim, round = rnd)
sgld = vr_sgld_reg.sgld_estimator(dim = dim, round = rnd)
sald = vr_sald_reg.sald_estimator(dim = dim, round = rnd)

saga_params = {'step_size': [0.001, 0.01, 0.1], \
                'temp': [0.1, 1, 5]}
svrg_params = {'step_size': [0.001, 0.01, 0.1], \
                'temp': [0.1, 1, 5]}
sgld_params = {'step_size': [0.000001, 0.00001, 0.0001]}
sald_params = {'step_size': [0.000001, 0.00001, 0.0001]}

cv_saga = GridSearchCV(estimator = saga, param_grid = saga_params, cv = 8)
cv_svrg = GridSearchCV(estimator = svrg, param_grid = svrg_params, cv = 8)
cv_sgld = GridSearchCV(estimator = sgld, param_grid = sgld_params, cv = 8)
cv_sald = GridSearchCV(estimator = sald, param_grid = sald_params, cv = 8)

cv_saga.fit(X_tv, y_tv)
cv_svrg.fit(X_tv, y_tv)
cv_sgld.fit(X_tv, y_tv)
cv_sald.fit(X_tv, y_tv)

print('saga params: ', cv_saga.best_params_)
print('svrg params: ', cv_svrg.best_params_)
print('sgld params: ', cv_sgld.best_params_)
print('sald params: ', cv_sald.best_params_)

saga_plot = cv_saga.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)
svrg_plot = cv_svrg.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)
sgld_plot = cv_sgld.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)
sald_plot = cv_sald.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test)

plt.ylabel('Test MSE')
plt.xlabel('Number of passes through data')

times = [10.0 / size_tv * i for i in range(len(sgld_plot))]

plt.semilogy(times, saga_plot, 'r-', label = 'SAGA')
plt.semilogy(times, svrg_plot, 'b-', label = 'SVRG')
plt.semilogy(times, sgld_plot, 'g-', label = 'SGLD')
plt.semilogy(times, sald_plot, 'y-', label = 'SALD')

plt.legend()
plt.savefig('mse.png')

plt.show()

