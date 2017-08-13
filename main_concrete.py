import numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import read_concrete
import vr_saga_reg
import vr_svrg_reg

dim, X, y = read_concrete.read_concrete()

kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rnd = 5; size = len(y_train)
    saga_plot = vr_saga_reg.train_test(dim, X_train, y_train, X_test, y_test, rnd)
    svrg_plot = vr_svrg_reg.train_test(dim, X_train, y_train, X_test, y_test, rnd)

    plt.ylabel('Test MSE')
    plt.xlabel('Number of passes through data')

    times = [1.0 / size * i for i in range(rnd * size)]
    plt.plot(times, saga_plot, 'r-', 'SAGA',
             times, svrg_plot, 'b-', 'SVRG')
