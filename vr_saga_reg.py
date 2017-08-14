import numpy
import math
from random import choice
from error_evaluation import eval_mse

def train_test(d, X_train, y_train, X_test, y_test, rounds):
    mse = []

    b = 10
    n = len(y_train)
    T = n * rounds
    h = 0.1
    D = 1.0

    samples = []
    theta = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
    samples.append(theta)

    moments = []
    p = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
    moments.append(p)

    alpha = []
    for i in range(n):
        alpha.append(theta)

    g = numpy.zeros(d)
    for i in range(n):
        g = g - (y_train[i] - numpy.dot(alpha[i], X_train[i, :])) * X_train[i, :]

    print('Total number of iters: ', T)
    for t in range(T):
        if t % 100 is 0:
            print('Iter ', t)

        theta = samples[t]
        p = moments[t]

        I = []
        for i in range(b):
            I.append(choice(range(n)))
        tmp = numpy.zeros(d)
        for i in I:
            tmp = tmp + (numpy.dot(theta, X_train[i, :]) - y_train[i]) * X_train[i, :] \
                    - (numpy.dot(alpha[i], X_train[i, :]) - y_train[i]) * X_train[i, :]
        nabla = - theta + float(n) / float(b) * tmp + g

        p_next = (1 - D*h) * p - h * nabla + math.sqrt(2*D*h) \
                    * numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
        theta_next = theta + h * p_next
        samples.append(theta_next)
        moments.append(p_next)

        for i in I:
            alpha[i] = theta
        g = g + tmp

        err = eval_mse(d, samples, X_test, y_test)
        mse.append(err)
    return mse
