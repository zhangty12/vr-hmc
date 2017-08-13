import numpy
import math
from random import choices
from error_evaluation import eval_mse

def train_test(d, X_train, y_train, X_test, y_test, rounds):
    mse = []

    b = 10
    n = len(y_train)
    T = n * rounds
    h = 0.1
    D = 1.0
    K = 50

    samples = []
    theta = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
    samples.append(theta)

    moments = []
    p = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
    moments.append(p)

    g = numpy.zeros(d)
    w = numpy.zeros(d)
    for t in range(T):
        theta = samples[t]
        if t % K == 0:
            tmp = numpy.zeros(d)
            for x, y in X_train, y_train:
                tmp = tmp + (numpy.dot(theta, x) - y) * x
            g = - theta + tmp
            w = theta

        I = choices(range(n), b)
        tmp = numpy.zeros(d)
        for i in I:
            tmp = tmp + (numpy.dot(theta, X[i]) - y[i]) * X[i] - (numpy.dot(w, X[i]) - y[i]) * X[i]
        nabla = - theta + float(n) / float(b) * tmp + g

        p_next = (1 - D*h) * moments[t] - h*nabla
                + math.sqrt(2*D*h) * numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
        theta_next = samples[t] + h*p_next
        samples.append(theta_next)
        moments.append(p_next)

        err = eval_mse(d, samples, X_test, y_test)
        mse.append(err)

    return mse
