import numpy

def eval_mse(d, samples, X_test, y_test):
    err = 0.0

    m = len(samples)
    n = len(y_test)

    for i in range(n):
        x = X_test[i, :]
        y = y_test[i]
        pred = 0.0
        for theta in samples:
            pred = pred + numpy.dot(x, theta)
        pred = pred / float(m)

        err = err + (y - pred) ** 2

    return err / float(n)
