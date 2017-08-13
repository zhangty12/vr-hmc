import numpy

def eval_mse(d, samples, X_test, y_test):
    err = 0.0

    m = len(samples)
    for x, y in X_test, y_test:
        pred = 0.0
        for theta in samples:
            pred = pred + numpy.dot(x, theta)
        pred = pred / float(m)

        err = err + (y - pred) ** 2

    return err / float(len(y_test))
