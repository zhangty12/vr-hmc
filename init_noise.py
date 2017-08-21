import numpy

from init import initiator

class init_noise(initiator):
    def __init__(self):
        pass

    def data_params(self):
        data = []
        ans = []
        for line in open('noise/noise.dat', 'r').readlines():
            datum_str = line.split()
            datum = [float(f) for f in datum_str]
            data.append(datum[:5])
            ans.append(datum[5])

        X = numpy.array(data)
        y = numpy.array(ans)
        X_normed = X / X.max(axis = 0)

        saga_params = {'step_size': [0.005, 0.01, 0.5], \
                        'temp': [5, 6, 7, 8, 9, 10, 15, 20]}
        svrg_params = {'step_size': [0.005, 0.01, 0.5], \
                        'temp': [5, 6, 7, 8, 9, 10, 15, 20]}
        sgld_params = {'step_size': [0.0005, 0.001, 0.005]}
        sald_params = {'step_size': [0.0005, 0.001, 0.005]}

        return 'noise', 5, X_normed, y, saga_params, svrg_params, \
                        sgld_params, sald_params
