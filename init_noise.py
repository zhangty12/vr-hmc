import numpy

from init import initiator

class init_noise(initiator):
    def __init__(self):
        pass

    def data_params(self):
        data = []
        ans = []
        for line in open('noise/noise.dat', 'r').readlines():
            datum_str = line.split(' ')
            datum = [float(f) for f in datum_str]
            data.append(datum[:4])
            ans.append(datum[5])

        X = numpy.array(data)
        y = numpy.array(ans)
        X_normed = X / X.max(axis = 0)

        saga_params = {'step_size': [0.005, 0.01, 0.05], \
                        'temp': [3, 5, 7]}
        svrg_params = {'step_size': [0.005, 0.01, 0.05], \
                        'temp': [3, 5, 7]}
        sgld_params = {'step_size': [0.00005, 0.0001, 0.0005]}
        sald_params = {'step_size': [0.00005, 0.0001, 0.0005]}

        return 'noise', X_normed, y, saga_params, svrg_params, \
                        sgld_params, sald_params
