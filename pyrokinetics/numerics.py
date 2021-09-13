import numpy as np
from cleverdict import CleverDict


class Numerics(CleverDict):
    """
    Set up numerical grid

    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if (args and not isinstance(args[0], CleverDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())

        if args:
            super(Numerics, self).__init__(*s_args, **kwargs)
        else:
            self.default()

    def default(self):

        _data_dict = {'ntheta': 32, 'theta0': 0.0, 'nenergy': 8, 'npitch': 8, 'nky': 1, 'nkx': 1, 'kx': 0.0, 'ky': 0.1,
                      'nperiod': 1, 'nonlinear': False, 'phi': True, 'apar': False, 'bpar': False, 'max_time': 500.0,
                      'delta_time': 0.001}

        super(Numerics, self).__init__(_data_dict)
