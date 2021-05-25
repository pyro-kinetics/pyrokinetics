import numpy as np
from collections import OrderedDict

class Numerics(OrderedDict):
    """
    Set up numerical grid

    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if (args and not isinstance(args[0], OrderedDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())

        if args:
            super(Numerics, self).__init__(*s_args, **kwargs)
        else:
            self.default()

    def default(self):

        self['ntheta'] = 32
        self['theta0'] = 0.0
        self['nenergy'] = 8
        self['npitch'] = 8

        self['nky'] = 1
        self['nkx'] = 1
        self['kx'] = 0.0
        self['ky'] = 0.1

        self['nperiod'] = 1

        self['nonlinear'] = False
