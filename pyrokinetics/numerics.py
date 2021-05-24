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

        super(Numerics, self).__init__(*s_args, **kwargs)
        
def default():

    num = Numerics()

    num['ntheta'] = 32
    num['nenergy'] = 8
    num['nlambda'] = 8
    num['nxi'] = 16
    num['nky'] = 1
    num['nkx'] = 1
    
    num['emax'] = 8.0
    num['nperiod'] = 2
    num['box_size'] = 1

    return num
