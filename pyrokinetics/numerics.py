from cleverdict import CleverDict
from copy import deepcopy


class Numerics(CleverDict):
    """
    Set up numerical grid

    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if args and not isinstance(args[0], CleverDict) and isinstance(args[0], dict):
            s_args[0] = sorted(args[0].items())

        if args:
            super().__init__(*s_args, **kwargs)
        else:
            self.default()

    def default(self):

        data_dict = {
            "ntheta": 32,
            "theta0": 0.0,
            "nenergy": 8,
            "npitch": 8,
            "nky": 1,
            "nkx": 1,
            "kx": 0.0,
            "ky": 0.1,
            "nperiod": 1,
            "nonlinear": False,
            "phi": True,
            "apar": False,
            "bpar": False,
            "max_time": 500.0,
            "delta_time": 0.001,
        }

        super().__init__(data_dict)

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Numerics object

        Returns
        -------
        Copy of Numerics object
        """

        new_numerics = Numerics()

        for key, value in self.items():
            setattr(new_numerics, key, deepcopy(value, memodict))

        return new_numerics
