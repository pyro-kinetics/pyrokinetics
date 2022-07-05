from cleverdict import CleverDict
from copy import deepcopy
from ..decorators import not_implemented
from ..factory import Factory


class LocalGeometry(CleverDict):
    """
    General geometry Object representing local LocalGeometry fit parameters

    Data stored in a ordered dictionary

    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if args and not isinstance(args[0], CleverDict) and isinstance(args[0], dict):
            s_args[0] = sorted(args[0].items())

            super(LocalGeometry, self).__init__(*s_args, **kwargs)

        elif len(args) == 0:
            _data_dict = {"local_geometry": None}
            super(LocalGeometry, self).__init__(_data_dict)

    # TODO replace this with an abstract classmethod
    @not_implemented
    def load_from_eq(self, eq, psi_n=None):
        """ "
        Loads LocalGeometry object from an Equilibrium Object

        """
        pass

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a LocalGeometry object

        Returns
        -------
        Copy of LocalGeometry object
        """
        # Create new empty object. Works for derived classes too.
        new_localgeometry = self.__class__()
        for key, value in self.items():
            new_localgeometry[key] = deepcopy(value, memodict)
        return new_localgeometry


# Create global factory for LocalGeometry objects
local_geometries = Factory(LocalGeometry)
