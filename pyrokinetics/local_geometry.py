from cleverdict import CleverDict
from .decorators import not_implemented


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

    @not_implemented
    def load_from_eq(self, eq, psi_n=None):
        """ "
        Loads LocalGeometry object from an Equilibrium Object

        """
        pass

    @not_implemented
    def load_from_gk_file(self, pyro, gk_code=None):
        """
        Loads Local geometry object from gk input file
        """

        pass

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a LocalGeometry object

        Returns
        -------
        Copy of LocalGeometry object
        """

        new_localgeometry = LocalGeometry()

        for key, value in self.items():
            setattr(new_localgeometry, key, value)

        return new_localgeometry
