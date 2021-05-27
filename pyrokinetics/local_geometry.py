from collections import OrderedDict
from .decorators import not_implemented


class LocalGeometry(OrderedDict):
    """
    General geometry Object representing local LocalGeometry fit parameters

    Data stored in a ordered dictionary

    """

    def __init__(self,
                 *args, **kwargs):

        s_args = list(args)
        
        if (args and not isinstance(args[0], OrderedDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())
                    
        super(LocalGeometry, self).__init__(*s_args, **kwargs)

        self.geometry_type = None

    @not_implemented
    def load_from_eq(self,
                     eq,
                     psi_n=None
                     ):
        """"
        Loads LocalGeometry object from an Equilibrium Object

        """
        pass

    @not_implemented
    def load_from_gk_file(self,
                          pyro,
                          gk_code=None
                          ):
        """
        Loads Local geometry object from gk input file
        """

        pass
