from .decorators import not_implemented

class GKCode:
    """
    Basic GK code object
    """

    def __init__(self):
        pass

    @not_implemented
    def read(self, pyro, **kwargs):
        """
        Reads in GK input file into Pyro object
        as a dictionary
        """
        pass

    @not_implemented
    def load_pyro(self, pyro, **kwargs):
        """
        Loads GK dictionary into Pyro object
        """
        pass

    @not_implemented
    def write(self, pyro, **kwargs):
        """
        For a given pyro object write a GK code input file

        """
        pass

    @not_implemented
    def load_miller(self, pyro, code, **kwargs):
        """
        Load Miller object from a GK code input file
        """
        pass

    @not_implemented
    def load_local_species(self, pyro, code):
        """
        Load local species object from a GK code input file
        """
        pass

    @not_implemented
    def add_flags(self, pyro, flags):
        """
        Add extra flags to a GK code input file

        """
        pass

    @not_implemented
    def load_numerics(self, pyro, code):
        """
        Load Numerics object from a GK code input file
        """
        pass

    @not_implemented
    def pyro_to_code_miller(self):
        """
        Generates dictionary of equivalent pyro and gk code parameter names
        for miller parameters
        """
        pass

    @not_implemented
    def pyro_to_code_species(self):
        """
        Generates dictionary of equivalent pyro and gk code parameter names
        for miller parameters
        """
        pass
