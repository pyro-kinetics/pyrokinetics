from path import Path
from .gk_code import GKCode
from .local_geometry import LocalGeometry
from .equilibrium import Equilibrium
from .kinetics import Kinetics
from .gk_output import GKOutput
from .miller import Miller


class Pyro:
    """
    Basic pyro object able to read, write, run, analyse and plot GK data

    """

    def __init__(
        self,
        eq_file=None,
        eq_type=None,
        kinetics_file=None,
        kinetics_type=None,
        gk_file=None,
        gk_type=None,
        local_geometry=None,
        linear=True,
        local=True,
    ):

        self._float_format = ""
        self.supported_gk_codes = ["GS2", "CGYRO", "GENE", None]
        self.supported_local_geometries = ["Miller", None]

        self.gk_file = gk_file
        self.gk_type = gk_type
        self.gk_code = self.gk_type
        self.gk_output = GKOutput()

        if self.gk_file is not None:
            self.file_name = Path(self.gk_file).basename()
            self.run_directory = Path(self.gk_file).dirname()

        self.local_geometry_type = local_geometry
        self.local_geometry = self.local_geometry_type
        self.linear = linear
        self.local = local
        self.eq_file = eq_file
        self.eq_type = eq_type
        self.kinetics_file = kinetics_file
        self.kinetics_type = kinetics_type

        # Load equilibrium file if it exists
        if self.eq_file is not None:
            self.load_global_eq()

        # Load kinetics file if it exists
        if self.kinetics_file is not None:
            self.load_global_kinetics()

        # Read gk_file if it exists (not necessarily load it)
        if self.gk_file is not None:
            self.read_gk_file()

        self.base_directory = Path(__file__).dirname()

    @property
    def gk_code(self):
        return self._gk_code

    @gk_code.setter
    def gk_code(self, value):
        """
        Sets the GK code to be used

        """

        if value in self.supported_gk_codes:

            self.gk_type = value

            if self.gk_type == "GS2":
                from .gs2 import GS2

                self._gk_code = GS2()

            elif self.gk_type == "CGYRO":
                from .cgyro import CGYRO

                self._gk_code = CGYRO()

            elif self.gk_type == "GENE":
                from .gene import GENE

                self._gk_code = GENE()

            elif value is None:
                self._gk_code = GKCode()

        else:
            raise NotImplementedError(f"GKCode {value} not yet supported")

    @property
    def local_geometry(self):
        return self._local_geometry

    @local_geometry.setter
    def local_geometry(self, value):
        """
        Sets the local geometry type
        """

        if value in self.supported_local_geometries:

            self.local_geometry_type = value

            if self.local_geometry_type == "Miller":
                self._local_geometry = Miller()

            elif value is None:
                self._local_geometry = LocalGeometry()

        else:
            raise NotImplementedError(f"LocalGeometry {value} not yet supported")

    def load_global_eq(self, eq_file=None, eq_type=None, **kwargs):
        """
        Loads in global equilibrium parameters

        """

        if eq_file is not None:
            self.eq_file = eq_file

        if eq_type is not None:
            self.eq_type = eq_type

        if self.eq_type is None or self.eq_file is None:
            raise ValueError("Please specify eq_type and eq_file")
        else:
            self.eq = Equilibrium(self.eq_file, self.eq_type, **kwargs)

    def load_global_kinetics(self, kinetics_file=None, kinetics_type=None):
        """
        Loads in global kinetic profiles

        """

        if kinetics_file is not None:
            self.kinetics_file = kinetics_file

        if kinetics_type is not None:
            self.kinetics_type = kinetics_type

        if self.kinetics_type is None or self.kinetics_file is None:
            raise ValueError("Please specify kinetics_type and kinetics_file")
        else:
            self.kinetics = Kinetics(self.kinetics_file, self.kinetics_type)

    def read_gk_file(self, gk_file=None, gk_type=None):
        """
        Read GK file

        if self has Equilibrium object then it will
        not load the equilibrium parameters into

        """

        if gk_file is not None:
            self.gk_file = gk_file

        if gk_type is not None:
            self.gk_type = gk_type

        if self.gk_type is None or self.gk_file is None:
            raise ValueError("Please specify gk_type and gk_file")
        else:

            # If equilibrium already loaded then it won't load the input file
            if hasattr(self, "eq"):
                template = True
            else:
                template = False

            # Better way to select code?
            self.gk_code.read(self, self.gk_file, template)

    def write_gk_file(self, file_name, template_file=None, directory=".", gk_code=None):
        """
        Writes single GK input file to file_name


        """

        self.file_name = file_name
        self.run_directory = directory

        # Store a copy of the current gk_code and then override if the
        # user has specified this.
        original_gk_code = self.gk_type
        if gk_code is not None:
            self.gk_code = gk_code

        code_input = self.gk_type.lower() + "_input"

        # Check if code has been read in before
        if not hasattr(self, code_input):
            if template_file is not None:
                self.gk_code.read(self, data_file=template_file, template=True)
            else:
                # Reads in default template
                self.gk_code.read(self, template=True)

        self.gk_code.write(self, file_name, directory=self.run_directory)

        # Ensure that gk_code is unchanged on exit
        self.gk_code = original_gk_code

    def add_flags(
        self,
        flags,
    ):
        """
        Adds flags to GK file

        """

        self.gk_code.add_flags(self, flags)

    def load_local_geometry(self, psi_n=None, **kwargs):
        """
        Loads local geometry parameters

        """

        if psi_n is None:
            raise ValueError("Need a psi_n to load local geometry")

        if self.eq is None:
            raise ValueError("Please load equilibrium first")

        if self.local_geometry_type is None:
            raise ValueError("Please specify local geometry type")

        # Load local geometry
        self.local_geometry.load_from_eq(self.eq, psi_n=psi_n, **kwargs)

    def load_local(
        self,
        psi_n=None,
        local_geometry=None,
    ):
        """
        Loads local geometry and kinetic parameters

        Adds specific geometry and speciesLocal attribute to Pyro
        """

        if psi_n is None:
            raise ValueError("Need a psi_n to load local parameters")

        if self.eq is None:
            raise ValueError("Please load equilibrium first")

        if local_geometry is not None:
            self.local_geometry = local_geometry

        self.load_local_geometry(psi_n=psi_n)

        # Load species data
        self.load_local_species(psi_n=psi_n)

    def load_local_species(
        self,
        psi_n=None,
    ):
        """
        Loads local species parameters

        Adds load_local_species attribute to Pyro
        """

        from .local_species import LocalSpecies

        if psi_n is None:
            raise ValueError("Need a psi_n to load")

        local_species = LocalSpecies()

        local_species.from_kinetics(self.kinetics, psi_n=psi_n, lref=self.eq.a_minor)

        self.local_species = local_species

    def load_gk_output(self, **kwargs):
        """
        Loads GKOutput object
        """

        self.gk_code.load_gk_output(self, **kwargs)

    @property
    def float_format(self):
        """Sets float format for input files"""

        return self._float_format

    @float_format.setter
    def float_format(self, value):
        """Setter for float_format"""

        self._float_format = value

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Pyro object

        Returns
        -------
        Copy of pyro object
        """
        from copy import deepcopy

        new_pyro = Pyro()

        for key, value in self.__dict__.items():
            setattr(new_pyro, key, deepcopy(value))

        return new_pyro
