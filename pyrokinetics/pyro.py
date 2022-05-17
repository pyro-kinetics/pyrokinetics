from path import Path
from .gk_code import GKCode, gk_codes, GKOutput
from .local_geometry import LocalGeometry, local_geometries
from .equilibrium import Equilibrium
from .kinetics import Kinetics
import warnings
from typing import Optional


class Pyro:
    """
    Basic pyro object able to read, write, run, analyse and plot GK data

    """

    # Define class level info
    supported_gk_codes = [*gk_codes, None]
    supported_local_geometries = [*local_geometries, None]

    def __init__(
        self,
        eq_file: Optional[str] = None,
        eq_type: Optional[str] = None,
        kinetics_file: Optional[str] = None,
        kinetics_type: Optional[str] = None,
        gk_file: Optional[str] = None,
        gk_type: Optional[str] = None,
        gk_code: Optional[str] = None,
        local_geometry: Optional[str] = None,
        linear: bool = True,
        local: bool = True,
    ):

        self._float_format = ""

        self.gk_file = gk_file
        if gk_type is not None and gk_code is None:
            warnings.warn(
                "gk_type is no longer used, please use gk_code instead",
                DeprecationWarning,
            )
            gk_code = gk_type

        self.gk_code = gk_code
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
            self.read_gk_file(self.gk_file, gk_code)

        self.base_directory = Path(__file__).dirname()

    @property
    def gk_code(self) -> GKCode:
        return self._gk_code

    @gk_code.setter
    def gk_code(self, value: Optional[str]):
        """
        Given the gk type as a string, sets the corresponding GKCode class.
        """
        if value is None:
            self._gk_code = None
            return
        try:
            self._gk_code = gk_codes[value]
        except KeyError:
            raise NotImplementedError(f"GKCode {value} not yet supported")

    @property
    def local_geometry(self):
        return self._local_geometry

    @local_geometry.setter
    def local_geometry(self, value):
        """
        Sets the local geometry type
        """
        if isinstance(value, LocalGeometry):
            self._local_geometry = value
        elif value in self.supported_local_geometries:
            self.local_geometry_type = value
            if value is None:
                self._local_geometry = LocalGeometry()
            else:
                self._local_geometry = local_geometries[value]
        else:
            raise NotImplementedError(f"LocalGeometry {value} not yet supported")

    def load_global_eq(self, eq_file=None, eq_type=None, **kwargs):
        """
        Loads in global equilibrium parameters

        """

        if eq_file is not None:
            # If given eq_file, overwrite stored filename
            self.eq_file = eq_file
            # set self.eq_type to None, as the new file may not share a type with
            # the old self.eq_file.
            # If eq_type is None, file type inferrence should be able to figure
            # out the new kinetics_type.
            # If eq_type is not none, it will be set in the next step
            self.eq_type = None

        if eq_type is not None:
            self.eq_type = eq_type

        if self.eq_file is None:
            raise ValueError("Please specify eq_file")

        self.eq = Equilibrium(self.eq_file, self.eq_type, **kwargs)

    def load_global_kinetics(self, kinetics_file=None, kinetics_type=None, **kwargs):
        """
        Loads in global kinetic profiles.
        If provided with kinetics_file or kinetics_type, these will overwrite their
        respective object attributes.

        """

        if kinetics_file is not None:
            # If given kinetics_file, overwrite stored filename
            self.kinetics_file = kinetics_file
            # set self.kinetics_type to None, as the new file may not share a type with
            # the old self.kinetics_file.
            # If kinetics_type is None, file type inferrence should be able to figure
            # out the new kinetics_type.
            # If kinetics_type is not none, it will be set in the next step
            self.kinetics_type = None

        if kinetics_type is not None:
            self.kinetics_type = kinetics_type

        if self.kinetics_file is None:
            raise ValueError("Please specify kinetics_file")

        self.kinetics = Kinetics(self.kinetics_file, self.kinetics_type, **kwargs)

    def read_gk_file(self, gk_file=None, gk_code=None):
        """
        Read GK file

        if self has Equilibrium object then it will
        read the gk_file but not load local_geometry
        """

        if gk_file is not None:
            self.gk_file = gk_file

        if self.gk_file is None:
            raise ValueError("Please specify gk_file")

        # if gk_code is not given, try inferring from possible GKCodes
        if gk_code is None:
            for key, gk in gk_codes.items():
                try:
                    gk.verify(self.gk_file)
                    self.gk_code = key
                    break
                except Exception:
                    continue
        else:
            self.gk_code = gk_code

        # if self.gk_code is still None, we couldn't infer a file type
        if self.gk_code is None:
            raise ValueError("Could not determine gk_code from file type")

        # If equilibrium already loaded then it won't load the input file
        template = hasattr(self, "eq")

        # Better way to select code?
        self.gk_code.read(self, self.gk_file, template)

        # Load in local geometry, local species, and numerics data
        # TODO uncomment when GKInput is implemented
        # self.local_geometry = self.gk_code.get_local_geometry()
        # self.local_species = self.gk_code.get_local_species()
        # self.numerics = self.gk_code.get_numerics(self.local_geometry)

    def write_gk_file(self, file_name, template_file=None, directory=".", gk_code=None):
        """
        Writes single GK input file to file_name


        """

        self.file_name = file_name
        self.run_directory = directory

        # Store a copy of the current gk_code and then override if the
        # user has specified this.
        original_gk_code = self.gk_code.code_name

        if gk_code is not None:
            self.gk_code = gk_code

        code_input = self.gk_code.code_name.lower() + "_input"

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
