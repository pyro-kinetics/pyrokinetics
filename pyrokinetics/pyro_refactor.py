from path import Path
from copy import deepcopy
from warnings import warn
from .typing import PathLike
from .gk_code import gk_input_readers, gk_input_writers, gk_output_readers
from .local_geometry import local_geometries
from .local_species import LocalSpecies
from .equilibrium import Equilibrium
from .kinetics import Kinetics
from typing import Optional


# FIXME: On completition of this upgrade, PyroAlt should be renamed to Pyro.
#       It fulfills the same role, but has been declared as a separate
#       class to facilitate testing during the upgrade.
class PyroAlt:
    """
    Basic pyro object able to read, write, run, analyse and plot GK data
    """

    # Define class level info
    supported_gk_codes = [*gk_input_readers]
    supported_local_geometries = [*local_geometries]
    supported_equilibrium_types = [*Equilibrium.supported_equilibrium_types]
    supported_kinetics_types = [*Kinetics.supported_kinetics_types]

    def __init__(
        self,
        gk_input_file: Optional[PathLike] = None,
        gk_input_type: Optional[str] = None,
        eq_file: Optional[PathLike] = None,
        eq_type: Optional[str] = None,
        kinetics_file: Optional[PathLike] = None,
        kinetics_type: Optional[str] = None,
        psi_n: Optional[float] = None,
        a_minor: Optional[float] = None,
    ):
        """
        Parameters:
        -----------
        gk_input_file (PathLike, optional): Filename for a gyrokinetics input file
            (GS2, GENE, CGYRO). When passed, the attributes 'local_geometry',
            'local_species', and 'numerics' are set.
        gk_input_type (str, optional): Type of gyrokinetics input file. When set, this
            will skip file type inference. Possible values are GS2, CGYRO, GENE. If set
            to None, the file type is inferred automatically.
        eq_file (PathLike, optional): Filename for outputs from an equilibrium code,
            such as GEQDSK or TRANSP. When passed, this will set the 'eq' attribute. If
            the parameter 'psi_n' is set, or psi_n can be inferred from gk_input_file,
            this will be used to create/overwrite local_geometry.
        eq_type (str, optional): Type of equilibrium file. When set, this will skip
            file type inference. Possible values are GEQDSK or TRANSP. If set to None,
            the file type is inferred automatically.
        kinetics_file (PathLike, optional): Filename for outputs from a global kinetics
            code, such as SCENE, JETTO, or TRANSP. When passed, this will set the
            'kinetics' attribute. If the parameter 'psi_n' is set, or psi_n can be
            inferred from the gk_input_file, this will be used to create/overwrite
            local_species.
        kinetics_type (str, optional): Type of kinetics file. When set, this will skip
            file type inference. Possible values are SCENE, JETTO, or TRANSP. If set to
            None, the file type is inferred automatically.
        psi_n (float, optional): The normalised local flux surface, with 0.0 being the
            central toroidal field line and 1.0 being the last closed flux surface.
            If set, local_geometry and/or local_species will be derived at that flux
            surface using info from eq_file and/or kinetics_file.
        a_minor (float, optional): The minor radius of the Tokamak, used when setting
            local_species from kinetics_file. This is only needed if psi_n is set, and
            we are not also setting local_geometry from eq_file.
        """

        # Read gk_file if it exists
        if gk_input_file is not None:
            self.read_gk_input_file(gk_input_file, gk_input_type)

        # Load equilibrium file if it exists
        if eq_file is not None:
            self.read_global_eq(eq_file, eq_type, psi_n=psi_n)

        # Load kinetics file if it exists
        if kinetics_file is not None:
            self.read_global_kinetics(
                kinetics_file, kinetics_type, psi_n=psi_n, a_minor=a_minor
            )

    @property
    def gk_input_file(self):
        try:
            return self._gk_input_file
        except AttributeError:
            return None

    @gk_input_file.setter
    def gk_input_file(self, value):
        if value is None:
            self._gk_input_file = None
            return
        filename = Path(value)
        if not filename.exists():
            raise FileNotFoundError(f"The GK input file {value} does not exist")
        self._gk_input_file = filename

    @property
    def gk_output_file(self):
        try:
            return self._gk_output_file
        except AttributeError:
            return None

    @gk_output_file.setter
    def gk_output_file(self, value):
        if value is None:
            self._gk_output_file = None
            return
        filename = Path(value)
        if not filename.exists():
            raise FileNotFoundError(f"The GK output file {value} does not exist")
        self._gk_output_file = filename

    @property
    def eq_file(self):
        try:
            return self._eq_file
        except AttributeError:
            return None

    @eq_file.setter
    def eq_file(self, value):
        if value is None:
            self._eq_file = None
            return
        filename = Path(value)
        if not filename.exists():
            raise FileNotFoundError(f"The equilibrium file {value} does not exist")
        self._eq_file = filename

    @property
    def kinetics_file(self):
        try:
            return self._kinetics_file
        except AttributeError:
            return None

    @kinetics_file.setter
    def kinetics_file(self, value):
        if value is None:
            self._kinetics_file = None
            return
        filename = Path(value)
        if not filename.exists():
            raise FileNotFoundError(f"The kinetic file {value} does not exist")
        self._kinetics_file = filename

    def read_global_eq(
        self,
        eq_file: PathLike,
        eq_type: Optional[str] = None,
        psi_n: Optional[float] = None,
        local_geometry_type: Optional[str] = "Miller",
        **kwargs,
    ):
        """
        Loads in global equilibrium parameters. Overwrites self.eq_file and
        self.eq_type. If psi_n is given, also overwrites self.local_geometry.
        """
        self.eq_file = eq_file
        self.eq_type = eq_type
        self.eq = Equilibrium(eq_file, eq_type, **kwargs)
        if self.eq_type is None:
            self.eq_type = self.eq.eq_type
        if psi_n is not None:
            self.set_local_geometry_from_global_eq(psi_n, local_geometry_type)

    def read_global_kinetics(
        self,
        kinetics_file: PathLike,
        kinetics_type: Optional[str] = None,
        psi_n: Optional[float] = None,
        a_minor: Optional[float] = None,
        **kwargs,
    ):
        """
        Loads in global kinetic profiles.
        Overwrites self.kinetics_file and self.kinetics_type
        """
        self.kinetics_file = kinetics_file
        self.kinetics_type = kinetics_type
        self.kinetics = Kinetics(kinetics_file, kinetics_type, **kwargs)
        if self.kinetics_type is None:
            self.kinetics_type = self.kinetics.kinetics_type
        if psi_n is not None:
            self.set_local_species_from_global_kinetics(psi_n, a_minor)

    def read_gk_input_file(
        self, gk_input_file: PathLike, gk_input_type: Optional[str] = None, **kwargs
    ):
        """
        Read GK file, set attributes local_geometry, local_species, and numerics
        """
        self.gk_input_file = gk_input_file
        # Load in local geometry, local species, and numerics data
        if gk_input_type is not None:
            reader = gk_input_readers[gk_input_type]
            self.gk_input_type = gk_input_type
        else:
            # Infer equilibrium type from file
            reader = gk_input_readers[gk_input_file]
            self.gk_input_type = reader.file_type
        # read data
        self.gk_input_data = reader(gk_input_file)
        # add user flags
        reader.add_flags(kwargs)
        # get info from input file
        self.local_geometry = reader.get_local_geometry()
        self.local_species = reader.get_local_species()
        self.numerics = reader.get_numerics()

    def write_gk_file(
        self,
        filename: PathLike,
        gk_code_type: str,
        template_file: Optional[PathLike] = None,
        float_format: str = "",
    ):
        """
        Writes GK input file to filename
        """
        write = gk_input_writers(gk_code_type, template_file)
        try:
            self.local_geometry
            self.local_species
            self.numerics
        except AttributeError:
            raise RuntimeError(
                "Pyro object must have the attributes local_geometry, local_species, "
                "and numerics. These may be set by reading a gyrokinetics input file. "
                "local_geometry may be set from global parameters using "
                "read_global_eq(), and local_species may be set from global kinetics "
                "parameters using read_global_kinetics(). Each of these attributes "
                "may also be set manually."
            )
        write(
            filename,
            self.local_geometry,
            self.local_species,
            self.numerics,
            float_format=float_format,
        )

    def set_local_geometry_from_global_eq(
        self, psi_n: float, local_geometry_type: str = "Miller"
    ):
        """
        Loads local geometry parameters from self.eq
        """
        if not hasattr(self, "eq"):
            raise RuntimeError("Pyro object must have a global Equilibrium first")
        LocalGeometryType = local_geometries.get_type(local_geometry_type)
        self.local_geometry = LocalGeometryType.from_global_eq(self.eq, psi_n=psi_n)

    def set_local_species_from_global_kinetics(
        self, psi_n: float, a_minor: Optional[float] = None
    ):
        """
        Loads local species parameters from self.kinetics.
        If the parameter 'a_minor' is set to None, it is inferred from self.eq
        """
        # Ensure we have a Kinetics before proceeding
        if not hasattr(self, "kinetics"):
            raise RuntimeError(
                "Pyro object must have a global Kinetics first. Set this using "
                "read_global_kinetcs, or pass a kinetics_file during construction"
            )

        # Get a_minor from eq, unless one is provided as an optional argument
        if hasattr(self, "eq"):
            if a_minor is None:
                a_minor = self.eq.a_minor
            else:
                warn(
                    "Pyro object already has a_minor derived from global equilibrium. "
                    f"Overwriting with {a_minor}",
                    RuntimeWarning,
                )

        # Raise an error if a_minor could not be determined
        if a_minor is None:
            raise RuntimeError(
                "Either provide minor radius 'a_minor', or Pyro object must have a "
                "global Equilibrium first"
            )

        self.local_species = LocalSpecies.from_global_kinetics(
            self.kinetics, psi_n=psi_n, lref=a_minor
        )

    def set_local_from_global(self, psi_n: float, local_geometry_type: str = "Miller"):
        """
        Loads local_geometry and local_species from self.eq and self.kinetics
        """
        self.set_local_geometry_from_global_eq(
            psi_n, local_geometry_type=local_geometry_type
        )
        self.set_local_species_from_global_kinetics(psi_n)

    def read_gk_output(
        self, gk_output_file: PathLike, gk_output_type: Optional[str] = None
    ):
        """
        Loads GKOutput object
        """
        self.gk_output_file = gk_output_file
        if gk_output_type is not None:
            reader = gk_output_readers[gk_output_type]
            self.gk_output_type = gk_output_type
        else:
            # Infer gk type from file
            reader = gk_output_readers[gk_output_file]
            self.gk_output_type = reader.file_type
        self.gk_output_data = reader(gk_output_file)

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Pyro object

        Returns
        -------
        Copy of pyro object
        """
        new_pyro = type(self)()  # Create new class with default args
        for key, value in self.__dict__.items():
            setattr(new_pyro, key, deepcopy(value))
        return new_pyro
