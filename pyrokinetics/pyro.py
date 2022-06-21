import warnings
from path import Path
from typing import Optional
from copy import deepcopy
import re

from .gk_code import gk_inputs, gk_output_readers
from .local_geometry import LocalGeometry, LocalGeometryMiller, local_geometries
from .local_species import LocalSpecies
from .equilibrium import Equilibrium
from .kinetics import Kinetics
from .typing import PathLike
from .templates import gk_templates


class Pyro:
    """
    Basic pyro object able to read, write, run, analyse and plot GK data

    """

    # Define class level info
    supported_gk_inputs = [*gk_inputs]
    supported_gk_output_readers = [*gk_output_readers]
    supported_local_geometries = [*local_geometries]
    supported_equilibrium_types = [*Equilibrium.supported_equilibrium_types]
    supported_kinetics_types = [*Kinetics.supported_kinetics_types]

    def __init__(
        self,
        eq_file: Optional[PathLike] = None,
        eq_type: Optional[str] = None,
        kinetics_file: Optional[PathLike] = None,
        kinetics_type: Optional[str] = None,
        gk_file: Optional[PathLike] = None,
        gk_input_file: Optional[PathLike] = None,  # synonym for gk_file
        gk_output_file: Optional[PathLike] = None,
        gk_code: Optional[str] = None,
        gk_type: Optional[str] = None,  # deprecated, synonym for gk_code
    ):
        """
        Parameters:
        -----------
        eq_file (PathLike, optional): Filename for outputs from a global equilibrium
            code, such as GEQDSK or TRANSP. When passed, this will set the 'eq'
            attribute. This can be used to set a local geometry using the function
            load_local_geometry or load_local.
            the parameter 'psi_n' is set, or psi_n can be inferred from gk_input_file,
            this will be used to create/overwrite local_geometry.
        eq_type (str, optional): Type of equilibrium file. When set, this will skip
            file type inference. Possible values are GEQDSK or TRANSP. If set to None,
            the file type is inferred automatically.
        kinetics_file (PathLike, optional): Filename for outputs from a global kinetics
            code, such as SCENE, JETTO, or TRANSP. When passed, this will set the
            'kinetics' attribute. This can be used to set local kinetics using the
            function load_local_kinetics or load_local.
        kinetics_type (str, optional): Type of kinetics file. When set, this will skip
            file type inference. Possible values are SCENE, JETTO, or TRANSP. If set to
            None, the file type is inferred automatically.
        gk_file (PathLike, optional): Filename for a gyrokinetics input file
            (GS2, GENE, CGYRO). When passed, the attributes 'local_geometry',
            'local_species', and 'numerics' are set.
        gk_input_file (PathLike, optional): Synonym for gk_file. gk_file takes
            precedence.
        gk_output_file (PathLike, optional): Filename or directory name for gyrokinetics
            output file(s). For GS2, the user should pass the '.out.nc' NetCDF4
            file. For CGYRO, the user should pass the directory containing output files
            with their standard names. For GENE, the user should pass one of
            'parameters_####', 'nrg_####' or 'field_####', where #### is 4 digits
            identifying the run. If one of these is passed, the others will be found
            in the same directory. Alternatively, the user should pass the directory
            containing 'parameters_0000', 'field_0000' and 'nrg_0000'
        gk_code (str, optional): Type of gyrokinetics input file. When set, this
            will skip file type inference. Possible values are 'GS2', 'CGYRO', or
            'GENE'. If set to None, the file type is inferred automatically.
            Sets the gyrokinetics type for both input files and output files.
        gk_type (str, optional): DEPRECATED, synonym for gk_code. gk_code takes
            precedence.
        """

        self.float_format = ""

        # Load equilibrium file if it exists
        if eq_file is not None:
            self.load_global_eq(eq_file, eq_type)

        # Load kinetics file if it exists
        if kinetics_file is not None:
            self.load_global_kinetics(kinetics_file, kinetics_type)

        # Prepare to read gk_file
        if gk_type is not None and gk_code is None:
            warnings.warn(
                "gk_type is no longer used, please use gk_code instead",
                DeprecationWarning,
            )
            gk_code = gk_type

        if gk_input_file is not None and gk_file is None:
            gk_file = gk_input_file

        # If provided gk_code but not gk_file, read default file
        if gk_code is not None and gk_file is None:
            gk_file = gk_templates[gk_code]

        # Read gk_file if it exists
        if gk_file is not None:
            self.read_gk_file(gk_file, gk_code)

        # Load gk_output if it exists
        self.gk_output_file = (
            Path(gk_output_file) if gk_output_file is not None else None
        )
        if self.gk_output_file is not None:
            self.read_gk_output_file(self.gk_output_file, gk_code)

        self.base_directory = Path(__file__).dirname()

    # Set properties for file attributes
    # files may be either None, or a Path

    @property
    def eq_file(self) -> Path:
        try:
            return self._eq_file
        except AttributeError:
            return None

    @eq_file.setter
    def eq_file(self, value: PathLike) -> None:
        self._eq_file = Path(value) if value is not None else None

    @property
    def kinetics_file(self):
        try:
            return self._kinetics_file
        except AttributeError:
            return None

    @kinetics_file.setter
    def kinetics_file(self, value: PathLike) -> None:
        self._kinetics_file = Path(value) if value is not None else None

    @property
    def gk_file(self):
        try:
            return self._gk_file
        except AttributeError:
            return None

    @gk_file.setter
    def gk_file(self, value: PathLike) -> None:
        self._gk_file = Path(value) if value is not None else None

    @property
    def gk_output_file(self):
        try:
            return self._gk_output_file
        except AttributeError:
            return None

    @gk_output_file.setter
    def gk_output_file(self, value: PathLike) -> None:
        self._gk_output_file = Path(value) if value is not None else None

    # Define local_geometry property
    # By providing string like 'Miller', sets self.local_geometry to LocalGeometryMiller

    @property
    def local_geometry(self) -> LocalGeometry:
        return self._local_geometry

    @local_geometry.setter
    def local_geometry(self, value) -> None:
        """
        Sets the local geometry type
        """
        # FIXME This should perhaps be reconsidered, as it can result in the creation of
        # uninitialised instances and cause unexpected behaviour. May be preferable to
        # implement a 'convert_local_geometry' function once other LocalGeometry types
        # are implemented, and to disallow converting LocalGeometry types by assigning
        # strings to the local_geometry attribute. Currently, this behaviour is only
        # used within load_local_geometry, where an uninitialised LocalGeometry is
        # created and then populated using load_from_eq. We can do away with this by
        # implementing a 'from_eq' classmethod within LocalGeometry types, to be
        # used as an alternative to the standard constructor.
        if isinstance(value, LocalGeometry):
            self._local_geometry = value
        elif value in self.supported_local_geometries:
            self._local_geometry = local_geometries[value]
        elif value is None:
            self._local_geometry = None
        else:
            raise NotImplementedError(f"LocalGeometry {value} not yet supported")

    @property
    def local_geometry_type(self) -> str:
        try:
            if isinstance(self._local_geometry, LocalGeometryMiller):
                return "Miller"
            elif self._local_geometry is None:
                return None
            else:
                raise RuntimeError(
                    "Pyro._local_geometry is set to an unknown geometry type"
                )
        except AttributeError:
            return None

    # Functions for reading equilibrium and kinetics files

    def load_global_eq(
        self, eq_file: PathLike, eq_type: Optional[str] = None, **kwargs
    ) -> None:
        """
        Loads in global equilibrium parameters
        """
        self.eq_file = eq_file  # property setter, converts to Path
        self.eq = Equilibrium(self.eq_file, eq_type, **kwargs)

    @property
    def eq_type(self) -> str:
        try:
            return self.eq.eq_type
        except AttributeError:
            return None

    def load_global_kinetics(
        self, kinetics_file: PathLike, kinetics_type: Optional[str] = None, **kwargs
    ) -> None:
        """
        Loads in global kinetic profiles.
        If provided with kinetics_file or kinetics_type, these will overwrite their
        respective object attributes.
        """
        self.kinetics_file = kinetics_file  # property setter, converts to Path
        self.kinetics = Kinetics(self.kinetics_file, kinetics_type, **kwargs)

    @property
    def kinetics_type(self) -> str:
        try:
            return self.kinetics.kinetics_type
        except AttributeError:
            return None

    # Functions for setting local_geometry and local_species from global Equilibrium
    # and Kinetics

    def load_local_geometry(
        self, psi_n: float, local_geometry: str = "Miller", **kwargs
    ) -> None:
        """
        Loads local geometry parameters

        """
        try:
            if self.eq is None:
                raise AttributeError
        except AttributeError:
            raise ValueError("Please load equilibrium first")

        self.local_geometry = local_geometry  # uses property setter

        # Load local geometry
        self.local_geometry.load_from_eq(self.eq, psi_n=psi_n, **kwargs)

    def load_local_species(self, psi_n: float, a_minor: Optional[float] = None) -> None:
        """
        Loads local species parameters

        Adds load_local_species attribute to Pyro
        """
        try:
            if self.kinetics is None:
                raise AttributeError
        except AttributeError:
            raise RuntimeError(
                "Pyro.load_local_species: Must have read global kinetics first. "
                "Use function load_global_kinetics."
            )

        if a_minor is None:
            try:
                if self.eq is None:
                    raise AttributeError
                a_minor = self.eq.a_minor
            except AttributeError:
                raise RuntimeError(
                    "Pyro.load_local_species: Must set a_minor, or read global "
                    "equilibrium first. To set global_equilibrium, use function "
                    "load_global_equilibrium."
                )

        local_species = LocalSpecies()
        local_species.from_kinetics(self.kinetics, psi_n=psi_n, lref=a_minor)
        self.local_species = local_species

    def load_local(self, psi_n: float, local_geometry: str = "Miller") -> None:
        """
        Loads local geometry and kinetic parameters

        Adds specific geometry and speciesLocal attribute to Pyro
        """
        self.load_local_geometry(psi_n, local_geometry=local_geometry)
        self.load_local_species(psi_n)

    # Functions for handling gyrokinetics inputs/outputs

    def read_gk_file(self, gk_file: PathLike, gk_code: Optional[str] = None):
        """
        Read GK file

        NOTE: In previous versions, if a global Equilibrium was loaded, then this would
        read the gk_file but not load local_geometry. Now, it will overwrite a
        local_geometry created via load_local_geometry, but this can be fixed by calling
        load_local_geometry again with the appropriate psi_n.
        """

        self.gk_file = gk_file  # uses property setter, converts to Path
        self.file_name = self.gk_file.name
        self.run_directory = self.gk_file.parent
        # if gk_code is not given, try inferring from possible GKInputs
        if gk_code is None:
            self.gk_input = gk_inputs[gk_file]
            self.gk_code = self.gk_input.file_type
        else:
            self.gk_input = gk_inputs[gk_code]
            self.gk_code = gk_code

        self.gk_input.read(gk_file)
        self.local_geometry = self.gk_input.get_local_geometry()
        self.local_species = self.gk_input.get_local_species()
        self.numerics = self.gk_input.get_numerics()

    def convert_gk_code(
        self, gk_code: str, template_file: Optional[PathLike] = None
    ) -> None:
        """
        Convert the gk_input attribute to a new type of gyrokinetics code. Uses the
        attributes local_geometry, local_species, and numerics. If these have been
        edited using local_local (or similar), this will be reflected in the new
        gk_input.
        """
        # Ensure gk_code is valid
        if gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro: Cannot convert to gk_code '{gk_code}'")

        # Ensure we have the correct attributes
        try:
            numerics = self.numerics
            local_geometry = self.local_geometry
            local_species = self.local_species
        except AttributeError:
            raise RuntimeError(
                "Pyro: Must have read a gyrokinetics input file before converting. "
                "Ensure the attributes 'numerics', 'local_geometry', and "
                "'local_species' are set."
            )

        # Create new gk_input and set. Don't touch self.gk_input until we're done,
        # in case something goes wrong.
        gk_input = gk_inputs[gk_code]
        gk_input.set(
            local_geometry=local_geometry,
            local_species=local_species,
            numerics=numerics,
            template_file=template_file,
        )

        # Assign to self
        self.gk_input = gk_input
        self.gk_code = gk_code

    def write_gk_file(
        self,
        file_name: PathLike,
        gk_code: Optional[str] = None,
        template_file: Optional[PathLike] = None,
    ) -> None:
        """
        Writes single GK input file to file_name
        """
        # TODO Should be able to write without first reading a file. The user should
        #      be able to provide their own local_geomtry, local_species and numerics.
        try:
            gk_input = self.gk_input
        except AttributeError:
            raise RuntimeError(
                "Pyro.write_gk_file: Must have already read gyrokinetics input file"
            )

        file_name = Path(file_name)
        self.file_name = file_name.name
        self.run_directory = file_name.parent

        if gk_code is None:
            writer = gk_input
        else:
            writer = gk_inputs[gk_code]

        writer.set(
            local_geometry=self.local_geometry,
            local_species=self.local_species,
            numerics=self.numerics,
            template_file=template_file,
        )
        writer.write(file_name, float_format=self.float_format)

    def add_flags(self, flags) -> None:
        """
        Adds flags to GK file.
        Updates local_geomtry, local_species, and numerics
        """
        try:
            self.gk_input.add_flags(flags)
        except AttributeError:
            raise RuntimeError(
                "Pyro.add_flags: Must have already read gyrokinetics input file"
            )
        self.local_geometry = self.gk_input.get_local_geometry()
        self.local_species = self.gk_input.get_local_species()
        self.numerics = self.gk_input.get_numerics()

    def load_gk_output(self, path: Optional[PathLike] = None, **kwargs) -> None:
        """
        Loads gyrokinetics output into Xarray Dataset. Sets the gk_output attribute.
        If provided with a path, it will attempt to read output from that path.
        Possible paths are:
        - GS2: Path to out.nc NetCDF4 file
        - CGYRO: Path to directory containing output files
        - GENE: Path to directory containing output files if numbered 0000, otherwise
                provide one filename from parameters_####, nrg_#### or field_####.
                Pyrokinetics will search for the other files in the same directory.
        """
        if path is None:
            # Check self.run_directory, and check self.gk_file.
            # If self.gk_file exists, figure out which code type it is.
            try:
                gk_file = self.gk_file
                run_directory = self.run_directory
            except AttributeError:
                raise RuntimeError(
                    "Pyro.load_gk_output: Please provide a path to the output file "
                    "(or directory of output files), or read in a gyrokinetics input "
                    "file first."
                )
            gk_type = gk_inputs[gk_file].file_type
            if gk_type == "GS2":
                path = gk_file.stem + ".out.nc"
            elif gk_type == "CGYRO":
                path = run_directory
            elif gk_type == "GENE":
                # If the input file is of the form name_####, get the numbered part and
                # search for 'parameters_####' in the run directory.
                num_part_regex = re.compile(r"(\d{4})")
                num_part_match = num_part_regex.search(str(gk_file.name))
                if num_part_match is None:
                    path = run_directory
                else:
                    path = run_directory / f"parameters_{num_part_match[0]}"
            else:
                # If you see this, it's likely because we haven't implemented a match
                # for a newly added gyrokinetics code.
                raise RuntimeError(
                    "Pyro.load_gk_output: Could not determine gyrokinetics type from "
                    "input file."
                )

        self.gk_ouput = gk_output_readers[path].read(path, **kwargs)

    # Utility for copying Pyro object

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Pyro object

        Returns
        -------
        Copy of pyro object
        """
        new_pyro = Pyro()

        for key, value in self.__dict__.items():
            setattr(new_pyro, key, deepcopy(value))

        return new_pyro
