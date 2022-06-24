import copy
import re
import warnings
import xarray as xr
import numpy as np

from pathlib import Path
from typing import Optional, List

from .gk_code import GKInput, gk_inputs, gk_output_readers
from .local_geometry import LocalGeometry, LocalGeometryMiller, local_geometries
from .local_species import LocalSpecies
from .numerics import Numerics
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
        gk_output_file (PathLike, optional): Filename or directory name for gyrokinetics
            output file(s). For GS2, the user should pass the '.out.nc' NetCDF4
            file. For CGYRO, the user should pass the directory containing output files
            with their standard names. For GENE, the user should pass one of
            'parameters_####', 'nrg_####' or 'field_####', where #### is 4 digits
            identifying the run. If one of these is passed, the others will be found
            in the same directory. Alternatively, the user should pass the directory
            containing 'parameters_0000', 'field_0000' and 'nrg_0000'.
        gk_code (str, optional): Type of gyrokinetics input file and output file. When
            set, this will skip file type inference. Possible values are 'GS2', 'CGYRO',
            or 'GENE'. If set to None, the file type is inferred automatically.
            If gk_code is set, but no gk_file is provided, the corresponding default
            template file will be read.
        gk_type (str, optional): DEPRECATED, synonym for gk_code. gk_code takes
            precedence.
        """

        self.float_format = ""
        self.base_directory = Path(__file__).parent

        # Each time a gk_file is read, we populate the following dicts, using the
        # provided/inferred gk_code as a key:
        self._gk_input_record = {}  # GKInput
        self._gk_file_record = {}  # Path, full file names (may be relative)
        self._numerics_record = {}  # Numerics
        self._local_geometry_record = {}  # LocalGeometry
        self._local_species_record = {}  # LocalSpecies
        self._gk_output_record = {}  # Xarray Dataset
        self._gk_output_file_record = {}  # Path, full file names (may be relative)

        # We also maintain a record of the last gyrokinetics type we used. This is used
        # to determine the current context, so when calling functions such as load_local,
        # we only overwrite the bits corresponding to the current context. Changing this
        # attribute will switch context to a different gyrokinetics code. If the user
        # hasn't already read in a gyrokinetics file for the new gk_code, a new GKInput
        # is created from a default template file, and the previous context is used to
        # overwrite its Numerics, LocalGeometry, and LocalSpecies.
        # We'll set this to None for now, as setting it to the provided gk_code without
        # first reading gk_file would cause us to read a default template file.
        self.gk_code = None

        # Prepare to read gk_file
        # Deprecation of gk_type: gk_code always takes precedent.
        if gk_type is not None:
            if gk_code is None:
                warnings.warn(
                    "gk_type is no longer used, please use gk_code instead",
                    DeprecationWarning,
                )
                gk_code = gk_type
            else:
                warnings.warn(
                    "gk_type is no longer used, gk_code will take precedence",
                    DeprecationWarning,
                )

        # If provided gk_code but not gk_file, read default file
        if gk_code is not None and gk_file is None:
            gk_file = gk_templates[gk_code]

        # Read gk_file if it exists
        # This will set the current context using self.gk_code, and fill in records for
        # gk_file, file_name, run_directory, local_geometry, local_species, and
        # numerics.
        if gk_file is not None:
            self.read_gk_file(gk_file, gk_code)

        # Set gyrokinetics context
        # If we just read a file, or gk_code is None, this won't do anything. Otherwise,
        # it'll read a default template file.
        if gk_code is not None:
            self.gk_code = gk_code

        # Load gk_output if it exists.
        # WARNING: gk_output_file may be of a different gyrokinetics type to gk_file,
        # and this may cause unexpected behaviour.
        if self.gk_output_file is not None:
            self.read_gk_output_file(self.gk_output_file, gk_code)

        # Load global equilibrium file if it exists
        if eq_file is not None:
            self.load_global_eq(eq_file, eq_type)

        # Load global kinetics file if it exists
        if kinetics_file is not None:
            self.load_global_kinetics(kinetics_file, kinetics_type)

    # ============================================================
    # Functions and  properties for handling gyrokinetics contexts

    @property
    def gk_code(self) -> str:
        """
        Returns the current gyrokinetics context as a string, typically the name of the
        gyrokinetics code (GS2, CGYRO, GENE, etc). If there is no gyrokinetics, this
        instead returns None.

        gk_code will be updated automatically when the user calls read_gk_file.
        """
        try:
            return self._gk_code
        except AttributeError:
            return None

    @gk_code.setter
    def gk_code(self, value: str) -> None:
        """
        Sets the current gyrokinetics context using a string, typically the name of
        the gyrokinetics code (GS2, CGYRO, GENE, etc). If set to None, this indicates
        that there is no gyrokinetics involved in this run. Raises an error if set to
        anything other than one of the strings in self.supported_gk_inputs or None.

        If the user has not previously read a gyrokinetics input file of the provided
        type, this will prompt Pyro to read in the default template file of that type.
        See the function '_switch_gk_context' for details.
        """
        if value is not None and value not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro: Invalid gk_code '{value}'")
        if value is not None and value not in self._gk_input_record:
            warn_msg = (
                f"Setting gk_code to {value} without first reading a gyrokinetics "
                "file of that type. Creating a new context using the default "
                f"template file for {value} and data from the previous context (if "
                "available)."
            )
            warnings.warn(warn_msg, UserWarning)
        self._switch_gk_context(value, force_overwrite=False)

    def _switch_gk_context(
        self,
        gk_code: str,
        template_file: Optional[PathLike] = None,
        force_overwrite: bool = False,
    ) -> None:
        """
        Sets a new gyrokinetics context. A Pyro object can contain data corresponding
        to multiple different gyrokinetics codes simultaneously, and these may each
        be modified independently. For instance, calling load_local will overwrite
        the LocalGeometry and LocalSpecies of only the current gyrokinetics context.
        If the user switches context and makes some changes, they may then switch
        back without losing any data.

        The usual way for the user to switch context is by setting the gk_code
        attribute. If they do so without first having read a gyrokinetics input file,
        this will read in the default template file. The context will also be switched
        when the user calls read_gk_file or convert_gk_file, both of which will
        overwrite the context. The function write_gk_file may create a new context, but
        will not switch to it.
        """
        # Check that the provided gk_code is valid
        if gk_code is not None and gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"The gyrokinetics code '{gk_code}' is not supported")

        # If we've already seen this context before, or the new context is None, change
        # context and return.
        if gk_code is None or (
            gk_code in self._gk_input_record and not force_overwrite
        ):
            # Bypass the property setter, as otherwise we'd create an infinite loop
            self._gk_code = gk_code
            return

        # If we've gotten this far, this is a new gyrokinetics context.
        # Determine if we have any local_geometry, local_species, or numerics to copy
        # accoss. These may have been created from the previous context, or may have
        # been created by a call to load_local_geometry, load_local_species, or
        # load_local. If they have not been previously set, they will be None.
        local_geometry = self.local_geometry
        local_species = self.local_species
        numerics = self.numerics

        # Read in a template.
        # Begin by getting a default template file, unless one was provided.
        if template_file is None:
            template_file = gk_templates[gk_code]
        # If we've just come from a valid context, don't bother processing the data
        # to obtain LocalGeometry, LocalSpecies, or Numerics -- we'll be replacing
        # them in the next step.
        no_process = []
        if local_geometry is not None:
            no_process.append("local_geometry")
        if local_species is not None:
            no_process.append("local_species")
        if numerics is not None:
            no_process.append("numerics")
        # The following call will switch the gk_code context, so the properties
        # gk_input, gk_file, file_name, run_directory, local_geometry, local_species and
        # numerics will now refer to different objects.
        self.read_gk_file(template_file, gk_code=gk_code, no_process=no_process)

        # Copy across the previous numerics, local_geometry and local_species, if they
        # were found. Note that the context has now been switched, so
        # self.local_geometry now refers to a new object.
        if local_geometry is not None:
            self.local_geometry = copy.deepcopy(local_geometry)
        if local_species is not None:
            self.local_species = copy.deepcopy(local_species)
        if numerics is not None:
            self.numerics = copy.deepcopy(numerics)

        # If local_geometry, local_species or numerics do not match the template, update
        # the new GKInput.
        if np.any([x is not None for x in [local_geometry, local_species, numerics]]):
            self.update_gk_code()

    def check_gk_code(self, raises: bool = True) -> bool:
        """
        Checks if the current gyrokinetics context is 'valid', meaning it contains a
        GKInput, Numerics, LocalGeometry, and LocalSpecies. If 'raises' is True
        (default), raises RuntimeError when any required objects are missing, and
        returns True otherwise. If 'raises' is False, instead returns False when any
        required objects are missing, and True otherwise.
        """
        missing = []
        if self.gk_input is None:
            missing.append("gk_input")
        if self.local_geometry is None:
            missing.append("local_geometry")
        if self.local_species is None:
            missing.append("local_species")
        if self.numerics is None:
            missing.append("numerics")
        if raises and missing:
            raise RuntimeError(f"Missing the attributes {', '.join(missing)}.")
        return not bool(missing)

    def update_gk_code(self) -> None:
        """
        Modifies gk_input to account for any changes to local_geometry, local_species,
        or numerics. Only modifies the current context, as specified by gk_code.
        """
        # Ensure we have the correct attributes set
        try:
            self.check_gk_code()
        except RuntimeError as e:
            raise RuntimeError(f"Pyro.update_gk_code: {str(e)}")
        # Update gk_input to account for changes
        self.gk_input.set(
            local_geometry=self.local_geometry,
            local_species=self.local_species,
            numerics=self.numerics,
        )

    def convert_gk_code(
        self, gk_code: str, template_file: Optional[PathLike] = None
    ) -> None:
        """
        Convert the current gyrokinetics context to a new one, overwriting any gk_input,
        gk_file, file_name, run_directory, local_geometry, local_species, and numerics
        already associated with it.
        Will create a new context if one is not provided.
        """
        # Ensure gk_code is valid
        if gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro: Cannot convert to gk_code '{gk_code}'")

        # Ensure we're in a valid context
        # Ensure we have the correct attributes set
        try:
            self.check_gk_code()
        except RuntimeError as e:
            raise RuntimeError(f"Pyro.convert_gk_code: {str(e)}")

        # Switch context and overwrite everything
        self._switch_gk_context(gk_code, template_file, force_overwrite=True)

    def add_flags(self, flags) -> None:
        """
        Adds flags to GK file.
        """
        # FIXME We currently call update_gk_code before writing, and this can overwrite
        #      some user-set flags. I'd considered storing the flags and adding them
        #      back in before writing, but this could lead to inconsistencies in the
        #      final input file.

        # Check that we have a gk_input to work with
        try:
            gk_input = self.gk_input
        except AttributeError:
            raise RuntimeError(
                "Pyro.add_flags: Must have already read gyrokinetics input file"
            )
        gk_input.add_flags(flags)
        self.local_geometry = gk_input.get_local_geometry()
        self.local_species = gk_input.get_local_species()
        self.numerics = gk_input.get_numerics()

    # ========================================================
    # Functions and properties for handling gyrokinetics files

    @property
    def gk_input(self) -> GKInput:
        """
        Get the GKInput object for the current gyrokinetics context (gk_code). If we
        have no gyrokinetics context, return None.
        """
        try:
            return self._gk_input_record[self.gk_code]
        except KeyError:
            return None

    @gk_input.setter
    def gk_input(self, value: GKInput) -> None:
        """
        Set the GKInput object for the current gyrokinetics context (gk_code). Raises
        RuntimeError of the gyrokinetics context is invalid or the input is not a
        GKInput.
        """
        if self.gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro.gk_input.setter: Invalid gk_code: {self.gk_code}")
        if not isinstance(value, GKInput):
            raise RuntimeError("Pyro.gk_input.setter: value is not a GKInput")
        self._gk_input_record[self.gk_code] = value

    @property
    def gk_output(self) -> xr.Dataset:
        """
        Get the GKInput object for the current gyrokinetics context (gk_code). If we
        have no gyrokinetics context, return None.
        """
        try:
            return self._gk_output_record[self.gk_code]
        except KeyError:
            return None

    @gk_output.setter
    def gk_output(self, value: xr.Dataset) -> None:
        """
        Set the gyrokinetics output for the current gyrokinetics context (gk_code).
        Raises RuntimeError of the gyrokinetics context is invalid.
        """
        if self.gk_code not in self.supported_gk_output_readers:
            raise RuntimeError(
                f"Pyro.gk_output.setter: Invalid gk_code: {self.gk_code}"
            )
        self._gk_output_record[self.gk_code] = value

    @property
    def gk_file(self) -> Path:
        """
        Retrieves the gyrokinetics input file path corresponding to the current
        gyrokinetics context (gk_code). If set via the corresponding property setter,
        this will be of type 'Path'. If gk_code is None, this function returns None.
        """
        try:
            return self._gk_file_record[self.gk_code]
        except KeyError:
            return None

    @gk_file.setter
    def gk_file(self, value: PathLike) -> None:
        """
        Converts the input to Path and sets the gyrokinetics input file path
        corresponding to the current gyrokinetics context (gk_code). Raises an error if
        the input cannot be converted to Path. Does not check if the provided path
        exists, or is a real gyrokinetics file. Also raises an error if the gyrokinetics
        context is invalid.
        """
        if self.gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro.gk_file.setter: Invalid gk_code: {self.gk_code}")
        self._gk_file_record[self.gk_code] = Path(value)

    @property
    def file_name(self) -> str:
        """
        Gets the final path component of gk_file, excluding any directories.
        """
        return self.gk_file.name

    @property
    def run_directory(self) -> Path:
        """
        Gets the directory of gk_file
        """
        return self.gk_file.parent

    @property
    def gk_output_file(self):
        """
        Retrieves the gyrokinetics output file path corresponding to the current
        gyrokinetics context (gk_code). If set via the corresponding property setter,
        this will be of type 'Path'. If gk_code is None, this function returns None.
        """
        try:
            return self._gk_output_file_record[self.gk_code]
        except KeyError:
            return None

    @gk_output_file.setter
    def gk_output_file(self, value: PathLike) -> None:
        """
        Converts 'value' to Path and sets the gyrokinetics output file path
        corresponding to the current gyrokinetics context (gk_code). Raises an error if
        the input cannot be converted to Path. Does not check if the provided path
        exists, or is a real gyrokinetics output. Also raises an error if the
        gyrokinetics context is invalid.
        """
        if self.gk_code not in self.supported_gk_output_readers:
            raise RuntimeError(
                f"Pyro.gk_output_file.setter: Invalid gk_code {self.gk_code}. The "
                "output reader for this gk_code may not be implemented."
            )
        self._gk_output_file_record[self.gk_code] = Path(value)

    def read_gk_file(
        self,
        gk_file: PathLike,
        gk_code: Optional[str] = None,
        no_process: List[str] = [],
    ):
        """
        Read gyrokinetics file, and set the gyrokinetics context to match the new file.
        Sets the gk_input, gk_file, file_name, run_directory, local_geometry,
        local_species, and numerics for this context (Advanced usage: the last three may
        optionally be skipped using the 'no_process' arg).

        NOTE: In previous versions, if a global Equilibrium was loaded, then this would
        read the gk_file but not load local_geometry. Now, it will overwrite a
        local_geometry created via load_local_geometry, but this can be fixed by calling
        load_local_geometry again with the appropriate psi_n.

        Parameters:
        -----------
            gk_file (PathLike): Path to a gyrokinetics input file.
            gk_code (Optional, str): The type of the gyrokinetics input file, such as
                'GS2', 'CGYRO', or 'GENE'. If unset, or set to None, the type will be
                inferred from gk_file.
            no_process (List[str]): Advanced, not recommended for use by users. If
                this list contains the string 'local_geometry', we do not create a
                LocalGeometry object from this gk_file. Similarly, if the list contains
                the string 'local_species', we do not create a LocalSpecies, and if the
                list contains the string 'numerics', we do not create a Numerics. This
                should be used if there is an expectation that these objects will not
                be needed, saving the overhead of creating them.
        """
        # Get an appropriate GKInput. Use gk_code if provided, or otherwise infer it
        # from gk_file.
        gk_input = gk_inputs[gk_file if gk_code is None else gk_code]

        # Read the file before setting any attributes. If an exception is raised here,
        # the Pyro object will be left in a usable state, and the context will not be
        # changed.
        gk_input.read(gk_file)

        # Switch to new context by setting self._gk_code.
        # Here we bypass property setter, as this function may be called by it, and this
        # could lead to an infinite loop.
        self._gk_code = gk_input.file_type

        # Set GKInput and file info within the new context
        # This uses property setters, which redirect to self._gk_input_record[gk_code]
        # or similar.
        self.gk_input = gk_input
        self.gk_file = gk_file  # property setter also converts to Path

        # Set LocalGeometry, LocalSpecies, Numerics, unless told not to.
        if "local_geometry" not in no_process:
            self.local_geometry = self.gk_input.get_local_geometry()
        if "local_species" not in no_process:
            self.local_species = self.gk_input.get_local_species()
        if "numerics" not in no_process:
            self.numerics = self.gk_input.get_numerics()

    def write_gk_file(
        self,
        file_name: PathLike,
        gk_code: Optional[str] = None,
        template_file: Optional[PathLike] = None,
    ) -> None:
        """
        Writes a gyrokinetics input file to file_name. If the user wishes to write to
        a new gk_code, a new gyrokinetics context is created using template_file
        (if template_file is not None) or the default template for that gk_code
        (if template_file is None). If gk_code corresponds to an existing gyrokinetics
        context, that context is written without change and template_file is ignored.

        If the user wishes to write the current gyrokinetics context (gk_code is None,
        or gk_code is set to the current value of self.gk_code), this will update the
        context with any changes made to local_species, local_geometry, or numerics.
        """
        # Get record of current gyrokinetics context
        prev_gk_code = self.gk_code

        # If the provided gk_code is None, set it to the current self.gk_code
        if gk_code is None:
            gk_code = self.gk_code

        # Switch context if needed, creating a new one if necessary.
        if prev_gk_code != gk_code:
            if gk_code in self._gk_input_record and template_file is not None:
                warn_msg = (
                    f"Provided template file '{template_file}' to write_gk_file, but "
                    "there is already data available for the gyrokinetics code "
                    f"{gk_code}. Ignoring the template file."
                )
                warnings.warn(warn_msg, UserWarning)
            # TODO Should this overwrite with the current context? It'll do so if the
            # new context doesn't already exist, but if it does already exist it won't
            # be changed.
            self._switch_gk_context(
                gk_code, template_file=template_file, force_overwrite=False
            )

        # Update to account for any changes to this context
        self.update_gk_code()

        # Set file info in new context
        self.gk_file = Path(file_name)

        # Write to disk
        self.gk_input.write(self.gk_file, float_format=self.float_format)

        # Switch back to original context
        self._switch_gk_context(prev_gk_code, force_overwrite=False)

    def load_gk_output(self, path: Optional[PathLike] = None, **kwargs) -> None:
        """
        Loads gyrokinetics output into Xarray Dataset. Sets the gk_output attribute for
        the current context. If provided with a path, it will attempt to read output
        from that path.
        The valid paths for each code are:
        - GS2: Path to out.nc NetCDF4 file
        - CGYRO: Path to directory containing output files
        - GENE: Path to directory containing output files if numbered 0000, otherwise
                provide one filename from parameters_####, nrg_#### or field_####.
                Pyrokinetics will search for the other files in the same directory.
        """
        if path is None:
            # Check self.run_directory, and check self.gk_file.
            # If self.gk_file exists, figure out which code type it is.
            gk_file = self.gk_file
            run_directory = self.run_directory
            if gk_file is None:
                raise RuntimeError(
                    "Pyro.load_gk_output: Please provide a path to the output file "
                    "(or directory of output files), or read in a gyrokinetics input "
                    "file first."
                )
            gk_type = gk_inputs[gk_file].file_type
            if gk_type == "GS2":
                path = self.run_directory / (gk_file.stem + ".out.nc")
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

        self.gk_output_file = path
        self.gk_output = gk_output_readers[path].read(path, **kwargs)

    # ==================================
    # Set properties for file attributes

    # Equilibrium files

    @property
    def eq_file(self) -> Path:
        """
        Returns self._eq_file if it exists, and otherwise returns None. self._eq_file
        should be of type 'Path' if assigned via the corresponding property setter.
        """
        try:
            return self._eq_file
        except AttributeError:
            return None

    @eq_file.setter
    def eq_file(self, value: PathLike) -> None:
        """
        Converts the input to Path and assigns to self._eq_file. Raises an error if the
        input cannot be converted to Path. Does not check if the provided path exists,
        or is a real equilibrium file.
        """
        self._eq_file = Path(value)

    # Kinetics files

    @property
    def kinetics_file(self):
        """
        Returns self._kinetics_file if it exists, and otherwise returns None.
        self._kinetics_file should be of type 'Path' if assigned via the corresponding
        property setter.
        """
        try:
            return self._kinetics_file
        except AttributeError:
            return None

    @kinetics_file.setter
    def kinetics_file(self, value: PathLike) -> None:
        """
        Converts the input to Path and assigns to self._kinetics_file. Raises an error
        if the input cannot be converted to Path. Does not check if the provided path
        exists, or is a real kinetics file.
        """
        self._kinetics_file = Path(value)

    # Define local_geometry property
    # By providing string like 'Miller', sets self.local_geometry to LocalGeometryMiller

    @property
    def local_geometry(self) -> LocalGeometry:
        """
        If there is no gyrokinetics, get _local_geometry_from_global. Otherwise, get
        local_geometry from the current context as specified by gk_code
        """
        try:
            return self._local_geometry_record[self.gk_code]
        except KeyError:
            try:
                return self._local_geometry_from_global
            except AttributeError:
                return None

    @local_geometry.setter
    def local_geometry(self, value) -> None:
        """
        Sets the local geometry type. If there is no gyrokinetics, this will assign
        to _local_geometry_from_global. Otherwise, it will assign to only the current
        context, in _local_geometry_record[gk_code]. If set
        """
        # FIXME When set with a string, this can result in the creation of
        # uninitialised instances and cause unexpected behaviour. May be preferable to
        # implement a 'convert_local_geometry' function once other LocalGeometry types
        # are implemented, and to disallow converting LocalGeometry types by assigning
        # strings to the local_geometry attribute. Currently, this behaviour is only
        # used within load_local_geometry, where an uninitialised LocalGeometry is
        # created and then populated using load_from_eq. We can do away with this by
        # implementing a 'from_eq' classmethod within LocalGeometry types, to be
        # used as an alternative to the standard constructor.
        if isinstance(value, LocalGeometry):
            local_geometry = value
        elif value in self.supported_local_geometries:
            local_geometry = local_geometries[value]
        elif value is None:
            local_geometry = None
        else:
            raise NotImplementedError(f"LocalGeometry {value} not yet supported")
        # If we have gyrokinetics, set to _local_geometry_record, and otherwise set
        # to _local_geometry_from_global
        if self.gk_code is None:
            self._local_geometry_from_global = local_geometry
        else:
            self._local_geometry_record[self.gk_code] = local_geometry

    @property
    def local_geometry_type(self) -> str:
        # Check we have a local geometry. Return None if we don't
        try:
            local_geometry = self.local_geometry
        except AttributeError:
            return None

        # Determine which kind of LocalGeometry we have
        if isinstance(local_geometry, LocalGeometryMiller):
            return "Miller"
        elif local_geometry is None:
            return None
        else:
            raise RuntimeError(
                "Pyro._local_geometry is set to an unknown geometry type"
            )

    # local species property
    @property
    def local_species(self) -> LocalSpecies:
        """
        If there is no gyrokinetics, get _local_species_from_global. Otherwise, get
        local_species from the current context as specified by gk_code
        """
        try:
            return self._local_species_record[self.gk_code]
        except KeyError:
            try:
                return self._local_species_from_global
            except AttributeError:
                return None

    @local_species.setter
    def local_species(self, value: LocalSpecies) -> None:
        """
        If there is no gyrokinetics, set _local_species_from_global. Otherwise, set
        local_species in the current context as specified by gk_code
        """
        if not isinstance(value, LocalSpecies):
            raise RuntimeError("Pyro.local_species.setter: value is not LocalSpecies")
        if self.gk_code is None:
            self._local_species_from_global = value
        else:
            self._local_species_record[self.gk_code] = value

    # numerics property
    @property
    def numerics(self) -> Numerics:
        """
        If there is no gyrokinetics (gk_code is None), returns None. Otherwise gets
        numerics from the current context.
        """
        try:
            return self._numerics_record[self.gk_code]
        except KeyError:
            return None

    @numerics.setter
    def numerics(self, value: Numerics) -> None:
        """
        Sets numerics in the current gyrokinetics context.
        Raises if there is no gyrokinetics context (gk_code is None), or if value is
        not a Numerics.
        """
        if not isinstance(value, Numerics):
            raise RuntimeError("Pyro.numerics.setter: value is not Numerics")
        try:
            self._numerics_record[self.gk_code] = value
        except KeyError:
            raise RuntimeError(
                "Pyro.numerics.setter: Must have a gyrokinetics context. "
                "Try setting gk_code first."
            )

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
            setattr(new_pyro, key, copy.deepcopy(value))

        return new_pyro

    # Add properties that allow direct access to GKInput.data
    # TODO This feels dangerous... could use a refactor
    # Not sure how to automate generation of these when new gk_codes are added
    @property
    def gs2_input(self):
        return self._gk_input_record["GS2"].data

    @property
    def cgyro_input(self):
        return self._gk_input_record["CGYRO"].data

    @property
    def gene_input(self):
        return self._gk_input_record["GENE"].data
