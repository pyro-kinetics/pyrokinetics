"""The `Pyro` class is the primary interface for
reading/writing/manipulating gyrokinetics files. It is used to manage
the following objects:

- `Equilibrium`
- :py:class:`LocalGeometry`
- `Kinetics`
- `Numerics`
- :py:class:`GKInput`
- :py:class:`GKOutput` (as `xarray.Dataset`)

"""

from __future__ import annotations

import copy
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import f90nml
import numpy as np

from .equilibrium import read_equilibrium, supported_equilibrium_types
from .gk_code import (
    GKInput,
    GKOutput,
    read_gk_input,
    read_gk_output,
    supported_gk_input_types,
    supported_gk_output_types,
)
from .kinetics import read_kinetics, supported_kinetics_types
from .local_geometry import (
    LocalGeometry,
    LocalGeometryFourierCGYRO,
    LocalGeometryFourierGENE,
    LocalGeometryMiller,
    LocalGeometryMillerTurnbull,
    LocalGeometryMXH,
    MetricTerms,
    local_geometry_factory,
)
from .local_species import LocalSpecies
from .normalisation import ConventionNormalisation as Normalisation
from .normalisation import SimulationNormalisation
from .numerics import Numerics
from .templates import gk_templates
from .typing import PathLike
from .units import PyroQuantity

if TYPE_CHECKING:
    import xarray as xr


class Pyro:
    """
    An object able to read, write, run, analyse and plot GK data

    Parameters
    ----------
    eq_file : PathLike, default ``None``
        Filename for outputs from a global equilibrium code, such as GEQDSK or TRANSP.
        When passed, this will set the 'eq' attribute. This can be used to set a local
        geometry using the function load_local_geometry or load_local.
    eq_type : str, default ``None``
        Type of equilibrium file. When set, this will skip file type inference. Possible
        values are GEQDSK or TRANSP. If set to None, the file type is inferred
        automatically.
    eq_kwargs : Optional[Dict[str, Any]] = None
        Keyword arguments to be used when building an Equilibrium object
    kinetics_file : PathLike, default ``None``
        Filename for outputs from a global kinetics code, such as SCENE, JETTO,
        TRANSP, or pFile. When passed, this will set the 'kinetics' attribute. This can be used to
        set local kinetics using the function load_local_kinetics or load_local.
    kinetics_type : str, default ``None``
        Type of kinetics file. When set, this will skip file type inference. Possible
        values are SCENE, JETTO, TRANSP, or pFile. If set to None, the file type is inferred
        automatically.
    kinetics_kwargs : Optional[Dict[str, Any]] = None
        Keyword arguments to be used when building a Kinetics object.
    gk_file : PathLike, default ``None``
        Filename for a gyrokinetics input file (GS2, GENE, CGYRO). When passed, the
        attributes 'local_geometry', 'local_species', and 'numerics' are set.
    gk_output_file : PathLike, default ``None``
        Filename or directory name for gyrokinetics output file(s). For GS2, the user
        should pass the '.out.nc' NetCDF4 file. For CGYRO, the user should pass the
        directory containing output files with their standard names. For GENE, the user
        should pass one of 'parameters_####', 'nrg_####' or 'field_####', where #### is
        4 digits identifying the run. If one of these is passed, the others will be found
        in the same directory. Alternatively, the user should pass the directory
        containing 'parameters_0000', 'field_0000' and 'nrg_0000'.
    gk_code : str, default ``None``
        Type of gyrokinetics input file and output file. When set, this will skip file
        type inference. Possible values are 'GS2', 'CGYRO', or 'GENE'. If set to None,
        the file type is inferred automatically. If gk_code is set, but no gk_file is
        provided, the corresponding default template file will be read.
    gk_type : str, default ``None``
        Deprecated, synonym for gk_code. gk_code takes precedence.
    """

    # Keep track of how many times we've seen a given name
    _RUN_NAMES = Counter()

    def __init__(
        self,
        eq_file: Optional[PathLike] = None,
        eq_type: Optional[str] = None,
        eq_kwargs: Optional[Dict[str, Any]] = None,
        kinetics_file: Optional[PathLike] = None,
        kinetics_type: Optional[str] = None,
        kinetics_kwargs: Optional[Dict[str, Any]] = None,
        gk_file: Optional[PathLike] = None,
        gk_output_file: Optional[PathLike] = None,
        gk_code: Optional[str] = None,
        gk_type: Optional[str] = None,  # deprecated, synonym for gk_code
        nocos: Union[str, Normalisation] = "pyrokinetics",
        name: Optional[str] = None,
    ):
        self.float_format = ""
        self.base_directory = Path(__file__).parent
        self._local_geometry_species_dependency = False

        # Get a unique name for this instance, based off any of the inputs
        self.name = self._unique_name(
            name or gk_file or eq_file or kinetics_file or "run"
        )

        self.norms = SimulationNormalisation(self.name, convention=nocos)

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
            self.read_gk_file(gk_file, gk_code, norms=self.norms)

        # Set gyrokinetics context
        # If we just read a file, or gk_code is None, this won't do anything. Otherwise,
        # it'll read a default template file.
        if gk_code is not None:
            self.gk_code = gk_code

        self.nocos_number = nocos

        # Load gk_output if it exists.
        # WARNING: gk_output_file may be of a different gyrokinetics type to gk_file,
        # and this may cause unexpected behaviour.
        if self.gk_output_file is not None:
            self.read_gk_output_file(self.gk_output_file, gk_code)

        # Load global equilibrium file if it exists
        if eq_kwargs is None:
            eq_kwargs = {}

        if eq_file is not None:
            self.load_global_eq(eq_file, eq_type, **eq_kwargs)

        # Load global kinetics file if it exists
        if kinetics_kwargs is None:
            kinetics_kwargs = {}

        if kinetics_file is not None:
            self.load_global_kinetics(kinetics_file, kinetics_type, **kinetics_kwargs)

        self._check_beta_consistency()

    def _unique_name(self, name: Union[str, PathLike]) -> str:
        """Return a unqiuely numbered run name from `name`"""
        # name might be a Path, in which case just use the filename
        # (without extension)

        name = getattr(Path(name), "stem", name)
        name = "".join([ch for ch in name if ch.isalpha() or ch.isdigit() or ch == "_"])

        new_name = f"{name}{self._RUN_NAMES[name]:04}"
        self._RUN_NAMES[name] += 1
        return new_name

    # ============================================================
    # Properties for determining supported features

    @property
    def supported_gk_inputs(self) -> List[str]:
        """
        Returns a list of supported `GKInput` classes, expressed as strings. The user
        can add new `GKInput` classes by 'registering' them with `GKInput.reader`

        Returns
        -------
        List[str]
            List of supported `GKInput` classes, expressed as strings.
        """
        return supported_gk_input_types()

    @property
    def supported_gk_output_readers(self) -> List[str]:
        """
        Returns a list of supported `GKOutput` reader classes, expressed as strings. The
        user can add new `GKOutput` reader class by 'registering' them with
        `GKOutput.reader`

        Returns
        -------
        List[str]
            List of supported `GKOutput` reader classes, expressed as strings.
        """
        return supported_gk_output_types()

    @property
    def supported_local_geometries(self) -> List[str]:
        """
        Returns a list of supported `LocalGeometry` classes, expressed as strings. The
        user can add new `LocalGeometry` classes by 'registering' them with
        `local_geometry.local_geometry_factory`.

        Returns
        -------
        List[str]
            List of supported LocalGeometry classes, expressed as strings.
        """
        return [*local_geometry_factory]

    @property
    def supported_equilibrium_types(self) -> List[str]:
        """
        Returns a list of supported `Equilibrium` types, expressed as strings (e.g.
        GEQDSK, TRANSP). The user can add new `Equilibrium` reader classes by
        'registering' them with `Equilibrium.reader`

        Returns
        -------
        List[str]
            Supported `Equilibrium` file types expressed as strings.
        """
        return supported_equilibrium_types()

    @property
    def supported_kinetics_types(self) -> List[str]:
        """
        Returns a list of supported `Kinetics` types, expressed as strings (e.g. JETTO,
        SCENE, TRANSP). The user can add new `Kinetics` reader classes by 'registering'
        them with `Kinetics.reader`

        Returns
        -------
        List[str]
            List of supported `Kinetics` file types, expressed as strings.
        """
        return supported_kinetics_types()

    # ============================================================
    # Functions and  properties for handling gyrokinetics contexts

    @property
    def gk_code(self) -> Union[str, None]:
        """
        The current gyrokinetics context, expressed as a string. This is typically the
        name of the gyrokinetics code (GS2, CGYRO, GENE, etc). If there is no
        gyrokinetics context (i.e. only global equilibrium or kinetics components exist)
        this is instead None.

        When setting gk_code, the gyrokinetics context. If set to ``None``, the context
        is voided, and the properties ``local_geometry``, ``local_species``, and
        ``numerics`` will no longer return anything meaningful.

        ``gk_code`` will be updated automatically when the user calls ``read_gk_file``
        or ``convert_gk_code``, as these functions create and switch to a new context.
        If ``gk_code`` is set to a new value without first having read a gyrokinetics
        input file, a new context is created by reading the appropriate default template
        file, and, if available, copying the current ``local_geometry``,
        ``local_species`` and  ``numerics``.

        See ``_switch_gk_context`` for details on gyrokinetics contexts.

        Returns
        -------
        ``str`` or ``None``
            The current gyrokinetics context if it exists, otherwise ``None``.

        Raises
        ------
        ValueError
            If set to anything other than one of the strings in ``supported_gk_inputs``
            or ``None``.

        """
        try:
            return self._gk_code
        except AttributeError:
            return None

    @gk_code.setter
    def gk_code(self, value: str) -> None:
        if value is not None and value not in self.supported_gk_inputs:
            raise ValueError(f"Pyro: Invalid gk_code '{value}'")
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
        Sets a new gyrokinetics context. Not intended for use by the user.

        A ``Pyro`` object can contain data corresponding to multiple different
        gyrokinetics codes simultaneously, and these may each be modified independently.
        For instance, calling ``load_local`` will overwrite ``local_geometry`` and
        ``local_species`` of only the current gyrokinetics context. If the user switches
        context and makes some changes, they may then switch back without losing any
        data. Each context contains the following attributes:

        - ``gk_input``: GKInput object
        - ``gk_file``: filename of the gyrokinetics input
        - ``numerics``: Numerics object
        - ``local_geometry``: LocalGeometry object
        - ``local_species``: LocalSpecies object
        - ``gk_output``: Xarray ``Dataset`` containing gyrokinetics output
        - ``gk_output_file``: filename of the gyrokinetics ouput

        These attributes are implemented using properties, and when changing context the
        attributes will redirect to new objects.

        The usual way for the user to switch context is by setting the ``gk_code``
        attribute. If they do so without first having read a gyrokinetics input file of
        that type, this will read in the file specified in ``template_file``, or the
        default template file if this is ``None``. If there is a ``local_geometry``,
        ``local_species`` or ``numerics`` in the current context, these are copied
        across, and these will be reflected in the new `gk_input`.

        The context will be switched when the user calls ``read_gk_file`` or
        ``convert_gk_file``, both of which will overwrite the context. The function
        ``write_gk_file`` may create a new context, but will not switch to it. The
        function ``update_gk_code`` will sync any changes made to the current context
        (i.e. if the user makes manual changes to ``numerics`` and calls
        ``update_gk_code``, those changes are then reflected in ``gk_input``).

        Parameters
        ----------
        gk_code : str
            The gyrokinetics context to switch to. May take the values of anything
            in ``supported_gk_inputs``, or ``None``.
        template_file : PathLike, default ``None``
            Sets the template file to read from when creating a new context. If set to
            ``None``, reads the default template file corresponding to ``gk_code``.
        force_overwrite : bool, default ``False``
            If ``True``, calling this function with a ``gk_code`` that has already
            been used, and therefore already has a context, will overwrite that context.
            If set to ``False``, calling this function with a ``gk_code`` that has
            already been used will simply enter the existing context without changing
            any internal data.

        Returns
        -------
        ``None``

        Raises
        ------
        ValueError
            If the user tries to set ``gk_code`` to something that isn't in
            ``supported_gk_inputs`` or ``None``.
        Exception
            A number of errors may be raised while reading template files.
        """
        # Check that the provided gk_code is valid
        if gk_code is not None and gk_code not in self.supported_gk_inputs:
            raise ValueError(f"The gyrokinetics code '{gk_code}' is not supported")

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

        # Only calculate beta/gamma_exb if not set already
        if self.numerics is not None:
            set_beta = False
            set_gamma_exb = False
        else:
            set_beta = True
            set_gamma_exb = True

        # Check if data requiring LocalGeometry & LocalSpecies has been loaded
        if (
            not self._local_geometry_species_dependency
            and local_species
            and local_geometry
        ):
            self._load_local_geometry_species_dependency(
                set_beta=set_beta, set_gamma_exb=set_gamma_exb
            )

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
        self.read_gk_file(
            template_file, gk_code=gk_code, no_process=no_process, norms=self.norms
        )

        # Need to remove beta from template file otherwise won't be set and set gamma_exb
        if self.numerics:
            self._load_local_geometry_species_dependency(set_rhoref=False)

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
        GKInput, Numerics, LocalGeometry, and LocalSpecies.

        Parameters
        ----------
        raises : bool, default ``True``
            If ``raises`` is  ``True``, the function raises ``RuntimeError`` when any
            required objects are missing, and returns ``True`` otherwise. If ``raises``
            is ``False``, the function does not raise, and instead returns ``False``
            when any required objects are missing, and ``True`` otherwise.

        Returns
        -------
        ``bool``
            ``True`` if the current context is valid, ``False`` otherwise

        Raises
        ------
        RuntimeError
            If the current context is valid (only when ``raises`` is ``True``)
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

    def update_gk_code(self, code_normalisation: Optional[str] = None) -> None:
        """
        Modifies ``gk_input`` to account for any changes to ``local_geometry``,
        ``local_species``, or ``numerics``. Only modifies the current context, as
        specified by ``gk_code``.

        code_normalisation: str, default ``None``
            When writing a file this selects which normalisation convention to use
            when populating the input file. If unset or set to ``None``, the default
            for each code is used
        Returns
        -------
        ``None``

        Raises
        ------
        RuntimeError
            If ``check_gk_code()`` fails
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
            local_norm=self.norms,
            code_normalisation=code_normalisation,
        )

    def convert_gk_code(
        self, gk_code: str, template_file: Optional[PathLike] = None
    ) -> None:
        """
        Convert the current gyrokinetics context to a new one, overwriting any gk_input,
        gk_file, file_name, run_directory, local_geometry, local_species, and numerics
        already associated with it.

        Will create a new context if one is not already present. If provided with a
        template file, this will be used to create a new GKInput object, which will then
        be modified by the current ``local_geometry``, ``local_species``, and
        ``numerics`` (if present). If no template file is specified, the default
        template corresponding to ``gk_code`` is used instead.

        If you don't wish to use the current ``local_geometry``, ``local_species`` and
        ``numerics``, it is recommended to use the function ``read_gk_file`` instead.

        Parameters
        ----------
        gk_code: str
            The gyrokinetics code to convert to. Must be a value in
            ``supported_gk_inputs``.
        template_file: PathLike, default ``None``
            The template file used to populate the new GKInput created. Note that some
            inputs in the template file will be overwritten with the contents of the
            current ``local_geometry``, ``local_species`` and ``numerics``. If ``None``,
            uses the default template file corresponding to ``gk_code``

        Returns
        -------
        ``None``

        Raises
        ------
        ValueError
            Provided gk_code is not in ``supported_gk_inputs``.
        RuntimeError
            If ``check_gk_code()`` fails.
        Exception
            A large variety of errors could occur when building a GKInput from a
            template file, or setting its values using the current ``local_geometry``,
            ``local_species``, and ``numerics``.

        """
        # Ensure gk_code is valid
        if gk_code not in self.supported_gk_inputs:
            raise ValueError(f"Pyro: Cannot convert to gk_code '{gk_code}'")

        # Ensure we're in a valid context
        # Ensure we have the correct attributes set
        try:
            self.check_gk_code()
        except RuntimeError as e:
            raise RuntimeError(f"Pyro.convert_gk_code: {str(e)}")

        # Switch context and overwrite everything
        self._switch_gk_context(gk_code, template_file, force_overwrite=True)

    def add_flags(self, flags: Dict[str, Any]) -> None:
        """
        Adds flags to ``gk_input``. Sets ``local_geometry``, ``local_species`` and
        ``numerics`` to account for any changes. Note that this will overwrite any
        changes the user has made to these objects that aren't already reflected in
        ``gk_input``.

        WARNING: Some flag changes are not persistent when writing to file or calling
        ``update_gk_code``, as calls to ``GKInput.set`` will sometimes overwrite flags
        in unexpected ways.

        Parameters
        ----------
        flags: Dict[str,Any]
            Dict of key-value pairs matching the format of a given gyrokinetics input
            file. For example, GS2 uses Fortran namelists, so flags should be a
            dict-of-dicts: one for each group in the namelist.

        Returns
        -------
        ``None``

        Raises
        ------
        RuntimeError
            If ``gk_input`` is ``None``, i.e. the user has not read a gyrokinetics file,
            or the user has set ``pyro.gk_code=None``.
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
        self._check_beta_consistency()

    # ========================================================
    # Functions and properties for handling gyrokinetics files

    @property
    def gk_input(self) -> Union[GKInput, None]:
        """
        The ``GKInput`` object for the current gyrokinetics context. The user should not
        need to set this manually.

        Returns
        -------
        GKInput or ``None``
            If ``gk_code`` is not ``None``, returns the ``GKInput`` object for the
            current gyrokinetics context. Otherwise, returns ``None``.

        Raises
        ------
        TypeError
            If setting to a value which is not an instance of ``GKInput``
        """
        try:
            return self._gk_input_record[self.gk_code]
        except KeyError:
            return None

    @gk_input.setter
    def gk_input(self, value: GKInput) -> None:
        # The following should never occur, unless somebody messes with self._gk_code
        if self.gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro.gk_input.setter: Invalid gk_code: {self.gk_code}")
        if not isinstance(value, GKInput):
            raise TypeError("Pyro.gk_input.setter: value is not a GKInput")
        self._gk_input_record[self.gk_code] = value

    @property
    def gk_output(self) -> Union[xr.Dataset, None]:
        """
        The gyrokinetics output for the current gyrokinetics context (gk_code).

        Returns
        -------
        xarray.Dataset or ``None``
            If the user has loaded gyrokinetics output, this will be contained within
            an Xarray Dataset. If the user hasn't loaded this, returns`` None``.
        """
        try:
            return self._gk_output_record[self.gk_code]
        except KeyError:
            return None

    @gk_output.setter
    def gk_output(self, value: xr.Dataset) -> None:
        # The following should never occur, unless somebody messes with self._gk_code
        if self.gk_code not in self.supported_gk_output_readers:
            raise RuntimeError(
                f"Pyro.gk_output.setter: Invalid gk_code: {self.gk_code}"
            )
        self._gk_output_record[self.gk_code] = value

    @property
    def gk_file(self) -> Union[Path, None]:
        """
        The gyrokinetics input file path corresponding to the current gyrokinetics
        context. The user should not need to set this manually.

        Returns
        -------
        pathlib.Path or ``None``
            If ``gk_code`` is not ``None``, the path to the last read/written
            gyrokinetics file. Otherwise, ``None``.

        Raises
        ------
        TypeError
            If value cannot be converted to pathlib.Path.
        """
        try:
            return self._gk_file_record[self.gk_code]
        except KeyError:
            return None

    @gk_file.setter
    def gk_file(self, value: PathLike) -> None:
        # The following should never occur, unless somebody messes with self._gk_code
        if self.gk_code not in self.supported_gk_inputs:
            raise RuntimeError(f"Pyro.gk_file.setter: Invalid gk_code: {self.gk_code}")
        self._gk_file_record[self.gk_code] = Path(value)

    @property
    def file_name(self) -> Union[str, None]:
        """
        The final path component of ``gk_file``, excluding any directories. Has no
        setter.

        Returns
        -------
        ``str`` or ``None``
            If ``gk_file`` is not ``None``, returns the final part of the path
            (see pathlib.Path.name). Otherwise, ``None``.
        """
        return self.gk_file.name if self.gk_file is not None else None

    @property
    def run_directory(self) -> Union[Path, None]:
        """
        The directory containing ``gk_file``. Has no setter.

        Returns
        -------
        pathlib.Path or ``None``
            If ``gk_file`` is not ``None``, returns directory that contains ``gk_file``.
            Otherwise, ``None``.
        """
        return self.gk_file.parent if self.gk_file is not None else None

    @property
    def gk_output_file(self) -> Union[Path, None]:
        """
        The gyrokinetics output file path corresponding to the current gyrokinetics
        context. The user should not need to set this manually. Due to the varied
        nature of gyrokinetics outputs, this may point to a single file or a directory.

        Returns
        -------
        pathlib.Path or ``None``
            If ``gk_code`` is not ``None``, the path to the gyrokinetics output.
            Otherwise, ``None``.

        Raises
        ------
        TypeError
            If value cannot be converted to pathlib.Path.
        """
        try:
            return self._gk_output_file_record[self.gk_code]
        except KeyError:
            return None

    @gk_output_file.setter
    def gk_output_file(self, value: PathLike) -> None:
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
        no_process: List[str] = None,
        norms: SimulationNormalisation = None,
    ) -> None:
        """
        Reads a gyrokinetics input file, and set the gyrokinetics context to match the
        new file.

        Sets the gk_input, gk_file, file_name, run_directory, local_geometry,
        local_species, and numerics for this context (Advanced usage: the last three may
        optionally be skipped using the 'no_process' arg).

        NOTE: In previous versions, if a global Equilibrium was loaded, then this would
        read the gk_file but not load local_geometry. Now, it will overwrite a
        local_geometry created via load_local_geometry, but this can be fixed by calling
        load_local_geometry again with the appropriate psi_n.

        Parameters
        ----------
        gk_file : PathLike
            Path to a gyrokinetics input file.
        gk_code : str, default None
            The type of the gyrokinetics input file, such as 'GS2', 'CGYRO', or 'GENE'.
            If unset, or set to None, the type will be inferred from gk_file. Default is
            None.
        no_process : List[str], default None
            Not recommended for use by users. If this list contains the string
            'local_geometry', we do not create a LocalGeometry object from this gk_file.
            Similarly, if the list contains the string 'local_species', we do not create
            a LocalSpecies, and if the list contains the string 'numerics', we do not
            create a Numerics. This should be used if there is an expectation that these
            objects will not be needed, saving the overhead of creating them. If set
            to None, all objects will be included.

        Returns
        -------
        ``None``

        Raises
        ------
        Exception
            A large number of errors could occur when reading a gyrokinetics input file.
            For example, FileNotFoundError if ``gk_file`` is not a real file, or
            KeyError if the input file is missing some crucial flags. The possible
            errors and the Exception types associated with them will vary depending on
            the gyrokinetics code.
        """
        # Set up no_process
        if no_process is None:
            no_process = []

        # Get an appropriate GKInput. Use gk_code if provided, or otherwise infer it
        # from gk_file.
        # GKInput classes are both 'reader' and 'readable'. Must hold on to instance of
        # the reader.
        gk_input = read_gk_input(gk_file, file_type=gk_code)

        # Switch to new context by setting self._gk_code.
        # Here we bypass property setter, as this function may be called by it, and this
        # could lead to an infinite loop.
        self._gk_code = gk_input.file_type

        # Set GKInput and file info within the new context
        # This uses property setters, which redirect to self._gk_input_record[gk_code]
        # or similar.
        self.gk_input = gk_input
        self.gk_file = gk_file  # property setter also converts to Path

        # Checks to see if normalisation convention used matches expected value and if not
        # then creates the appropriate ConventionNormalisation and adds it to the
        # SimulationNormalisation object
        if norms:
            if self.gk_input._convention_dict:
                self.norms.add_convention_normalisation(
                    name=self.gk_input.norm_convention,
                    convention_dict=self.gk_input._convention_dict,
                )
                self.gk_input._convention_dict = {}

            self.gk_input.convention = getattr(
                self.norms, self.gk_input.norm_convention
            )

        # Set LocalGeometry, LocalSpecies, Numerics, unless told not to.
        if "local_geometry" not in no_process:
            self.local_geometry = self.gk_input.get_local_geometry()
            self.norms.set_ref_ratios(self.local_geometry)
        if "local_species" not in no_process:
            self.local_species = self.gk_input.get_local_species()
        if "numerics" not in no_process:
            self.numerics = self.gk_input.get_numerics()

        if norms:
            reference_dict = self.gk_input.get_reference_values(norms)
            if reference_dict:
                self.set_reference_values(**reference_dict)

    def read_gk_dict(
        self,
        gk_dict: dict,
        gk_code: str,
        no_process: List[str] = None,
    ) -> None:
        """
        Reads a dictionary equivalent of a gyrokinetics input file , and set the
        gyrokinetics context to match the dict

        Sets the gk_input, gk_file, file_name, run_directory, local_geometry,
        local_species, and numerics for this context (Advanced usage: the last three may
        optionally be skipped using the 'no_process' arg).

        NOTE: In previous versions, if a global Equilibrium was loaded, then this would
        read the gk_file but not load local_geometry. Now, it will overwrite a
        local_geometry created via load_local_geometry, but this can be fixed by calling
        load_local_geometry again with the appropriate psi_n.

        Parameters
        ----------
        gk_dict : dict
            Dictionary equivalent of gk_input file
        gk_code : str, default None
            The type of the gyrokinetics input file, such as 'GS2', 'CGYRO', or 'GENE'.
            If unset, or set to None, the type will be inferred from gk_file. Default is
            None.
        no_process : List[str], default None
            Not recommended for use by users. If this list contains the string
            'local_geometry', we do not create a LocalGeometry object from this gk_file.
            Similarly, if the list contains the string 'local_species', we do not create
            a LocalSpecies, and if the list contains the string 'numerics', we do not
            create a Numerics. This should be used if there is an expectation that these
            objects will not be needed, saving the overhead of creating them. If set
            to None, all objects will be included.

        Returns
        -------
        ``None``

        Raises
        ------
        Exception
            A large number of errors could occur when reading a gyrokinetics input file.
            For example, FileNotFoundError if ``gk_file`` is not a real file, or
            KeyError if the input file is missing some crucial flags. The possible
            errors and the Exception types associated with them will vary depending on
            the gyrokinetics code.
        """
        # Set up no_process
        if no_process is None:
            no_process = []

        # Get the appropriate GKInput type.
        gk_input = GKInput._factory(gk_code)

        # Read the file before setting any attributes. If an exception is raised here,
        # the Pyro object will be left in a usable state, and the context will not be
        # changed.
        gk_input.read_dict(gk_dict)

        # Switch to new context by setting self._gk_code.
        # Here we bypass property setter, as this function may be called by it, and this
        # could lead to an infinite loop.
        self._gk_code = gk_input.file_type

        # Set GKInput and file info within the new context
        # This uses property setters, which redirect to self._gk_input_record[gk_code]
        # or similar.
        self.gk_input = gk_input

        # Set LocalGeometry, LocalSpecies, Numerics, unless told not to.
        if "local_geometry" not in no_process:
            self.local_geometry = self.gk_input.get_local_geometry()
            self.norms.set_ref_ratios(self.local_geometry)
        if "local_species" not in no_process:
            self.local_species = self.gk_input.get_local_species()
        if "numerics" not in no_process:
            self.numerics = self.gk_input.get_numerics()

    def write_gk_file(
        self,
        file_name: PathLike,
        gk_code: Optional[str] = None,
        template_file: Optional[PathLike] = None,
        code_normalisation: Optional[str] = None,
    ) -> None:
        """
        Creates a new gyrokinetics input file. If ``gk_code`` is ``None``, or the same
        as the current context, this will call ``update_gk_code`` within the current
        context before writing. If ``gk_code`` instead corresponds to a different
        existing context, ``update_gk_code`` is called within that context before
        writing a file.

        If ``gk_code`` corresponds to a context that does not already exist, a new
        gyrokinetics context is created using the template file provided. If no template
        file is provided, the default template file for that ``gk_code`` is used.
        The provided template file is ignored if ``gk_code`` corresponds to an existing
        context, and a warning will be raised.

        This function will not change the context to ``gk_code``, unless an exception
        is raised part way through the function call, in which case the scenario could
        be either the current context of that of the provided ``gk_code``. The ``Pyro``
        object may be in an unstable state if this occurs.

        Parameters
        ----------
        file_name: PathLike
            Path to the new file. If file_name exists, the file will be overwritten.
        gk_code: str, default ``None``
            The type of the gyrokinetics input file to write, such as 'GS2', 'CGYRO',
            or 'GENE'. If unset, or set to ``None``, ``self.gk_code`` is used.
        template_file: PathLike, default ``None``
            When writing to a new ``gk_code``, this file will be used to populate the
            new ``GKInput`` created, which will in turn be updated using the current
            values of ``local_geometry``, ``local_species`` and ``numerics`` (if
            available). If ``gk_code`` corresponds to a context that already exists,
            this argument is ignored and a warning is raised.
        code_normalisation: str, default ``None``
            When writing a file this selects which normalisation convention to use
            when populating the input file. If unset or set to ``None``, the default
            for each code is used

        Returns
        -------
        ``None``

        Raises
        ------
        ValueError
            If ``gk_code`` is not ``None``, and not in ``supported_gk_inputs``.
        Exception
            Various errors may be raised while processing ``template_file``, calling
            ``update_gk_code()``, or when writing to disk.

        """
        # FIXME Ideally, this function should only write files, and should not be
        # modifying the internal state of the Pyro object, except perhaps when writing
        # to a new gk_code that the user hasn't used before. It may help if functions
        # such as load_geometry(), load_local_species() and add_flags() were updated to
        # include calls to update_gk_code().

        # Throw exception if gk_code is invalid
        if gk_code is not None and gk_code not in self.supported_gk_inputs:
            raise ValueError(f"Pyro.write_gk_file: Invalid gk_code '{gk_code}'")

        # Check if data requiring LocalGeometry & LocalSpecies has been loaded
        if not self._local_geometry_species_dependency:
            self._load_local_geometry_species_dependency(
                set_beta=False, set_gamma_exb=False
            )

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
        self.update_gk_code(code_normalisation=code_normalisation)

        # Set file info in new context
        self.gk_file = Path(file_name)

        # Write to disk
        self.gk_input.write(
            self.gk_file,
            float_format=self.float_format,
            local_norm=self.norms,
            code_normalisation=code_normalisation,
        )

        # Switch back to original context
        self._switch_gk_context(prev_gk_code, force_overwrite=False)

    def load_gk_output(
        self,
        path: Optional[PathLike] = None,
        local_norm: Optional[SimulationNormalisation] = None,
        output_convention="pyrokinetics",
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
        drop_nan=False,
        **kwargs,
    ) -> None:
        """
        Loads gyrokinetics output into Xarray Dataset.

        Sets the attributes gk_output and gk_output_file for the current context. If
        provided with a path, will attempt to read output from that path. Otherwise,
        will infer the path from the value of ``gk_file``.

        Parameters
        ----------
        path: PathLike, default None
            Path to the gyrokinetics output file/directory. Valid ``path`` for each code
            are:

            - GS2: Path to '\\*.out.nc' NetCDF4 file
            - CGYRO: Path to directory containing output files
            - GENE: Path to directory containing output files if numbered 0000,\
            otherwise provide one filename from parameters_####, nrg_#### or field_####.\
            Pyrokinetics will search for the other files in the same directory.

            If set to None, infers path from ``gk_file``.

        local_norm: SimulationNormalisation, default None
            SimulationNormalisation object used to convert between different unit systems
        output_convention: ConventionNormalisation, default "pyrokinetics"
            Convention to convert output to
        load_fields: bool, default True
            Flag to load fields or not
        load_fluxes: bool, default True
            Flag to load fluxes or not
        load_moments: bool, default False
            Flag to load moments or not
        drop_nan: bool, default False
            If NaNs are found in the output then that data is dropped. Off by default
        **kwargs
            Arguments to pass to the ``GKOutputReader``.

        Returns
        -------
        ``None``

        Raises
        ------
        RuntimeError
            If no path is provided, and no ``gk_file`` exists. Also if there is no
            current gyrokinetics context (i.e. ``pyro.gk_code`` is ``None``).
        Exception
            Various errors may occur while processing a gyrokinetics output.
        NotImplementedError
            If there is not GKOutputReader for ``gk_code``.
        """
        # TODO Currently require gk_code is not None. Is this a necessary restriction?

        if self.gk_code is None:
            raise RuntimeError(
                "Pyro.load_gk_output: gk_code must not be None. Try reading a "
                "gyrokinetics input file first."
            )

        if self.gk_code not in self.supported_gk_output_readers:
            raise NotImplementedError(
                "Pyro.load_gk_output: Have not implemented GKOutputReader for the "
                f"gk_code '{self.gk_code}'"
            )

        if path is None:
            if self.gk_file is None:
                raise RuntimeError(
                    "Pyro.load_gk_output: Please provide a path to the output file "
                    "(or directory of output files), or read in a gyrokinetics input "
                    "file first."
                )
            GKOutputReaderType = GKOutput._factory[self.gk_code]
            path = GKOutputReaderType.infer_path_from_input_file(self.gk_file)

        if local_norm is None:
            local_norm = self.norms

        self.gk_output_file = path
        self.gk_output = read_gk_output(
            path,
            norm=local_norm,
            output_convention=output_convention,
            load_fields=load_fields,
            load_fluxes=load_fluxes,
            load_moments=load_moments,
            **kwargs,
        )

        if drop_nan:
            self.gk_output.data = self.gk_output.data.dropna(dim="time")
            # Calculate growth_rate_tolerance with default inputs
            if (
                "eigenvalues" in self.gk_output
                and "time" in self.gk_output["eigenvalues"].dims
            ):
                self.gk_output.data["growth_rate_tolerance"] = (
                    self.gk_output.get_growth_rate_tolerance()
                )

    # ==================================
    # Set properties for file attributes

    # Equilibrium files

    @property
    def eq_file(self) -> Union[Path, None]:
        """
        Path to the global equilibrium file, if it exists. Otherwise returns None. The
        user should not have to set this manually. There is only one ``eq_file`` per
        ``Pyro`` object, shared by all ``gk_code``.

        Returns
        -------
        pathlib.Path or ``None``
            Path to the global equilibrium file if it exists, ``None`` otherwise.

        Raises
        ------
        TypeError
            If provided value cannot be converted to a pathlib.Path
        """
        try:
            return self._eq_file
        except AttributeError:
            return None

    @eq_file.setter
    def eq_file(self, value: PathLike) -> None:
        self._eq_file = Path(value)

    # Kinetics files

    @property
    def kinetics_file(self) -> Union[Path, None]:
        """
        Path to the global kinetics file, if it exists. Otherwise returns None. The
        user should not have to set this manually. There is only one ``kinetics_file``
        per ``Pyro`` object, shared by all ``gk_code``.

        Returns
        -------
        pathlib.Path or ``None``
            Path to the global kinetics file if it exists, ``None`` otherwise.

        Raises
        ------
        TypeError
            If provided value cannot be converted to a pathlib.Path
        """
        try:
            return self._kinetics_file
        except AttributeError:
            return None

    @kinetics_file.setter
    def kinetics_file(self, value: PathLike) -> None:
        self._kinetics_file = Path(value)

    # Define local_geometry property
    # By providing string like 'Miller', sets self.local_geometry to LocalGeometryMiller

    @property
    def local_geometry(self) -> Union[LocalGeometry, None]:
        """
        The ``LocalGeometry`` instance for the current gyrokinetics context, or if there
        is no gyrokinetics context (``self.gk_code`` is ``None``), a ``LocalGeometry``
        instance that isn't assigned to a context.

        The user may set ``local_geometry`` using a string matching any of the values in
        ``supported_local_geometries``, though this will create an empty
        ``LocalGeometry`` instance.

        Returns
        -------
        LocalGeometry or ``None``
            If ``self.gk_code`` is not ``None``, returns the ``local_geometry`` for this
            gyrokinetics context if it exists. Otherwise, returns an 'unassigned'
            ``local_geometry`` if it exists. Failing this, returns None.

        Raises
        ------
        NotImplementedError
            If setting to a value that isn't an instance of ``LocalGeometry``, or a
            string matching those in ``supported_local_geometries``, or isn't ``None``.
        """
        try:
            return self._local_geometry_record[self.gk_code]
        except KeyError:
            try:
                return self._local_geometry_from_global
            except AttributeError:
                return None

    @local_geometry.setter
    def local_geometry(self, value: Optional[LocalGeometry]) -> None:
        # If we have gyrokinetics, set to _local_geometry_record, and otherwise set
        # to _local_geometry_from_global
        if self.gk_code is None:
            self._local_geometry_from_global = value
        else:
            self._local_geometry_record[self.gk_code] = value

    @property
    def local_geometry_type(self) -> Union[str, None]:
        """
        Returns the type of ``self.local_geometry``, expressed as a string. If
        ``self.local_geometry`` does not exist, returns None. Has no setter.

        Returns
        -------
        ``str`` or ``None``
            ``"Miller"`` if ``self.local_geometry`` is of type ``LocalGeometryMiller``.
            ``None`` if ``self.local_geometry`` is ``None``, meaning no local geometry
            is set.

        Raises
        ------
        TypeError
            If ``self.local_geometry`` is set to a non-``LocalGeometry`` type and is
            not ``None``.
        """

        # Determine which kind of LocalGeometry we have
        if isinstance(self.local_geometry, LocalGeometryMillerTurnbull):
            return "MillerTurnbull"
        elif isinstance(self.local_geometry, LocalGeometryMiller):
            return "Miller"
        if isinstance(self.local_geometry, LocalGeometryMXH):
            return "MXH"
        if isinstance(self.local_geometry, LocalGeometryFourierGENE):
            return "FourierGENE"
        if isinstance(self.local_geometry, LocalGeometryFourierCGYRO):
            return "FourierCGYRO"
        elif self.local_geometry is None:
            return None
        else:
            raise TypeError("Pyro._local_geometry is set to an unknown geometry type")

    def switch_local_geometry(self, local_geometry: str, show_fit=False, **kwargs):
        """Converts ``LocalGeometry`` to a different fitting type."""

        # Check if already loaded and if show then switch geometries
        if not isinstance(self.local_geometry, LocalGeometry):
            raise ValueError("Please load local geometry before switching")

        if local_geometry not in self.supported_local_geometries:
            raise ValueError(
                f"Unsupported local geometry type. Got '{local_geometry}', expected "
                f"one of: {self.supported_local_geometries}"
            )

        LocalGeometryT = local_geometry_factory.type(local_geometry)
        self.local_geometry = LocalGeometryT.from_local_geometry(
            self.local_geometry, show_fit=show_fit, **kwargs
        )

        # Change metric_terms is loaded
        if hasattr(self, "metric_terms"):
            self.load_metric_terms()

    # local species property
    @property
    def local_species(self) -> Union[LocalSpecies, None]:
        """
        The ``LocalSpecies`` instance for the current gyrokinetics context, or if there
        is no gyrokinetics context (``self.gk_code`` is ``None``), a ``LocalSpecies``
        instance that isn't assigned to a context.

        Returns
        -------
        LocalSpecies or ``None``
            If ``self.gk_code`` is not ``None``, returns the ``local_species`` for this
            gyrokinetics context if it exists. Otherwise, returns an 'unassigned'
            ``local_species`` if it exists. Failing this, returns None.

        Raises
        ------
        TypeError
            If setting to a value that isn't an instance of ``LocalSpecies`` or
            ``None``.
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
        if not isinstance(value, LocalSpecies):
            raise TypeError("Pyro.local_species.setter: value is not LocalSpecies")
        if self.gk_code is None:
            self._local_species_from_global = value
        else:
            self._local_species_record[self.gk_code] = value

    # numerics property
    @property
    def numerics(self) -> Union[Numerics, None]:
        """
        The ``Numerics`` instance belonging to the current gyrokinetics context. If
        this does not exist, ``None``.

        Returns
        -------
        Numerics or ``None``
            The ``Numerics`` instance for this gyrokinetics context, or ``None`` if this
            does not exist.

        Raises
        ------
        TypeError
           When setting to an instance of a class other than ``Numerics`` or ``None``
        RuntimeError
            When setting without a gyrokinetics context. Ensure ``pyro.gk_code`` is
            set first.
        """
        try:
            return self._numerics_record[self.gk_code]
        except KeyError:
            return None

    @numerics.setter
    def numerics(self, value: Numerics) -> None:
        if value is not None and not isinstance(value, Numerics):
            raise TypeError("Pyro.numerics.setter: value is not Numerics")
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
        Reads a global equilibrium file, sets the property ``eq_file`` to that file
        path, and sets the attribute ``eq`` to an Equilibrium.

        Parameters
        ----------
        eq_file: PathLike
            Path to a global equilibrium file.
        eq_type: ``str``, default ``None``
            String denoting the file type used to create Equilibrium (e.g. GEQDSK,
            TRANSP). If set to ``None``, this will be inferred automatically
        **kwargs
            Args to pass to Equilibrium constructor.

        self.gk_code.add_flags(self, flags)

        Raises
        ------
        Exception
            Various errors can be raised while reading ``eq_file`` and creating an
            Equilibrium.
        """
        self.eq_file = eq_file  # property setter, converts to Path
        self.eq = read_equilibrium(self.eq_file, eq_type, **kwargs)

    @property
    def eq_type(self) -> Union[str, None]:
        """
        The type of global equilibrium (GEQDSK, TRANSP) if it exists, otherwise
        ``None``. Has no setter.

        Returns
        -------
        ``str`` or ``None``
            If a global equilibrium has been loaded, either via ``load_global_eq()`` or
            the constructor, the type of that Equilibrium. If no equilibrium has been
            loaded, ``None``.
        """
        try:
            return self.eq.eq_type
        except AttributeError:
            return None

    def load_global_kinetics(
        self,
        kinetics_file: PathLike,
        kinetics_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Reads a global kinetics file, sets the property ``kinetics_file`` to that file
        path, and sets the attribute ``kinetics`` to a Kinetics.

        Parameters
        ----------
        kinetics_file: PathLike
            Path to a global kinetics file.
        kinetics_type: ``str``, default ``None``
            String denoting the file type used to create Kinetics (e.g. SCENE, JETTO,
            TRANSP, pFile). If set to ``None``, this will be inferred automatically.
        **kwargs
            Args to pass to Kinetics constructor.

        Returns
        -------
        ``None``

        Raises
        ------
        Exception
            Various errors can be raised while reading ``kinetics_file`` and creating a
            Kinetics.
        """
        self.kinetics_file = kinetics_file  # property setter, converts to Path
        try:
            self.kinetics = read_kinetics(self.kinetics_file, kinetics_type, **kwargs)
        except ValueError as exc:
            # Some kinetics readers need an eq_file to work properly.
            if "Please load an Equilibrium." in str(exc) and self.eq is not None:
                self.kinetics = read_kinetics(
                    self.kinetics_file,
                    kinetics_type,
                    eq=self.eq,
                    **kwargs,
                )
            else:
                raise exc

    @property
    def kinetics_type(self) -> Union[str, None]:
        """
        The type of global kinetics (JETTO, SCENE, TRANSP, pFile) if it exists, otherwise
        ``None``. Has no setter.

        Returns
        -------
        ``str`` or ``None``
            If a global kinetics has been loaded, either via ``load_global_kinetics()``
            or the constructor, the type of that Kinetics. If no kinetics has been
            loaded, ``None``.
        """
        try:
            return self.kinetics.kinetics_type
        except AttributeError:
            return None

    def _check_beta_consistency(self):
        """Check that the value of ``beta`` in ``self.numerics`` agrees
        with the physical reference value"""
        beta = getattr(self.numerics, "beta", 0.0)

        # Bail early if there's nothing to check. They'll only both be
        # non-zero if we have all three of a GK sim, geometry, and
        # kinetics. In any other situation, we can't check, so don't bother
        if beta == 0.0 or self.norms.beta == 0.0 or beta is None:
            return

        # No units, so scalar, for example because the user has changed
        # beta and forgotten the units. Assume beta has been given in
        # pyrokinetics normalisation.
        if not hasattr(beta, "units"):
            beta = self.numerics.beta * self.norms.units.beta_ref_ee_B0

        # If they agree, we don't need to say anything
        if np.isclose(beta, self.norms.beta):
            return

        warnings.warn(
            f"Explicitly set value of beta ({beta.to(self.norms)}) is inconsistent with "
            f"value from physical reference values ({self.norms.beta})"
        )

    # Functions for setting local_geometry and local_species from global Equilibrium
    # and Kinetics

    def load_local_geometry(
        self,
        psi_n: float,
        local_geometry: str = "Miller",
        show_fit: bool = False,
        **kwargs,
    ) -> None:
        """
        Uses a global Equilibrium to generate ``local_geometry``. If there is a
        gyrokinetics context, overwrites the local geometry of that context only. If
        there is no gyrokinetics context, saves to an 'unassigned' local geometry.

        Parameters
        ----------
        psi_n: float
            Normalised flux surface on which to calculate local geometry. 0 is the
            center of the equilibrium, 1 is the Last-Closed-Flux-Surface (LCFS).
        local_geometry: str, default "Miller"
            The type of LocalGeometry to create, expressed as a string. Must be in
            ``supported_local_geometries``.
        show_fit: bool, default False
            Flag to show fits to flux surface and poloidal field
        **kwargs
            Args used to build the LocalGeometry.

        Returns
        -------
        ``None``

        Raises
        ------
        RuntimeError
            If a global Equilibrium has not been loaded.
        ValueError
            If psi_n is less than 0 or greater than 1.
        Exception
            A number of errors may be raised when creating a LocalGeometry.
        """
        try:
            if self.eq is None:
                raise AttributeError
        except AttributeError:
            raise RuntimeError(
                "Pyro.load_local_equilibrium: Global equilbrium not found. Please use "
                "load_global_eq() first."
            )

        if psi_n < 0 or psi_n > 1:
            raise ValueError(
                "Pyro.load_local_geometry: psi_n must be between 0 and 1, received "
                f"{psi_n}."
            )

        # Load local geometry
        LocalGeometryT = local_geometry_factory.type(local_geometry)
        local_geometry = LocalGeometryT.from_global_eq(
            self.eq, psi_n=psi_n, norms=self.norms, show_fit=show_fit, **kwargs
        )
        # Set references and normalise
        self.norms.set_bref(local_geometry)
        self.norms.set_lref(local_geometry)
        self.local_geometry = local_geometry.normalise(self.norms)

    def load_metric_terms(
        self, ntheta: Optional[int] = None, theta: Optional[List] = None
    ):
        """
        Uses the local_geometry object to load up the metric tensor terms

        Parameters
        ----------
        ntheta: int default None
            Number of theta points to use when generating the metric tensor terms

        Returns
        -------
        ``None``

         Raises
        ------
        RuntimeError
            If a local_geometry has not been loaded.
        """

        try:
            if self.local_geometry is None:
                raise AttributeError
        except AttributeError:
            raise RuntimeError(
                "Pyro.load_metric_terms: Must have loaded a local geometry first. "
                "Use function load_local_geometry."
            )

        if ntheta is None and theta is None:
            ntheta = len(self.local_geometry.theta)

        self.metric_terms = MetricTerms(self.local_geometry, ntheta=ntheta, theta=theta)

    def load_local_species(self, psi_n: float, a_minor: Optional[float] = None) -> None:
        """
        Uses a global Kinetics to generate ``local_species``. If there is a
        gyrokinetics context, overwrites the local species of that context only. If
        there is no gyrokinetics context, saves to an 'unassigned' local species.

        Parameters
        ----------
        psi_n: float
            Normalised flux surface on which to calculate local geometry. 0 is the
            center of the equilibrium, 1 is the Last-Closed-Flux-Surface (LCFS).
        a_minor: float, default None
            The minor radius of the global Equilibrium. If set to ``None``, this value
            is obtained from ``self.eq``. It is recommended to only set this if there
            is no global Equilibrium.

        Returns
        -------
        ``None``

        Raises
        ------
        RuntimeError
            If a global Kinetics has not been loaded. Raised if a_minor is ``None``, but
            no global Equilibrium has been loaded.
        ValueError
            If psi_n is less than 0 or greater than 1.
        Exception
            A number of errors may be raised when creating a LocalSpecies.
        """
        try:
            if self.kinetics is None:
                raise AttributeError
        except AttributeError:
            raise RuntimeError(
                "Pyro.load_local_species: Must have read global kinetics first. "
                "Use function load_global_kinetics."
            )

        if psi_n < 0 or psi_n > 1:
            raise ValueError(
                "Pyro.load_local_species: psi_n must be between 0 and 1, received "
                f"{psi_n}."
            )

        if a_minor is not None:
            if not isinstance(a_minor, PyroQuantity):
                raise ValueError("a_minor must be specified with units")
            self.norms.set_lref(minor_radius=a_minor)

        self.norms.set_kinetic_references(self.kinetics, psi_n=psi_n)

        local_species = LocalSpecies()
        local_species.from_kinetics(self.kinetics, psi_n=psi_n, norm=self.norms)
        self.local_species = local_species

    def load_local(
        self, psi_n: float, local_geometry: str = "Miller", show_fit: bool = False
    ) -> None:
        """
        Combines calls to ``load_local_geometry()`` and ``load_local_species()``

        Parameters
        ----------
        psi_n: float
            Normalised flux surface on which to calculate local geometry. 0 is the
            center of the equilibrium, 1 is the Last-Closed-Flux-Surface (LCFS).
        local_geometry: str, default "Miller"
            The type of LocalGeometry to create, expressed as a string. Must be in
            ``supported_local_geometries``.
        show_fit: bool
            Show fit of LocalGeometry, default is False
        Returns
        -------
        ``None``

        Raises
        ------
        Exception
            See exceptions for ``load_local_geometry()`` and ``load_local_species()``.
        """
        self.load_local_geometry(
            psi_n, local_geometry=local_geometry, show_fit=show_fit
        )
        self.load_local_species(psi_n)

        self._load_local_geometry_species_dependency()

    def _load_local_geometry_species_dependency(
        self, set_rhoref=True, set_beta=True, set_gamma_exb=True
    ):
        """
        Load data that requires both LocalGeometry and LocalSpecies to be present

        Loads in Larmor radius rhoref, ensures beta is taken from Numerics and
        sets ExB shear

        Parameters
        ----------
        set_rhoref: bool, default True
            Sets rhoref if True
        set_beta: bool, default True
            Sets beta=None if True
        set_gamma_exb: bool, default True
            Sets gamma_exb

        Returns
        -------
        ``None``

        Raises
        ------
        ValueError
            If local_species and local_geometry are not loaded then a ValueError
        is raised

        """

        if self.local_geometry is None or self.local_species is None:
            raise ValueError(
                "Please load both local_species and local_geometry before calling _load_local_geometry_species_dependency"
            )

        if set_rhoref:
            self.norms.set_rhoref(local_geometry=self.local_geometry)

        # If we have both kinetics and eq file we should set beta/gamma_exb from there
        if self.numerics and set_beta:
            self.numerics.beta = None

            self._check_beta_consistency()

        if self.numerics and set_gamma_exb:
            self.numerics.gamma_exb = (
                -self.local_geometry.rho
                / self.local_geometry.q
                * self.local_species.domega_drho
            ).to(self.norms.vref / self.norms.lref)

        self._local_geometry_species_dependency = True

    def set_reference_values(
        self,
        tref_electron=None,
        nref_electron=None,
        bref_B0=None,
        lref_minor_radius=None,
        lref_major_radius=None,
    ):
        """
        Manually set the reference values used in normalisations

        Parameters
        ----------
        tref_electron: [eV] pint.Quantity
            Electron temperature
        nref_electron: [meter**-3] pint.Quantity
            Electron density
        bref_b0: [tesla] pint.Quantity
            Toroidal magnetic field at centre of flux surface
        lref_major_radius: [meter] pint.Quantity
            Minor radius of last closed flux surface

        Returns
        -------
        ``None``
        """

        self.norms.set_all_references(
            self,
            tref_electron=tref_electron,
            nref_electron=nref_electron,
            bref_B0=bref_B0,
            lref_minor_radius=lref_minor_radius,
            lref_major_radius=lref_major_radius,
        )

    # Utility for copying Pyro object

    def __deepcopy__(self, memodict):
        """
        Create a new Pyro, recursively copying all structures.

        Returns
        -------
        Pyro
            Deep copy of self.
        """

        new_pyro = Pyro()

        for key, value in self.__dict__.items():
            setattr(new_pyro, key, copy.deepcopy(value))

        return new_pyro

    # Add properties that allow direct access to GKInput.data
    # TODO This feels dangerous... could use a refactor
    # TODO Not sure how to automate generation of these when new gk_codes are added
    @property
    def gs2_input(self) -> Union[f90nml.Namelist, None]:
        """
        Return the raw data from the ``GKInput`` corresponding to the GS2 context. If it
        doesn't exist, returns ``None``. Has no setter.

        Returns
        -------
        f90nml.Namelist or ``None``
            Fortran namelist object holding input data for the GS2 context if it exists,
            otherwise ``None``.
        """
        try:
            return self._gk_input_record["GS2"].data
        except KeyError:
            return None

    @property
    def cgyro_input(self) -> Union[Dict[str, Any], None]:
        """
        Return the raw data from the ``GKInput`` corresponding to the CGYRO context. If
        it doesn't exist, returns ``None``. Has no setter.

        Returns
        -------
        Dict[str,Any] or ``None``
            Dict holding input data for the CGYRO context if it exists, otherwise
            ``None``.
        """
        try:
            return self._gk_input_record["CGYRO"].data
        except KeyError:
            return None

    @property
    def gene_input(self) -> Union[f90nml.Namelist, None]:
        """
        Return the raw data from the ``GKInput`` corresponding to the GENE context. If
        it doesn't exist, returns ``None``. Has no setter.

        Returns
        -------
        f90nml.Namelist or ``None``
            Fortran namelist holding input data for the GENE context if it exists,
            otherwise ``None``.
        """
        try:
            return self._gk_input_record["GENE"].data
        except KeyError:
            return None
