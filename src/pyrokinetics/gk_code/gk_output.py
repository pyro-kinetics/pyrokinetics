from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generator,
    Iterable,
    List,
    NoReturn,
    Optional,
    Tuple,
)

import numpy as np
import pint
from numpy.typing import ArrayLike

from ..dataset_wrapper import DatasetWrapper
from ..file_utils import ReadableFromFile
from ..normalisation import ConventionNormalisation, SimulationNormalisation
from ..typing import PathLike
from ..units import ureg as units

if TYPE_CHECKING:
    import xarray as xr


@dataclasses.dataclass
class GKOutputArgs:
    """
    Utility base dataclass used to pass quantities to ``GKOutput``. Derived classes
    include ``Coords``, ``Fields``, ``Fluxes``, etc. This class contains features such
    as automatic unit conversion and a dict-like interface to quantities.

    Derived classes should define an ``InitVar[Tuple[str, ...]]`` called ``dims``,
    which sets the dimensionality of each quantity, e.g. ``("kx", "ky", "time")``.
    This should be set in ``__post_init__``, along with a call to
    ``_set_and_check_dims``. This is to work around the pre-Python 3.10 dataclass
    issues regarding keyword arguments in base classes
    """

    @property
    def names(self) -> Tuple[str, ...]:
        """Names of all quantities held by this dataclass"""
        return tuple(x.name for x in dataclasses.fields(self))

    @abstractmethod
    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        """Return the units for each quantity"""
        pass

    def __getitem__(self, key: str) -> Any:
        """Look up quantities with dict-like inteface"""
        if key not in self.names:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, val: Any) -> None:
        """
        Set quantities with dict-like inteface.
        Warning: No validity checking is performed.
        """
        if key not in self.names:
            raise KeyError(key)
        setattr(self, key, val)

    @property
    def coords(self) -> Tuple[str, ...]:
        """
        Tuple containing the names of each supplied field (those that aren't ``None``).
        """
        return tuple(k for k in self.names if self[k] is not None)

    def __iter__(self) -> Generator[str, None, None]:
        """Iterate over quantity names. Skips ``None`` quantities."""
        return iter(self.coords)

    def values(self) -> Generator[Any, None, None]:
        """Dict-like values iteration"""
        try:
            it = iter(self)
            while True:
                yield self[next(it)]
        except StopIteration:
            return

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Dict-like items iteration"""
        return zip(iter(self), self.values())

    #: List of quantities that have normalised units, e.g. 'lref'.
    #: Should be overridden in derived classes.
    _has_normalised_units: ClassVar[Tuple[str, ...]] = ()

    #: List of quantities that have physical units, e.g. 'radians'.
    #: Should be overridden in derived classes.
    _has_physical_units: ClassVar[Tuple[str, ...]] = ()

    def with_units(self, c: ConventionNormalisation):
        """
        Apply units to each quantity in turn and return a new ``Coords``.
        If units are already applied, renormalises according to the convention supplied.
        """
        kwargs = {}
        for key, val in self.items():
            if val is None:
                kwargs[key] = None
                continue
            if key in self._has_normalised_units:
                if hasattr(val, "units"):
                    kwargs[key] = val.to(c)
                else:
                    kwargs[key] = val * self.units(key, c)
                continue
            if key in self._has_physical_units:
                if hasattr(val, "units"):
                    kwargs[key] = val.to(self.units(key, c))
                else:
                    kwargs[key] = val * self.units(key, c)
                continue
            # Pass everything else through
            kwargs[key] = val
        # Pass through the pseudo-field 'dims'
        if hasattr(self, "dims"):
            kwargs["dims"] = self.dims
        return self.__class__(**kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of quantities. Raises error if all are None.
        Should be overridden in ``Coords``, where this function makes no sense.
        """
        for _, val in self.items():
            if val is not None:
                return np.shape(val)
        raise ValueError("Fields contains no data")

    def _set_and_check_dims(self, dims) -> None:
        """Set ``dims``, perform checks on the values assigned."""
        self.dims = dims
        for key, val in self.items():
            if val is not None and np.ndim(val) != len(self.dims):
                raise ValueError(f"Quantity '{key}' has incorrect number of dims")


@dataclasses.dataclass
class Coords(GKOutputArgs):
    """Utility dataclass type used to pass coordinates to ``GKOutput``"""

    #: 1D grid of radial wave-numbers used in the simulation
    #: Units of [rhoref ** -1]
    kx: ArrayLike

    #: 1D grid of bi-normal wave-numbers used in the simulation
    #: Units of [rhoref ** -1]
    ky: ArrayLike

    #: 1D grid of time of the simulation output
    #: Units of [lref / vref]
    time: ArrayLike

    #: List of species names in the simulation
    species: Iterable[str]

    #: 1D grid of theta used in the simulation
    #: Units of [radians]
    theta: Optional[ArrayLike] = None

    #: TODO document
    mode: Optional[ArrayLike] = None

    #: 1D grid of the energy in the simulation
    #: Units of [dimensionless]
    energy: Optional[ArrayLike] = None

    #: 1D grid of the pitch angle in the simulation
    #: Units of [dimensionless]
    pitch: Optional[ArrayLike] = None

    #: List of fields. Normally this information is obtained from the ``Fields`` class,
    #: but there are edge cases in which no ``Fields`` can be supplied but a fields
    #: coordinate must be defined.
    field: Optional[Iterable[str]] = None

    _has_physical_units: ClassVar[Tuple[str, ...]] = (
        "theta",
        "mode",
        "energy",
        "pitch",
    )

    _has_normalised_units: ClassVar[Tuple[str, ...]] = ("kx", "ky", "time")

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        if name not in self.names:
            raise ValueError(f"The coord '{name}' is not recognised")
        if name in ("kx", "ky"):
            return c.rhoref**-1
        if name == "time":
            return c.lref / c.vref
        if name == "theta":
            return units.radians
        return units.dimensionless

    @property
    def shape(self) -> NoReturn:
        raise RuntimeError("Coords does not implement 'shape'")


@dataclasses.dataclass
class Fields(GKOutputArgs):
    """Utility dataclass type used to pass field data to ``GKOutput``."""

    #: :math:`\phi`: the electrostatic potential.
    #: Units of ``[tref * rhoref / (qref * lref)]``
    phi: Optional[ArrayLike] = None

    #: :math:`A_\parallel`: Parallel component of the magnetic vector potential.
    #: Units of ``[bref * rhoref**2 / lref]``.
    apar: Optional[ArrayLike] = None

    #: :math:`B_\parallel`: Parallel component of the magnetic flux density.
    #: Units of ``[bref * rhoref / lref]``.
    bpar: Optional[ArrayLike] = None

    _has_normalised_units: ClassVar[Tuple[str, ...]] = ("phi", "apar", "bpar")

    #: The dimensionality of the fields.
    #: Each field should have the same number of dimensions.
    dims: dataclasses.InitVar[Tuple[str, ...]] = ("theta", "kx", "ky", "t")

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        """Return units associated with each field for a given convention"""
        if name == "phi":
            return c.tref * c.rhoref / (c.qref * c.lref)
        elif name == "apar":
            return c.bref * c.rhoref**2 / c.lref
        elif name == "bpar":
            return c.bref * c.rhoref / c.lref
        else:
            raise ValueError(f"Field name '{name}' not recognised")

    def __post_init__(self, dims):
        self._set_and_check_dims(dims)


@dataclasses.dataclass
class Fluxes(GKOutputArgs):
    """Utility dataclass type used to pass fluxes to ``GKOutput``."""

    #: Units of ``[nref * vref * (rhoref / lref)**2]``.
    particle: Optional[ArrayLike] = None

    #: Units of ``[nref * vref * tref * (rhoref / lref)**2]``.
    heat: Optional[ArrayLike] = None

    #: units of ``[nref * lref * tref * (rhoref / lref)**2]``.
    momentum: Optional[ArrayLike] = None

    _has_normalised_units: ClassVar[Tuple[str, ...]] = ("particle", "heat", "momentum")

    #: The dimensionality of the fluxes.
    #: Each array should have the same dimensionality.
    dims: dataclasses.InitVar[Tuple[str, ...]] = ("field", "species", "kx", "ky", "t")

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        """Return units associated with each flux for a given convention"""
        if name == "particle":
            return c.nref * c.vref * (c.rhoref / c.lref) ** 2
        elif name == "momentum":
            return c.nref * c.lref * c.tref * (c.rhoref / c.lref) ** 2
        elif name == "heat":
            return c.nref * c.vref * c.tref * (c.rhoref / c.lref) ** 2
        else:
            raise ValueError(f"Flux name '{name}' not recognised.")

    def __post_init__(self, dims):
        self._set_and_check_dims(dims)


@dataclasses.dataclass
class Moments(GKOutputArgs):
    """Utility dataclass type used to pass moments to ``GKOutput``."""

    #: Units of ``[nref * rhoref / lref)]``
    density: Optional[ArrayLike] = None

    #: Units of ``[tref * rhoref / lref]``
    temperature: Optional[ArrayLike] = None

    #: Units of ``[vref * rhoref / lref]``
    velocity: Optional[ArrayLike] = None

    _has_normalised_units: ClassVar[Tuple[str, ...]] = (
        "density",
        "temperature",
        "velocity",
    )

    #: The dimensionality of the moments.
    #: Each array should have the same dimensionality.
    dims: dataclasses.InitVar[Tuple[str, ...]] = ("theta", "kx", "species", "ky", "t")

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        """Return units associated with each moment for a given convention"""
        if name == "density":
            return c.nref * c.rhoref / c.lref
        elif name == "temperature":
            return c.tref * c.rhoref / c.lref
        elif name == "velocity":
            return c.vref * c.rhoref / c.lref
        else:
            raise ValueError(f"Moment name '{name}' not recognised.")

    def __post_init__(self, dims):
        self._set_and_check_dims(dims)


@dataclasses.dataclass
class Eigenvalues(GKOutputArgs):
    """
    Utility dataclass type used to pass eigenvalues to ``GKOutput``. Unlike the classes
    ``Fields``, ``Fluxes``, and ``Moments``, all entries to ``Eigenvalues`` are
    non-optional.
    """

    #: Units of ``[lref / vref]``.
    growth_rate: ArrayLike

    #: Units of ``[lref / vref]``.
    mode_frequency: ArrayLike

    _has_normalised_units: ClassVar[Tuple[str, ...]] = ("growth_rate", "mode_frequency")

    #: The dimensionality of the eigenvalues
    #: Each array should have the same dimensionality.
    dims: dataclasses.InitVar[Tuple[str, ...]] = ("kx", "ky", "time")

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        """Return units for a given convention"""
        return c.lref / c.vref

    def __post_init__(self, dims):
        self._set_and_check_dims(dims)


@dataclasses.dataclass
class Eigenfunctions(GKOutputArgs):
    """Utility dataclass type used to pass eigenfunctions to ``GKOutput``."""

    #: Eigenfunction data to pass to ``GKOutput``
    eigenfunctions: ArrayLike

    #: The dimensionality of the eigenfunctions. Should match ``data``.
    dims: dataclasses.InitVar[Tuple[str, ...]] = ("field", "theta", "kx", "ky", "time")

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        """Return units for a given convention"""
        return units.dimensionless

    def __post_init__(self, dims):
        self._set_and_check_dims(dims)


# TODO define Diagnostics stuff on this class. could use accessors
#      https://docs.xarray.dev/en/stable/internals/extending-xarray.html


class GKOutput(DatasetWrapper, ReadableFromFile):
    """
    Contains the output data from gyrokinetics codes. Converts the results of each code
    to a standard set of normalisation conventions, which allows for easier cross-code
    comparisons.

    Users are not expected to initialise ``GKOutput`` objects directly,
    and in most cases should instead make use of the function ``read_gk_output``.

    The inputs to ``GKOutput`` should be given "physical units", as defined on the
    `normalisation` page, appropriate to the code that generated the output
    data. If inputs are not given units, it is assumed they are already compliant with
    the Pyrokinetics standards.

    Parameters
    ----------
    coords
        Dataclass specifying the coordinates of the simulation
    norm
        The normalisation scheme of the simulation.
    fields
        Dataclass specifying the fields in the simulation
    fluxes
        Dataclass specifying the fluxes in the simulation
    moments
        Dataclass specifying the moments in the simulation
    eigenvalues
        Dataclass specifying the eigenvalues in the simulation. Should only be supplied
        for linear runs. If not provided, will be set from field data.
    eigenfunctions
        Dataclass specifying the eigenfunctions in the simulation Should only be
        supplied for linear runs. If not provided, will be set from field data.
    linear
        Set True for linear gyrokinetics runs, False for nonlinear runs.
    normalise_flux_moment
        TODO write docs
    gk_code:
        The gyrokinetics code that generated the results.
    input_file
        The input file used to generate the results.

    Attributes
    ----------

    data: xarray.Dataset
        The internal representation of the ``GKOutput`` object. The functions
        ``__getattr__`` and ``__getitem__`` redirect most attribute/indexing lookups
        here, but the Dataset itself may be accessed directly by the user if they wish
        to perform more complex manipulations.
    linear: bool
        ``True`` if the run is linear, ``False`` if the run is nonlinear
    gk_code: str
        The gyrokinetics code that generated the data.
    input_file: str
        Gyrokinetics input file expressed as a string.
    norm: SimulationNormalisation
        The normalisation scheme used for the data.
    """

    def __init__(
        self,
        *,  # args are keyword only
        coords: Coords,
        norm: SimulationNormalisation,
        fields: Optional[Fields] = None,
        fluxes: Optional[Fluxes] = None,
        moments: Optional[Moments] = None,
        eigenvalues: Optional[Eigenvalues] = None,
        eigenfunctions: Optional[Eigenfunctions] = None,
        linear: bool = True,
        normalise_flux_moment: bool = False,
        gk_code: Optional[str] = None,
        input_file: Optional[str] = None,
    ):
        self.norm = norm
        convention = norm.pyrokinetics

        # Renormalise inputs
        coords = coords.with_units(convention)

        if fields is not None:
            fields = fields.with_units(convention)

        if fluxes is not None:
            fluxes = fluxes.with_units(convention)

        if moments is not None:
            moments = moments.with_units(convention)

        # Get coords to hand over to underlying Dataset
        def make_var(dim, val, desc):
            if val is None:
                return None
            if hasattr(val, "units"):
                return (dim, val.m, {"units": str(val.u), "long_name": desc})
            return (dim, val, {"units": None, "long_name": desc})

        dataset_coords = {
            "time": make_var("time", coords.time, "Time"),
            "kx": make_var("kx", coords.kx, "Radial wavenumber"),
            "ky": make_var("ky", coords.ky, "Bi-normal wavenumber"),
            "theta": make_var("theta", coords.theta, "Angle"),
            "energy": make_var("energy", coords.energy, "Energy"),
            "pitch": make_var("pitch", coords.pitch, "Pitch angle"),
            "mode": make_var("mode", coords.mode, "Mode"),
            "species": make_var("species", coords.species, "Species"),
        }

        # Add field, flux and moment coords
        if fields is not None:
            dataset_coords["field"] = make_var(
                "field", np.array(fields.coords), "Field"
            )
        if fluxes is not None:
            dataset_coords["flux"] = make_var("flux", np.array(fluxes.coords), "Flux")
        if moments is not None:
            dataset_coords["moment"] = make_var(
                "moment", np.array(moments.coords), "Moment"
            )

        # Edge case where field coord is set but not fields
        if fields is None and coords.field is not None:
            dataset_coords["field"] = make_var("field", coords.field, "Field")

        # Remove None entries
        dataset_coords = {k: v for k, v in dataset_coords.items() if v is not None}

        # Renormalise fields, fluxes, moments

        # Normalise QL fluxes and moments if linear and needed
        if fields is not None and linear and normalise_flux_moment:
            if fluxes is not None:
                fluxes = self._normalise_to_fields(fields, coords.theta, fluxes)
            if moments is not None:
                moments = self._normalise_to_fields(fields, coords.theta, moments)

        # Normalise fields to GKDB standard
        if fields is not None and "time" in fields.dims and linear:
            fields = self._normalise_linear_fields(fields, coords.theta.m)

        # Set up data vars to hand over to underlying Dataset
        data_vars = {}

        if fields is not None:
            field_desc = {
                "phi": "Electrostatic potential",
                "apar": "Parallel magnetic vector potential",
                "bpar": "Parallel magnetic flux density",
            }
            for key in fields.coords:
                data_vars[key] = make_var(fields.dims, fields[key], field_desc[key])

        if moments is not None:
            moment_desc = {
                "density": "Density fluctuations",
                "temperature": "Temperature fluctuations",
                "velocity": "Velocity fluctuations",
            }
            for key in moments.coords:
                data_vars[key] = make_var(moments.dims, moments[key], moment_desc[key])

        if fluxes is not None:
            flux_desc = {
                "particle": "Particle flux",
                "heat": "Heat flux",
                "momentum": "Momentum flux",
            }
            for key in fluxes.coords:
                data_vars[key] = make_var(fluxes.dims, fluxes[key], flux_desc[key])

        # Add eigenvalues. If not provided, try to generate from fields
        if eigenvalues is None and fields is not None and linear:
            eigenvalues = self._eigenvalues_from_fields(
                fields, coords.theta.magnitude, coords.time.magnitude
            )

        if eigenvalues is not None:
            eigenvalues = eigenvalues.with_units(convention)

            data_vars["growth_rate"] = make_var(
                eigenvalues.dims, eigenvalues.growth_rate, "Growth rate"
            )
            data_vars["mode_frequency"] = make_var(
                eigenvalues.dims, eigenvalues.mode_frequency, "Mode frequency"
            )
            data_vars["eigenvalues"] = make_var(
                eigenvalues.dims,
                eigenvalues.mode_frequency + 1.0j * eigenvalues.growth_rate,
                "Eigenvalues",
            )

        # Add eigenfunctions. If not provided, try to generate from fields
        if eigenfunctions is None and fields is not None and linear:
            eigenfunctions = self._eigenfunctions_from_fields(fields, coords.theta.m)

        if eigenfunctions is not None:
            data_vars["eigenfunctions"] = make_var(
                eigenfunctions.dims,
                eigenfunctions.eigenfunctions,
                "Eigenfunctions",
            )

        # Set up attrs to hand over to underlying dataset
        attrs = {
            "linear": linear,
            "gk_code": gk_code if gk_code is not None else "",
            "input_file": input_file if input_file is not None else "",
        }

        # Hand over to underlying dataset
        super().__init__(data_vars=data_vars, coords=dataset_coords, attrs=attrs)

        # Calculate growth_rate_tolerance with default inputs
        if eigenvalues is not None and "time" in eigenvalues.dims and linear:
            self.data["growth_rate_tolerance"] = self.get_growth_rate_tolerance()

    def field(self, name: str) -> xr.DataArray:
        if name not in Fields.names:
            raise ValueError(
                f"'name' should be one of {', '.join(Fields.names)}. "
                f"Received '{name}'."
            )
        if name not in self.data_vars:
            raise ValueError(f"GKOutput does not contain the field '{name}'")
        return self.data_vars[name]

    def flux(self, name: str) -> xr.DataArray:
        if name not in Fluxes.names:
            raise ValueError(
                f"'name' should be one of {', '.join(Fluxes.names)}. "
                f"Received '{name}'"
            )
        if name not in self.data_vars:
            raise ValueError(f"GKOutput does not contain the flux '{name}'")
        return self.data_vars[name]

    def moment(self, name: str) -> xr.DataArray:
        if name not in Moments.names:
            raise ValueError(
                f"'name' should be one of {', '.join(Moments.names)}. "
                f"Received '{name}'"
            )
        if name not in self.data_vars:
            raise ValueError(f"GKOutput does not contain the moment '{name}'")
        return self.data_vars[name]

    def get_growth_rate_tolerance(self, time_range: float = 0.8) -> float:
        """
        Given a pyrokinetics output dataset with eigenvalues determined, calculate the
        growth rate tolerance. This is calculated starting at the time given by
        time_range * max_time.
        """

        if "growth_rate" not in self.data_vars:
            raise ValueError(
                "GKOutput does not have 'growth rate'. Only results associated with "
                "linear gyrokinetics runs will have this."
            )
        growth_rate = self["growth_rate"]
        final_growth_rate = growth_rate.isel(time=-1)

        difference = ((growth_rate - final_growth_rate) / final_growth_rate) ** 2

        final_time = self["time"].isel(time=-1)

        # Average over the end of the simulation, starting at time_range*final_time
        difference = difference.where(
            difference["time"] > time_range * final_time, drop=True
        )

        tolerance = np.sqrt(
            difference.integrate("time")
            / (difference["time"].isel(time=-1) - difference["time"].isel(time=0))
        )

        return tolerance

    @staticmethod
    def _eigenvalues_from_fields(
        fields: Fields, theta: ArrayLike, time: ArrayLike
    ) -> Eigenvalues:
        """
        Call during __init__ after converting to pyro normalisations
        """
        # field dims are (theta, kx, ky, time)
        # FIXME field dims are now variable!
        shape = fields.shape
        sum_fields = np.zeros(shape, dtype=complex)
        square_fields = np.zeros(shape)
        for field in fields.values():
            sum_fields += field.magnitude
            square_fields += np.abs(field.magnitude) ** 2

        # Integrate over theta
        field_amplitude = np.trapz(square_fields, theta, axis=0) ** 0.5
        # Differentiate with respect to time
        growth_rate = np.gradient(np.log(field_amplitude), time, axis=-1)

        field_angle = np.angle(np.trapz(sum_fields, theta, axis=0))

        # Change angle by 2pi for every rotation so gradient is easier to calculate
        pi_change = np.cumsum(
            np.where(
                field_angle[:, :, :-1] * field_angle[:, :, 1:] < -np.pi,
                -np.sign(field_angle[:, :, 1:]),
                0,
            ),
            axis=-1,
        )
        field_angle[:, :, 1:] += 2 * np.pi * pi_change

        mode_frequency = -np.gradient(field_angle, time, axis=-1)

        return Eigenvalues(
            growth_rate=growth_rate,
            mode_frequency=mode_frequency,
        )

    @staticmethod
    def _eigenfunctions_from_fields(fields: Fields, theta: ArrayLike) -> Eigenfunctions:
        # field coords are (theta, kx, ky, time)
        square_fields = np.zeros(fields.shape)
        for field in fields.values():
            square_fields += np.abs(field.magnitude) ** 2
        field_amplitude = np.sqrt(np.trapz(square_fields, theta, axis=0)) / (2 * np.pi)
        eigenfunctions = np.zeros(
            (len(fields.coords),) + square_fields.shape, dtype=complex
        )
        for ifield, field in enumerate(fields.values()):
            eigenfunctions[ifield] = field.magnitude / field_amplitude
        # TODO are these dims correct?

        eigenfunctions *= units.dimensionless

        return Eigenfunctions(eigenfunctions, dims=("field",) + fields.dims)

    @staticmethod
    def _get_field_amplitude(fields: Fields, theta):
        field_squared = 0.0
        for field in fields.values():
            field_squared += np.abs(field.m) ** 2

        amplitude = np.sqrt(np.trapz(field_squared, theta, axis=0) / 2 * np.pi)

        return amplitude

    def _normalise_linear_fields(self, fields: Fields, theta) -> Fields:
        """
        Normalise fields as done in GKDB manual sec 5.5.3->5.5.5

        Parameters
        ----------
        fields

        Returns
        -------
        fields
        """

        amplitude = self._get_field_amplitude(fields, theta)[:, :, -1]

        if "phi" in fields.coords:
            phase_field = "phi"
        else:
            phase_field = fields.coords[0]

        phi = fields[phase_field][:, :, :, -1]
        theta_star = np.argmax(abs(phi), axis=0)

        phi_theta_star = phi[theta_star][-1, -1, :, :]
        phase = np.abs(phi_theta_star) / phi_theta_star

        # Add theta and time dimension back in
        normalising_factor = np.expand_dims(phase / amplitude, axis=(0, -1))

        for field_name in fields.coords:
            fields[field_name] *= normalising_factor

        return fields

    def _normalise_to_fields(self, fields: Fields, theta, outputs):
        """
        Normalise output (moments/fluxes) to fields to obtain quasi-linear value
        Only valid for linear simulations

        Parameters
        ----------
        fields : FieldDict
            Field data used to normalise output
        theta: ArrayLike
            theta grid over which fields are integrated
        outputs: MomentDict or FluxDict
            Output to renormalise

        Returns
        -------
        outputs: MomentDict or FluxDict
            Re-normalised outputs
        """

        amplitude = self._get_field_amplitude(fields, theta)

        for output_name in outputs.coords:
            outputs[output_name] /= amplitude**2

        return outputs

    def to_netcdf(self, *args, **kwargs) -> None:
        """Writes self.data to disk. Forwards all args to xarray.Dataset.to_netcdf."""
        import pint_xarray  # noqa
        import xarray as xr

        data = self.data.expand_dims("ReIm", axis=-1)  # Add ReIm axis at the end
        data = xr.concat([data.real, data.imag], dim="ReIm")

        data.pint.dequantify().to_netcdf(*args, **kwargs)

    def to(self, norms: ConventionNormalisation):
        """

        Parameters
        ----------
        norms : ConventionNormalisation
            Normalisation convention to convert to

        Returns
        -------
        GKOutput with units from norms
        """
        for data_var in self.data_vars:
            self.data[data_var].data = self.data[data_var].data.to(norms)

        # Coordinates with units not supported in xarray need to manually change
        new_coords = {}
        for coord in self.coords:
            if hasattr(self.data[coord], "units"):
                new_coord = (self.data[coord].data * self.data[coord].units).to(norms)
                new_coords[coord] = (
                    coord,
                    new_coord.m,
                    {"units": new_coord.units, "long_name": self.data[coord].long_name},
                )

        self.data = self.data.assign_coords(coords=new_coords)


def read_gk_output(
    path: PathLike,
    norm: SimulationNormalisation,
    load_fields=True,
    load_fluxes=True,
    load_moments=False,
    gk_type: Optional[str] = None,
    **kwargs,
):
    r"""
    Read gyrokinetics output file(s) from disk, returning an `GKOutput` instance.

    Parameters
    ----------
    path: PathLike
        Location of the file(s) on disk.
    norm: SimulationNormalisation
        The normalisation scheme of the simulation that produced the data. Usually
        generated by a Pyro object.
    gk_type: Optional[str]
        String specifying the type of gyrokinetics files. If unset, the file type
        will be inferred automatically. Specifying the file type may improve
        performance.
    **kwargs:
        Keyword arguments forwarded to the gyrokinetics output file reader.

    Raises
    ------
    ValueError
        If path doesn't exist or isn't a file.
    """
    return GKOutput.from_file(
        path,
        file_type=gk_type,
        norm=norm,
        load_fields=load_fields,
        load_fluxes=load_fluxes,
        load_moments=load_moments,
        **kwargs,
    )


def supported_gk_output_types() -> List[str]:
    """
    Returns a list of all registered GKOutput file types. These file types are
    readable by ``from_file``.
    """
    return GKOutput.supported_file_types()
