import dataclasses
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, Optional, Type, TypedDict, List

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from ..readers import create_reader_factory, Reader
from ..dataset_wrapper import DatasetWrapper
from ..units import ureg as units
from ..normalisation import SimulationNormalisation, ConventionNormalisation
from ..typing import PathLike


class FieldDict(TypedDict, total=False):
    """
    The dict used to pass field data into a GKOutput. No keys are strictly required.
    """

    #: Electrostatic potential. Units of ``[tref * rhoref / (qref * lref)]``
    phi: ArrayLike

    #: Parallel component of the magnetic vector potential. Units of
    #: ``[bref * rhoref**2 / lref]``.
    apar: ArrayLike

    #: Parallel component of the magnetic flux density. Units of
    #: ``[bref * rhoref / lref]``.
    bpar: ArrayLike


@dataclasses.dataclass(frozen=True)
class FluxDict:
    """
    Utility type used to identify the type of a flux array. Used to index the dict of
    flux arrays passed to GKOutput.
    """

    #: The type of flux. Possible moments, and their corresponding units, are:

    #: - ``"particle"``, units of ``[nref * vref * (rhoref / lref)**2]``.
    particle: ArrayLike

    #: - ``"heat"``, units of ``[nref * vref * tref * (rhoref / lref)**2]
    heat: ArrayLike

    #: - ``"momentum"``. units of ``[nref * lref * tref * (rhoref / lref)**2]``.
    momentum: ArrayLike

    def __post_init__(self):
        """
        Perform checks on the values assigned.
        """
        for name, var in vars(self).items():
            if not isinstance(var, str):
                raise TypeError(f"{name} must be of type str")
        possible_moments = ("particle", "momentum", "heat")
        if self.moment not in possible_moments:
            err_msg = dedent(
                f"""
                moment must be one of '{"', '".join(possible_moments)}', but received
                '{self.moment}'
                """
            )
            raise ValueError(err_msg.replace("\n", " "))
        possible_fields = ("phi", "apar", "bpar")
        if self.field not in possible_fields:
            err_msg = dedent(
                f"""
                field must be one of '{"', '".join(possible_fields)}', but received
                '{self.field}'
                """
            )
            raise ValueError(err_msg.replace("\n", " "))

    def key(self) -> str:
        """
        Generates a key which can label the flux in a Dataset
        """
        return f"{self.field}_{self.species}_{self.moment}_flux"


def get_eigenvalues_units(c: ConventionNormalisation):
    return {
        "eigenvalues": c.lref / c.vref,
        "growth_rate": c.lref / c.vref,
        "mode_frequency": c.lref / c.vref,
    }

def get_eigenfunctions_units(c: ConventionNormalisation):
    return {
        "eigenfunctions": units.dimensionless,
    }

def get_flux_units(c: ConventionNormalisation):
    return {
        "particle": c.nref * c.vref * (c.rhoref / c.lref) ** 2,
        "momentum": c.nref * c.lref * c.tref * (c.rhoref / c.lref) ** 2,
        "heat": c.nref * c.vref * c.tref * (c.rhoref / c.lref) ** 2,
    }


def get_field_units(c: ConventionNormalisation):
    return {
        "phi": c.tref * c.rhoref / (c.qref * c.lref),
        "apar": c.bref * c.rhoref**2 / c.lref,
        "bpar": c.bref * c.rhoref / c.lref,
    }


def get_coord_units(c: ConventionNormalisation):
    return {
        "ky": c.rhoref**-1,
        "kx": c.rhoref**-1,
        "time": c.lref / c.vref,
        "theta": units.radians,
        "energy": units.dimensionless,
        "pitch": units.dimensionless,
        "field": units.dimensionless,
        "moment": units.dimensionless,
        "species": units.dimensionless,
        "mode": units.dimensionless
    }


# TODO define Diagnostics stuff on this class. could use accessors
#      https://docs.xarray.dev/en/stable/internals/extending-xarray.html


class GKOutput(DatasetWrapper):
    """
    Contains the output data from gyrokinetics codes. Converts the results of each code
    to a standard set of normalisation conventions, which allows for easier cross-code
    comparisons.

    Users are not expected to initialise ``GKOutput`` objects directly,
    and in most cases should instead make use of the function ``read_gk_output``.

    The inputs to ``GKOutput`` should be given "physical units", as defined on the
    :ref:`sec-normalisation` page, appropriate to the code that generated the output
    data. If inputs are not given units, it is assumed they are already compliant with
    the Pyrokinetics standards.

    Parameters
    ----------
    time: ArrayLike, units [vref / lref]
        1D grid of time of the simulation output
    kx: ArrayLike, units [1.0 / rhoref]
        1D grid of radial wave-numbers used in the simulation
    ky: ArrayLike, units [1.0 / rhoref]
        1D grid of bi-normal wave-numbers used in the simulation
    theta: ArrayLike, units [radians]
        1D grid of theta used in the simulation
    energy: ArrayLike, units [dimensionless]
        1D grid of energy grid used in the simulation. (TODO are the units here really
        dimensionless? Is it relative to some reference energy?)
    pitch: ArrayLike, units [dimensionless]
        1D grid of pitch-angle grid used in the simulation. (TODO which definition of
        pitch angle are we using? Are the units correct?)
    fields: FieldDict
        Complex field data as a function of ``(theta, kx, ky, time)``.
    fluxes: Dict[FluxDict, ArrayLike]
        Flux data as a function of ``(ky, time)``. ``FluxDict`` is a dataclass
        denoting the moment, species, and field of each flux.
    norm: SimulationNormalisation
        The normalisation scheme of the simulation.
    linear: bool, default True
        Set True for linear gyrokinetics runs, False for nonlinear runs.
    gk_code: Optional[str], default None
        The gyrokinetics code that generated the results.
    input_file: Optional[str], default None
        The input file used to generate the results.
    growth_rate: Optional[ArrayLike], default None
        Function of ``(kx, ky, time)``. Only included for linear runs. If not provided,
        will be set from field data.
    mode_frequency: Optional[ArrayLike], default None
        Function of ``(kx, ky, time)``. Only included for linear runs. If not provided,
        will be set from field data.
    eigenfunctions: Optional[FieldDict], default None
        Complex function of ``(theta, kx, ky, time)``. One should be provided for
        each field. Only included for linear runs. If not provided, will be set from
        field data.

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

    # Instance of reader factory
    # Classes which can read output files (CGYRO, GS2, GENE etc) are registered
    # to this using the `reader` decorator below.
    _readers = create_reader_factory()

    def __init__(
        self,
        *,  # args are keyword only
        time: ArrayLike,
        kx: ArrayLike,
        ky: ArrayLike,
        theta: ArrayLike,
        field_dim: ArrayLike,
        moment: ArrayLike,
        species: ArrayLike,
        fields: FieldDict,
        fluxes: FluxDict,
        norm: SimulationNormalisation,
        linear: bool = True,
        mode: Optional[ArrayLike] = None,
        energy: Optional[ArrayLike] = None,
        pitch: Optional[ArrayLike] = None,
        gk_code: Optional[str] = None,
        input_file: Optional[str] = None,
        growth_rate: Optional[ArrayLike] = None,
        mode_frequency: Optional[ArrayLike] = None,
        eigenfunctions: Optional[FieldDict] = None,
    ):
        self.norm = norm
        convention = norm.pyrokinetics
        coord_units = get_coord_units(convention)

        def _renormalise(data: ArrayLike, convention: ConventionNormalisation, units):
            """
            Assign units to data if it doesn't have any. If it does, convert it to the
            provided convention.
            """
            # Not sure on type hints for units or return type here
            if hasattr(data, "units"):
                return data.to(convention)
            return np.asarray(data) * units

        def _assign_units(data: ArrayLike, units):
            """
            Assign non-normalised units to data if it doesn't have any. If it does,
            convert to the provided units.
            """
            if data is None:
                return None
            # Not sure on type hints for units or return type here
            if hasattr(data, "units"):
                return data.to(units)
            return np.asarray(data) * units

        # Assign correct normalised units to each input
        time = _renormalise(time, convention, coord_units["time"])
        kx = _renormalise(kx, convention, coord_units["kx"])
        ky = _renormalise(ky, convention, coord_units["ky"])
        theta = _assign_units(theta, coord_units["theta"])
        energy = _assign_units(energy, coord_units["energy"])
        pitch = _assign_units(pitch, coord_units["pitch"])
        mode = _assign_units(mode, coord_units["mode"])

        field_units = get_field_units(convention)
        for name, field in fields.items():
            fields[name] = _renormalise(field, convention, field_units[name])
            # check dims
            if gk_code == "TGLF":
                if np.shape(field) != (len(ky), len(mode)):
                    raise ValueError(f"field '{name}' has incorrect shape")
            else:
                if np.shape(field) != (len(theta), len(kx), len(ky), len(time)):
                    raise ValueError(f"field '{name}' has incorrect shape")

        flux_units = get_flux_units(convention)
        for flux_type, flux in fluxes.items():
            units = flux_units[flux_type]
            fluxes[flux_type] = _renormalise(fluxes[flux_type], convention, units)
            # check dims
            if gk_code == "GENE":
                if np.shape(flux) != (len(field_dim), len(species), len(time)):
                    raise ValueError(f"flux '{flux_type}' has incorrect shape")
            elif gk_code == "TGLF":
                if np.shape(flux) != (len(field_dim), len(species), len(ky)):
                    raise ValueError(f"flux '{flux_type}' has incorrect shape")
            else:
                if np.shape(flux) != (len(field_dim), len(species), len(ky), len(time)):
                    raise ValueError(f"flux '{flux_type}' has incorrect shape")

        # Assemble grids into underlying xarray Dataset
        def make_var(dim, val, desc):
            if val is None:
                return None
            else:
                return (dim, val.magnitude, {"units": str(val.units), "long_name": desc})

        def make_var_unitless(dim, val, desc):
            if val is None:
                return None
            else:
                return (dim, val, {"units": None, "long_name": desc})

        coords = {
            "time": make_var("time", time, "Time"),
            "kx": make_var("kx", kx, "Radial wavenumber"),
            "ky": make_var("ky", ky, "Bi-normal wavenumber"),
            "theta": make_var("theta", theta, "Angle"),
            "energy": make_var("energy", energy, "Energy"),
            "pitch": make_var("pitch", pitch, "Pitch angle"),
            "field": make_var_unitless("field", field_dim, "Field"),
            "moment": make_var_unitless("moment", moment, "Moment"),
            "species": make_var_unitless("species", species, "Species"),
            "mode": make_var_unitless("mode", mode, "Mode"),
        }

        coords = {key: value for key, value in coords.items() if value is not None}

        data_vars = {}
        field_desc = {
            "phi": "Electrostatic potential",
            "apar": "Parallel magnetic vector potential",
            "bpar": "Parallel magnetic flux density",
        }

        if gk_code == "TGLF":
            field_var = ("ky", "mode")
        else:
            field_var = ("theta", "kx", "ky", "time")

        for key, value in fields.items():
            data_vars[key] = make_var(
                field_var,
                value,
                field_desc[key],
            )

        if gk_code == "GENE":
            flux_vars = ("field", "species", "time")
        elif gk_code == "TGLF":
            flux_vars = ("field", "species", "ky")
        else:
            flux_vars = ("field", "species", "ky", "time")

        for flux_type, flux in fluxes.items():
            data_vars[flux_type] = make_var(
                flux_vars,
                flux,
                flux_type,
            )

        # TODO need way to stringify norm so we can keep track of units in the dataset
        attrs = {
            "linear": linear,
            "gk_code": gk_code if gk_code is not None else "",
            "input_file": input_file if input_file is not None else "",
        }

        # Add eigenvalues. If not provided, try to generate from fields
        eigenvalues_none = sum([growth_rate is None, mode_frequency is None])
        if eigenvalues_none == 1:
            raise ValueError(
                "Must provide both growth_rate and mode_frequency or neither one"
            )
        eig_units = get_eigenvalues_units(convention)
        if eigenvalues_none:
            if fields and linear:
                eigenvalues_dict = self._eigenvalues_from_fields(
                    fields, theta.magnitude, time.magnitude
                )
                for key, value in eigenvalues_dict.items():
                    eigenvalues_dict[key] = value * eig_units[key]
            else:
                eigenvalues_dict = {}
        else:
            growth_rate = _renormalise(
                growth_rate, convention, eig_units["growth_rate"]
            )
            mode_frequency = _renormalise(
                mode_frequency, convention, eig_units["mode_frequency"]
            )
            eigenvalues_dict = {
                "growth_rate": growth_rate,
                "mode_frequency": mode_frequency,
                "eigenvalues": mode_frequency + 1j * growth_rate
            }

        if gk_code == "TGLF":
            eigval_var = ("ky", "mode")
        else:
            eigval_var = ("kx", "ky", "time")
        for key, value in eigenvalues_dict.items():
            data_vars[key] = make_var(eigval_var, value, key)

        # Add eigenfunctions. If not provided, try to generate from fields
        eigenfunctions_dict = {}
        if eigenfunctions is None:
            if fields and linear:
                eigenfunctions_dict["eigenfunctions"] = self._eigenfunctions_from_fields(
                    fields, theta.magnitude
                )
        else:
            eigenfunctions_dict["eigenfunctions"] = eigenfunctions

        if gk_code == "TGLF":
            eigenfunctions_var = ("theta", "mode", "field")
        else:
            eigenfunctions_var = ("field", "theta", "kx", "ky", "time")

        for key, value in eigenfunctions_dict.items():
            data_vars[key] = (
                    eigenfunctions_var,
                    value,
                    {"long_name": "Eigenfunctions"},
                )

        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

    def field(self, name: str) -> xr.DataArray:
        if name not in ("phi", "apar", "bpar"):
            raise ValueError(
                f"name should be one of 'phi', 'apar', 'bpar'. Received '{name}'"
            )
        if name not in self.data_vars:
            raise ValueError(f"GKOutput does not contain the field '{name}'")
        return self.data_vars[name]

    def flux(self, name: str) -> xr.DataArray:
        if name not in ("particle", "heat", "momentum"):
            raise ValueError(
                f"name should be one of 'particle', 'heat', 'momentum'. Received '{name}'"
            )
        if name not in self.data_vars:
            raise ValueError(f"GKOutput does not contain the field '{name}'")
        return self.data_vars[name]

    def growth_rate_tolerance(self, time_range: float = 0.8) -> float:
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
        growth_rate = self.data_vars["growth_rate"]
        final_growth_rate = growth_rate.isel(time=-1)
        difference = np.abs((growth_rate - final_growth_rate) / final_growth_rate)
        final_time = self.coords["time"].isel(time=-1).data
        # Average over the end of the simulation, starting at time_range*final_time
        within_time_range = self.coords["time"].data > time_range * final_time
        tolerance = np.sum(
            np.where(within_time_range, difference, 0), axis=-1
        ) / np.sum(within_time_range, axis=-1)
        return tolerance

    @staticmethod
    def _eigenvalues_from_fields(
        fields: FieldDict, theta: ArrayLike, time: ArrayLike
    ) -> Dict[str, np.ndarray]:
        """
        Call during __init__ after converting to pyro normalisations
        """
        # field dims are (theta, kx, ky, time)
        shape = np.shape([*fields.values()][0])
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

        eigenvalues = mode_frequency + 1j * growth_rate

        return {"growth_rate": growth_rate, "mode_frequency": mode_frequency, "eigenvalues": eigenvalues}

    @staticmethod
    def _eigenfunctions_from_fields(fields: FieldDict, theta: ArrayLike) -> FieldDict:
        # field coords are (theta, kx, ky, time)
        square_fields = np.zeros(np.shape([*fields.values()][0]))
        for name, field in fields.items():
            square_fields += np.abs(field.magnitude) ** 2
        field_amplitude = np.sqrt(np.trapz(square_fields, theta, axis=0)) / (2 * np.pi)
        eigenfunctions = np.zeros((len(fields),) + square_fields.shape)
        for ifield, (name, field) in enumerate(fields.items()):
            eigenfunctions[ifield] = field.magnitude / field_amplitude
        return eigenfunctions

    @classmethod
    def reader(cls, key: str) -> Callable:
        r"""
        Decorator for classes that inherit Reader and create ``GKOutput`` objects.
        Registers classes with the global factory, and sets the class-level attribute
        'file_type' to the provided key. Can be used to register user-created plugins
        for equilibrium file readers.

        Parameters
        ----------
        key: str
            The registered name for the Reader class. When building ``GKOutput`` from a
            file using ``from_file``, the optional ``eq_type`` argument will correspond
            to this name.

        Returns
        -------
        Callable
            The decorator function that registers the class with the
            ``GKOutput._readers`` factory.

        Examples
        --------

        ::

            # Use this to decorate classes which inherit Reader and define the functions
            # 'read' and (optionally) 'verify'. Provide a key that will be used as an
            # identifier.
            @GKOutout.reader("MyGKOutput")
            class MyGKOutputReader(Reader):

                def read(self, path):
                    pass

                def verify(self, path):
                    pass

            # MyGKOutputReader will now contain the 'file_type' attribute
            assert MyGKOutputReader.file_type == "MyGKOutput"

            # The user can now read files of this type
            gk_output = GKOutput.from_file("MyGKOutput.nc", eq_type="MyGKOutput")
        """

        def decorator(t: Type[Reader]) -> Type[Reader]:
            cls._readers[key] = t
            t.file_type = key
            return t

        return decorator

    @classmethod
    def from_file(
        cls,
        path: PathLike,
        norm: SimulationNormalisation,
        gk_type: Optional[str] = None,
        **kwargs,
    ):
        r"""
        Read gyrokinetics output file(s) from disk, returning an ``GKOutput`` instance.

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
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File {path} not found.")
        # Infer reader type from path if not provided with eq_type
        reader = cls._readers[path if gk_type is None else gk_type]
        gk_output = reader(path, norm=norm, **kwargs)
        if not isinstance(gk_output, cls):
            raise RuntimeError("GKOutput reader did not return a GKOutput")
        return gk_output

    @classmethod
    def supported_types(cls):
        """
        Returns a list of all registered GKOutput file types. These file types are
        readable by ``from_file``.
        """
        return [*cls._readers]


def supported_gk_output_types() -> List[str]:
    """
    Returns a list of all registered GKOutput file types. These file types are
    readable by ``from_file``.
    """
    return GKOutput.supported_types()
