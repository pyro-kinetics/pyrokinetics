import dataclasses
import json
import pprint
from typing import Any, ClassVar, Dict, Generator, Optional, Tuple

import pint

from .metadata import metadata
from .normalisation import ConventionNormalisation
from .units import ureg as units


@dataclasses.dataclass
class Numerics:
    """
    Stores information describing numerical features common to most gyrokinetics
    solvers, such as the dimensions of numerical grids, the presence of electromagnetic
    field components, and time steps. Numerics is not used to store special flags
    belonging to only one gyrokinetics code.
    """

    #: Number of elements in the :math:`\theta` (poloidal) grid
    ntheta: int = 32

    #: Number of :math:`2\pi` segments in the toroidal direction.
    nperiod: int = 1

    #: Number of elements in the energy grid
    nenergy: int = 8

    #: Number of elements in the pitch grid
    npitch: int = 8

    #: Number of elements in the velocity-space :math:`k_y` grid
    nky: int = 1

    #: Number of elements in the velocity-space :math:`k_x` grid
    nkx: int = 1

    #: Value of :math:`k_y\rho`
    ky: float = 0.1

    #: Value of :math:`k_x\rho`
    kx: float = 0.0

    #: Initial time step, in units of ``lref / vref``
    delta_time: float = 0.001

    #: Time step, in units of ``lref / vref``
    max_time: float = 500.0

    #: The ballooning angle (the point at which the radial wavenumber is zero)
    theta0: float = 0.0

    #: Boolean flag denoting whether this run evolves the :math:`\phi` field
    #: (electric potential).
    phi: bool = True

    #: Boolean flag denoting whether this run evolves the :math:`A_\parallel` field
    #: (component of the magnetic vector potential running parallel to the field line)
    apar: bool = False

    #: Boolean flag denoting whether this run evolves the :math:`B_\parallel` field
    #: (component of the magnetic flux density running parallel to the field line)
    bpar: bool = False

    #: Ratio of plasma pressure to magnetic pressure
    beta: Optional[float] = None

    #: Boolean flag noting whether this run includes non-linear features
    nonlinear: bool = False

    #: Perpendicular ExB shearing rate ``vref / lref``
    gamma_exb: Optional[float] = None

    #: Dict containing metadata about this Pyrokinetics session
    _metadata: Optional[Dict[str, str]] = None

    #: Title to be written to _metadata.
    #: Defined as an 'InitVar' meaning this isn't a variable stored by the dataclass,
    #: but instead is an optional argument to the constructor. It is used in the
    #: __post_init__ function. If unset, this defaults to the class name.
    title: dataclasses.InitVar[Optional[str]] = None

    _has_physical_units: ClassVar[Tuple[str, ...]] = ("theta0",)

    _has_normalised_units: ClassVar[Tuple[str, ...]] = (
        "kx",
        "ky",
        "delta_time",
        "max_time",
        "gamma_exb",
        "beta",
    )

    def __post_init__(self, title: Optional[str] = None):
        """Performs secondary construction after calling __init__"""
        if self._metadata is None:
            if title is None:
                title = self.__class__.__name__
            self._metadata = metadata(title, self.__class__.__name__)

    @property
    def names(self) -> Tuple[str, ...]:
        """Names of all quantities held by this dataclass"""
        return tuple(x.name for x in dataclasses.fields(self))

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in vars(self):
            raise KeyError(f"Numerics does not have a key '{key}'")
        setattr(self, key, value)

    def __setattr__(self, attr: str, value: Any) -> None:
        # TODO when minimum version is 3.10, can just use dataclass(slots=True)
        if attr not in (field.name for field in dataclasses.fields(self)):
            raise AttributeError(f"Numerics does not have an attribute '{attr}'")
        super().__setattr__(attr, value)

    def __str__(self) -> str:
        """'Pretty print' self"""
        # TODO when minimum version is 3.10, can remove asdict
        return pprint.pformat(dataclasses.asdict(self))

    def __iter__(self) -> Generator[str, None, None]:
        """Iterate over quantity names. Skips ``None`` quantities."""
        return iter(self.coords)

    @property
    def coords(self) -> Tuple[str, ...]:
        """
        Tuple containing the names of each supplied field (those that aren't ``None``).
        """
        return tuple(k for k in self.names if self[k] is not None)

    def values(self) -> Generator[Any, None, None]:
        """Dict-like values iteration"""
        try:
            it = iter(self)
            while True:
                yield self[next(it)]
        except StopIteration:
            return

    def to_json(self, **kwargs: Any) -> str:
        """
        Converts self to json string. Includes metadata describing the current
        Pyrokinetics session.

        Parameters
        ----------
        **kwargs: Any
            Parameters passed on to ``json.dumps``


        Examples
        --------
        ::

            with open("my_numerics.json", "w") as f:
                # Use indent=4 for pretty print
                f.write(my_numerics.to_json(indent=4))
        """
        return json.dumps(dataclasses.asdict(self), **kwargs)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        overwrite_metadata: bool = False,
        overwrite_title: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Creates a new Numerics from a previously saved Numerics json.

        Parameters
        ----------
        json_str: str
            Json string to read
        overwrite_metadata: bool, default False
            Take ownership of the Json data, overwriting attributes such as 'title',
            'software_name', 'date_created', etc.
        overwrite_title: Optional[str]
            If ``overwrite_metadata`` is ``True``, this is used to set the ``title``
            attribute in ``self._metadata``. If unset, the class name is used.
        **kwargs: Any
            Keyword arguments forwarded to ``json.loads``

        Examples
        --------
        ::

            with open("my_numerics.json", "r") as f:
                my_numerics = Numerics.from_json(f.read())
        """
        numerics_dict = json.loads(json_str, **kwargs)
        if overwrite_metadata:
            numerics_dict.pop("_metadata")
        return Numerics(**numerics_dict, title=overwrite_title)

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Dict-like items iteration"""
        return zip(iter(self), self.values())

    def units(self, name: str, c: ConventionNormalisation) -> pint.Unit:
        if name not in self.names:
            raise ValueError(f"The coord '{name}' is not recognised (expected one of {Self.names}")
        if name in ("kx", "ky"):
            return c.rhoref**-1
        if name in ("delta_time", "max_time"):
            return c.lref / c.vref
        if name == "theta0":
            return units.radians
        if name == "gamma_exb":
            return c.vref / c.lref
        if name == "beta":
            return c.beta_ref
        return units.dimensionless

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
