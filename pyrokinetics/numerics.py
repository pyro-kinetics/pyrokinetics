import dataclasses
import json
import pprint
from typing import Any, Dict, Optional

from .metadata import metadata


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

    #: Initial time step, in units of ``tref``
    delta_time: float = 0.001

    #: Time step, in units of ``tref``
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

    #: Dict containing metadata about this Pyrokinetics session
    _metadata: Optional[Dict[str, str]] = None

    #: Title to be written to _metadata.
    #: Defined as an 'InitVar' meaning this isn't a variable stored by the dataclass,
    #: but instead is an optional argument to the constructor. It is used in the
    #: __post_init__ function. If unset, this defaults to the class name.
    title: dataclasses.InitVar[Optional[str]] = None

    def __post_init__(self, title: Optional[str] = None):
        """Performs secondary construction after calling __init__"""
        if self._metadata is None:
            if title is None:
                title = self.__class__.__name__
            self._metadata = metadata(title=title, object_type=self.__class__.__name__)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __str__(self) -> str:
        """'Pretty print' self"""
        # TODO when minimum version is 3.10, can remove asdict
        return pprint.pformat(dataclasses.asdict(self))

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
