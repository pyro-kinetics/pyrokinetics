import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


from ..readers import create_reader_factory
from ..dataset_wrapper import DatasetWrapper
from ..normalisation import SimulationNormalisation as Normalisation
from pyrokinetics.units import ureg as units


from .utils import gk_output_units


def get_growth_rate_tolerance(data: xr.Dataset, time_range: float = 0.8):
    """
    Given a pyrokinetics output dataset with eigenvalues determined, calculate the
    growth rate tolerance. This is calculated starting at the time given by
    time_range * max_time.
    """
    if "growth_rate" not in data:
        raise ValueError(
            "Provided Dataset does not have growth rate. The dataset should be "
            "associated with a linear gyrokinetics runs"
        )
    growth_rate = data["growth_rate"]
    final_growth_rate = growth_rate.isel(time=-1)
    difference = np.abs((growth_rate - final_growth_rate) / final_growth_rate)
    final_time = difference["time"].isel(time=-1).data
    # Average over the end of the simulation, starting at time_range*final_time
    within_time_range = difference["time"].data > time_range * final_time
    tolerance = np.sum(np.where(within_time_range, difference, 0), axis=-1) / np.sum(
        within_time_range, axis=-1
    )
    return tolerance


class GKOutput(DatasetWrapper):
    """
    GKOutput contains the output data from gyrokinetics codes, and converts it to
    Users are not expected to initialise ``GKOutput`` objects directly,
    and in most cases should instead make use of the function ``read_gk_output``,
    which can read gk_outputs files from different gk_codes
    A standardised schema to allow for easier cross-code comparisons. Using the read
    method, it takes in output data typically expressed as a .cdf file, and converts it
    to an xarray Dataset. The functions _set_grids, _set_fields, _set_fluxes,
    _set_eigenvalues, _set_eigenfunctions, and _set_growth_rate_tolerance are used to
    build up the Dataset, and need not be called by the user.

    The produced xarray Dataset should have the following:

    Parameters
    ----------
    time: ArrayLike, units [vref / lref]
        1D grid of time of the simulation output
    kx: ArrayLike, units [/ rhoref]
        1D grid of radial wave-numbers used in the simulation
    ky: ArrayLike, units [/ rhoref]
        1D grid of bi-normal wave-numbers used in the simulation
    theta: ArrayLike, units [radians]
        1D grid of theta used in the simulation
    energy_dim: ArrayLike, units [dimensionless]
        1D grid of energy grid used in the simulation
    pitch: [dimensionless]
        1D grid of pitch-angle grid used in the simulation


    phi: ArrayLike, units [qref / tref * lref / rhoref]
        4D (theta, kx, ky, time) complex array of the electrostatic potential, may be zeros
    apar: ArrayLike, units [1 / bref * lref / rhoref **2]
        4D (theta, kx, ky, time) complex array of the parallel component of the magnetic vector potential, may be zeros
    bpar: ArrayLike, units [1 / bref * lref / rhoref]
        4D (theta, kx, ky, time) complex array of the parallel component of the magnetic field,

    particle               (species, field, ky, time) float array, units [nref * vref * (rhoref / lref)**2], may be zeros
    momentum               (species, field, ky, time) float array, units [nref * lref * tref * (rhoref / lref)**2], may be zeros
    energy                 (species, field, ky, time) float array, units [nref * vref * tref * (rhoref / lref)**2], may be zeros
    growth_rate            (kx, ky, time) float array, units = [vref / lref] linear only
    mode_frequency         (kx, ky, time) float array, units = [vref / lref] linear only
    eigenvalues            (kx, ky, time) float array, units = [vref / lref] linear only
    eigenfunctions         (field, theta, kx, ky, time) units [dimensionless] float array, linear only
    growth_rate_tolerance  (kx, ky) float array, [dimensionless] linear only

    Attributes
    ----------

    data: xarray.Dataset
        The internal representation of the ``Equilibrium`` object. The functions
        ``__getattr__`` and ``__getitem__`` redirect most attribute/indexing lookups
        here, but the Dataset itself may be accessed directly by the user if they wish
        to perform more complex manipulations.
    moment      ["particle", "heat", "momentum"]
    field       ["phi", "apar", "bpar"] (the number appearing depends on nfield)
    species     list of species names (e.g. "electron", "ion1", "deuterium", etc)
    local_norm  SimulationNormalisation of output

    input_file   gk input file expressed as a string
    ntime        length of time coords
    nkx          length of kx coords
    nky          length of ky coords
    ntheta       length of theta coords
    nenergy      length of energy coords
    npitch       length of pitch coords
    nfield       length of field coords
    nspecies     length of species coords
    """

    # Instance of reader factory
    # Classes which can read output files (CGYRO, GS2, GENE etc) are registered
    # to this using the `gk_output_reader` decorator below.
    _readers = create_reader_factory()

    _init_units = {
        "self": None,
        "ky": gk_output_units["ky"],
        "kx": gk_output_units["kx"],
        "theta": gk_output_units["theta"],
        "time": gk_output_units["time"],
        "energy": gk_output_units["energy"],
        "pitch": gk_output_units["pitch"],
        "phi": gk_output_units["phi"],
        "apar": gk_output_units["apar"],
        "bpar": gk_output_units["bpar"],
        "particle": gk_output_units["particle"],
        "momentum": gk_output_units["momentum"],
        "heat": gk_output_units["heat"],
        "growth_rate": gk_output_units["growth_rate"],
        "mode_frequency": gk_output_units["mode_frequency"],
        "eigenvalues": gk_output_units["eigenvalues"],
        "eigenfunctions": gk_output_units["eigenfunctions"],
    }

    @units.wraps(None, [*_init_units.values()], strict=False)
    def __init__(
        self,
        ky: ArrayLike,
        kx: ArrayLike,
        theta: ArrayLike,
        time: ArrayLike,
        energy: ArrayLike,
        pitch: ArrayLike,
        moment: ArrayLike,
        field: ArrayLike,
        species: ArrayLike,
        phi: ArrayLike,
        apar: ArrayLike,
        bpar: ArrayLike,
        particle: ArrayLike,
        momentum: ArrayLike,
        heat: ArrayLike,
        growth_rate: ArrayLike,
        mode_frequency: ArrayLike,
        eigenvalues: ArrayLike,
        eigenfunctions: ArrayLike,
        local_norm: Normalisation,
        linear: bool,
    ):
        pass

        # Assemble grids into underlying xarray Dataset
        def make_var(dim, val, desc):
            return (dim, val, {"units": str(val.units), "long_name": desc})

        coords = {
            "time": make_var("time_dim", time, "Time"),
            "kx": make_var("kx_dim", kx, "Radial wavenumber"),
            "ky": make_var("ky_dim", ky, "Bi-normal wavenumber"),
            "theta": make_var("theta_dim", theta, "Angle"),
            "energy": make_var("energy_dim", energy, "Energy"),
            "pitch": make_var("pitch_dim", pitch, "Pitch angle"),
            "moment": make_var("moment_dim", moment, "Moment of distribution fn"),
            "field": make_var("field_dim", field, "Field"),
            "species": make_var("species_dim", species, "Species"),
        }

        data_vars = {
            "phi": make_var(
                ("theta", "kx", "ky", "time"), phi, "Electrostatic potential"
            ),
            "apar": make_var(
                ("theta", "kx", "ky", "time"),
                apar,
                "Parallel magnetic vector potential",
            ),
            "bpar": make_var(
                ("theta", "kx", "ky", "time"), bpar, "Parallel magnetic field"
            ),
            "particle": make_var(
                ("species", "moment", "field", "ky", "time"), particle, "Particle flux"
            ),
            "momentum": make_var(
                ("species", "moment", "field", "ky", "time"), momentum, "Momentum flux"
            ),
            "heat": make_var(
                ("species", "moment", "field", "ky", "time"), heat, "Heat flux"
            ),
            "growth_rate": make_var(
                ("kx", "ky", "field", "theta", "time"), growth_rate, "Growth rate"
            ),
            "mode_frequency": make_var(
                ("theta", "kx", "ky", "time"), mode_frequency, "Mode frequency"
            ),
            "eigenvalues": make_var(
                ("theta", "kx", "ky", "time"), eigenvalues, "Eigenvalues"
            ),
            "eigenfunctions": make_var(
                ("field", "theta", "kx", "ky", "time"), eigenfunctions, "Eigenfunctions"
            ),
        }

        attrs = {
            "ntime": len(time),
            "nkx": len(kx),
            "nky": len(ky),
            "ntheta": len(theta),
            "nenergy": len(energy),
            "npitch": len(pitch),
            "nmoment": len(moment),
            "nfield": len(field),
            "nspecies": len(species),
            "local_norm": local_norm,
            "linear": linear,
        }
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
