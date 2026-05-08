import re
from textwrap import dedent
from typing import Optional

import numpy as np
from path import Path

from ..constants import deuterium_mass, electron_mass, hydrogen_mass, tritium_mass
from ..equilibrium import Equilibrium
from ..file_utils import FileReader
from ..species import Species
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .kinetics import Kinetics

_FLOAT_PATTERN = re.compile(r"[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[Ee][+-]?\d+)?")

_ION_MASSES = {
    "hydrogen": hydrogen_mass,
    "deuterium": deuterium_mass,
    "tritium": tritium_mass,
}


def _numbers_from_line(line):
    return [float(match.group(0)) for match in _FLOAT_PATTERN.finditer(line)]


def _read_iterdb_blocks(filename: PathLike):
    with open(filename) as file:
        lines = file.readlines()

    blocks = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if "DEPENDENT VARIABLE LABEL" not in line or "INDEPENDENT" in line:
            index += 1
            continue

        variable = line.split()[0].upper()

        while index < len(lines) and "# OF X PTS" not in lines[index]:
            index += 1
        if index == len(lines):
            raise ValueError(f"ITERDB block '{variable}' is missing X grid size")
        nx = int(_numbers_from_line(lines[index])[0])

        index += 1
        if index == len(lines) or "# OF Y PTS" not in lines[index]:
            raise ValueError(f"ITERDB block '{variable}' is missing Y grid size")
        ny = int(_numbers_from_line(lines[index])[0])

        expected_values = nx + ny + nx * ny
        values = []
        index += 1
        while index < len(lines) and len(values) < expected_values:
            values.extend(_numbers_from_line(lines[index]))
            index += 1

        if len(values) < expected_values:
            raise ValueError(
                f"ITERDB block '{variable}' ended before all data was read"
            )

        x_grid = np.asarray(values[:nx])
        time_grid = np.asarray(values[nx : nx + ny])
        data = np.asarray(values[nx + ny : expected_values]).reshape(ny, nx)
        blocks[variable] = {
            "x": x_grid,
            "time": time_grid,
            "data": data,
        }

    if not blocks:
        raise ValueError("No ITERDB UFILE profile blocks were found")

    return blocks


def _select_time_slice(block, time_index: int = -1, time: Optional[float] = None):
    if time_index != -1 and time is not None:
        raise ValueError("Cannot set both `time` and `time_index`")

    if time is not None:
        time_index = int(np.argmin(np.abs(block["time"] - time)))
    return block["x"], block["data"][time_index]


def _profile(blocks, name, time_index: int, time: Optional[float], required=True):
    try:
        return _select_time_slice(blocks[name], time_index, time)
    except KeyError:
        if required:
            raise ValueError(f"ITERDB file does not contain required profile '{name}'")
        return None


def _rho_from_equilibrium(eq: Equilibrium):
    rho = eq["r_minor"].data
    rho = rho / rho[-1] * units.lref_minor_radius
    return UnitSpline(eq["psi_n"].data, rho)


class KineticsReaderITERDB(FileReader, file_type="ITERDB", reads=Kinetics):
    def read_from_file(
        self,
        filename: PathLike,
        eq: Equilibrium = None,
        time_index: int = -1,
        time: Optional[float] = None,
        main_ion: str = "deuterium",
        main_ion_charge: float = 1.0,
        main_ion_mass=None,
        rotation_sign: float = 1.0,
        use_rhotor_as_rho: bool = False,
    ) -> Kinetics:
        """
        Reads an ITERDB/UFILE-style profile file.

        The reader expects profile blocks with ``RHOTOR`` as the independent radial
        coordinate and at least ``TE``, ``TI``, ``NE`` and ``NM1``. ``VROT`` is used
        when present, otherwise the angular rotation is set to zero.
        """
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(filename)

        if eq is None and not use_rhotor_as_rho:
            raise ValueError(dedent(f"""\
                    {self.__class__.__name__} must be provided with an Equilibrium
                    object via the keyword argument 'eq'. Please load an Equilibrium.
                    """))

        blocks = _read_iterdb_blocks(filename)

        te_rhotor, te = _profile(blocks, "TE", time_index, time)
        ti_rhotor, ti = _profile(blocks, "TI", time_index, time)
        ne_rhotor, ne = _profile(blocks, "NE", time_index, time)
        ni_rhotor, ni = _profile(blocks, "NM1", time_index, time)
        vrot_profile = _profile(blocks, "VROT", time_index, time, required=False)

        electron_psi_n = te_rhotor**2 * units.dimensionless
        ion_temp_psi_n = ti_rhotor**2 * units.dimensionless
        electron_dens_psi_n = ne_rhotor**2 * units.dimensionless
        ion_dens_psi_n = ni_rhotor**2 * units.dimensionless

        electron_temp_func = UnitSpline(electron_psi_n, te * units.eV)
        ion_temp_func = UnitSpline(ion_temp_psi_n, ti * units.eV)
        electron_dens_func = UnitSpline(electron_dens_psi_n, ne * units.meter**-3)
        ion_dens_func = UnitSpline(ion_dens_psi_n, ni * units.meter**-3)

        if vrot_profile is None:
            omega_psi_n = electron_psi_n
            omega = np.zeros(len(te_rhotor)) * units.radians / units.second
        else:
            vrot_rhotor, vrot = vrot_profile
            omega_psi_n = vrot_rhotor**2 * units.dimensionless
            omega = rotation_sign * vrot * units.radians / units.second
        omega_func = UnitSpline(omega_psi_n, omega)

        if use_rhotor_as_rho:
            rho_func = UnitSpline(electron_psi_n, te_rhotor * units.lref_minor_radius)
        else:
            rho_func = _rho_from_equilibrium(eq)

        unit_charge_array = np.ones(len(electron_psi_n))
        electron_charge = UnitSpline(
            electron_psi_n, -1 * unit_charge_array * units.elementary_charge
        )

        if main_ion_mass is None:
            try:
                main_ion_mass = _ION_MASSES[main_ion.lower()]
            except KeyError:
                raise ValueError(
                    "main_ion_mass must be provided when main_ion is not one of "
                    f"{', '.join(sorted(_ION_MASSES))}"
                )
        if not hasattr(main_ion_mass, "units"):
            main_ion_mass = main_ion_mass * units.kg

        ion_charge = main_ion_charge * units.elementary_charge
        ion_charge_func = UnitSpline(electron_psi_n, ion_charge * unit_charge_array)

        electron = Species(
            species_type="electron",
            charge=electron_charge,
            mass=electron_mass,
            dens=electron_dens_func,
            temp=electron_temp_func,
            omega0=omega_func,
            rho=rho_func,
        )

        ion = Species(
            species_type=main_ion,
            charge=ion_charge_func,
            mass=main_ion_mass,
            dens=ion_dens_func,
            temp=ion_temp_func,
            omega0=omega_func,
            rho=rho_func,
        )

        return Kinetics(kinetics_type="ITERDB", electron=electron, **{main_ion: ion})

    def verify_file_type(self, filename: PathLike) -> None:
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(filename)

        blocks = _read_iterdb_blocks(filename)
        required_profiles = {"TE", "TI", "NE", "NM1"}
        missing_profiles = required_profiles.difference(blocks)
        if missing_profiles:
            raise ValueError(
                "ITERDB file is missing required profiles "
                f"{', '.join(sorted(missing_profiles))}"
            )
