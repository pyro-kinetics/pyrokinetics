from itertools import product
from typing import Any, Dict, Tuple
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .gk_output import (
    GKOutput,
    get_flux_units,
    get_field_units,
    get_coord_units,
    get_eigenvalues_units,
    FieldDict,
    FluxType,
)
from .GKInputGS2 import GKInputGS2
from ..typing import PathLike
from ..readers import Reader
from ..normalisation import SimulationNormalisation


@GKOutput.reader("GS2")
class GKOutputReaderGS2(Reader):
    def read(self, filename: PathLike, norm: SimulationNormalisation) -> GKOutput:
        raw_data, gk_input, input_str = self._get_raw_data(filename)
        coords = self._get_coords(raw_data, gk_input)
        fields = self._get_fields(raw_data)
        fluxes = self._get_fluxes(raw_data, gk_input)

        # Assign units and return GKOutput
        convention = norm.gs2
        coord_units = get_coord_units(convention)
        field_units = get_field_units(convention)
        flux_units = get_flux_units(convention)
        eig_units = get_eigenvalues_units(convention)

        for field_name, field in fields.items():
            fields[field_name] = field * field_units[field_name]

        for flux_type, flux in fluxes.items():
            fluxes[flux_type] = flux * flux_units[flux_type.moment]

        if fields or coords["linear"]:
            # Rely on gk_output to generate eigenvalues
            growth_rate = None
            mode_frequency = None
        else:
            eigenvalues = self._get_eigenvalues(raw_data, coords["time_divisor"])
            growth_rate = eigenvalues["growth_rate"] * eig_units["growth_rate"]
            mode_frequency = eigenvalues["mode_frequency"] * eig_units["mode_frequency"]

        return GKOutput(
            time=coords["time"] * coord_units["time"],
            kx=coords["kx"] * coord_units["kx"],
            ky=coords["ky"] * coord_units["ky"],
            theta=coords["theta"] * coord_units["theta"],
            pitch=coords["pitch"] * coord_units["pitch"],
            energy=coords["energy"] * coord_units["energy"],
            fields=fields,
            fluxes=fluxes,
            norm=norm,
            linear=coords["linear"],
            gk_code="GS2",
            input_file=input_str,
            growth_rate=growth_rate,
            mode_frequency=mode_frequency,
        )

    def verify(self, filename: PathLike):
        data = xr.open_dataset(filename)
        if "software_name" in data.attrs:
            if data.attrs["software_name"] != "GS2":
                raise RuntimeError
        elif "code_info" in data.data_vars:
            if data["code_info"].long_name != "GS2":
                raise RuntimeError
        else:
            raise RuntimeError

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Gets path by removing ".in" and replacing it with ".out.nc"
        """
        filename = Path(filename)
        return filename.parent / (filename.stem + ".out.nc")

    @staticmethod
    def _get_raw_data(filename: PathLike) -> Tuple[xr.Dataset, GKInputGS2, str]:
        raw_data = xr.open_dataset(filename)
        # Read input file from netcdf, store as GKInputGS2
        input_file = raw_data["input_file"]
        if input_file.shape == ():
            # New diagnostics, input file stored as bytes
            # - Stored within numpy 0D array, use [()] syntax to extract
            # - Convert bytes to str by decoding
            # - \n is represented as character literals '\' 'n'. Replace with '\n'.
            input_str = input_file.data[()].decode("utf-8").replace(r"\n", "\n")
        else:
            # Old diagnostics (and eventually the single merged diagnostics)
            # input file stored as array of bytes
            input_str = "\n".join((line.decode("utf-8") for line in input_file.data))
        gk_input = GKInputGS2()
        gk_input.read_str(input_str)
        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(raw_data: xr.Dataset, gk_input: GKInputGS2) -> Dict[str, Any]:
        # ky coords
        ky = raw_data["ky"].data

        # time coords
        time_divisor = 1 / 2
        try:
            if gk_input.data["knobs"]["wstar_units"]:
                time_divisor = ky / 2
        except KeyError:
            pass

        time = raw_data["t"].data / time_divisor

        # kx coords
        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(raw_data["kx"].data)

        # theta coords
        theta = raw_data["theta"].data

        # energy coords
        try:
            energy = raw_data["energy"].data  # new diagnostics
        except KeyError:
            energy = raw_data["egrid"].data  # old diagnostics

        # pitch coords
        pitch = raw_data["lambda"].data

        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "linear": gk_input.is_linear(),
            "time_divisor": time_divisor,
        }

    @staticmethod
    def _get_fields(raw_data: xr.Dataset) -> FieldDict:
        """
        For GS2 to print fields, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also instruct GS2 to print each field
        individually in the gs2_diagnostics_knobs using:
        - write_phi_over_time = .true.
        - write_apar_over_time = .true.
        - write_bpar_over_time = .true.
        - write_fields = .true.
        """
        field_names = ("phi", "apar", "bpar")
        results = {}

        # Loop through all fields and add field if it exists
        for field_name in field_names:
            key = f"{field_name}_t"
            if key not in raw_data:
                continue

            # raw_field has coords (t,ky,kx,theta,real/imag).
            # We wish to transpose that to (real/imag,theta,kx,ky,t)
            field = raw_data[key].transpose("ri", "theta", "kx", "ky", "t").data
            field = field[0, ...] + 1j * field[1, ...]

            # Adjust fields to account for differences in defintions/normalisations
            if field_name == "apar":
                field *= 0.5

            if field_name == "bpar":
                bmag = raw_data["bmag"].data[:, np.newaxis, np.newaxis, np.newaxis]
                field *= bmag

            # Shift kx=0 to middle of axis
            field = np.fft.fftshift(field, axes=1)
            results[field_name] = field

        return results

    @staticmethod
    def _get_fluxes(raw_data: xr.Dataset, gk_input: GKInputGS2) -> Dict[str, ArrayLike]:
        """
        For GS2 to print fluxes, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also set the following in
        gs2_diagnostics_knobs:
        - write_fluxes = .true. (default if nonlinear)
        - write_fluxes_by_mode = .true. (default if nonlinear)
        """
        # field names change from ["phi", "apar", "bpar"] to ["es", "apar", "bpar"]
        # Take whichever fields are present in data, relabelling "phi" to "es"
        fields = {"phi": "es", "apar": "apar", "bpar": "bpar"}
        moments = {"particle": "part", "heat": "heat", "momentum": "mom"}

        # Get species names from input file
        species = []
        ion_num = 0
        for idx in range(gk_input.data["species_knobs"]["nspec"]):
            if gk_input.data[f"species_parameters_{idx+1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        results = {}

        for (field, gs2_field), (moment, gs2_moment) in product(
            fields.items(), moments.items()
        ):
            flux_key = f"{gs2_field}_{gs2_moment}_flux"
            # old diagnostics
            by_k_key = f"{gs2_field}_{gs2_moment}_by_k"
            # new diagnostics
            by_mode_key = f"{gs2_field}_{gs2_moment}_flux_by_mode"

            if by_k_key in raw_data.data_vars or by_mode_key in raw_data.data_vars:
                key = by_mode_key if by_mode_key in raw_data.data_vars else by_k_key
                flux = raw_data[key].transpose("species", "kx", "ky", "t")
                # Sum over kx
                flux = flux.sum(dim="kx")
                # Divide non-zonal components by 2 due to reality condition
                flux[:, 1:, :] *= 0.5
            elif flux_key in raw_data.data_vars:
                # coordinates from raw are (t,species)
                # convert to (species, ky, t)
                flux = raw_data[flux_key]
                flux = flux.expand_dims("ky").transpose("species", "ky", "t")
            else:
                continue

            for idx in raw_data["species"].data:
                flux_type = FluxType(moment=moment, field=field, species=species[idx])
                results[flux_type] = flux[idx, :, :]

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: xr.Dataset, time_divisor: float
    ) -> Dict[str, ArrayLike]:
        # should only be called if no field data were found
        mode_frequency = raw_data.omega_average.isel(ri=0).transpose("kx", "ky", "time")
        growth_rate = raw_data.omega_average.isel(ri=1).transpose("kx", "ky", "time")
        return {
            "mode_frequency": mode_frequency.data / time_divisor,
            "growth_rate": growth_rate.data / time_divisor,
        }
