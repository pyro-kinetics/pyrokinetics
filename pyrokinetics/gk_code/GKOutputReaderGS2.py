import numpy as np
import xarray as xr
import logging
from itertools import product
from typing import Tuple, Optional, Any
from pathlib import Path
import warnings

from .GKOutputReader import GKOutputReader
from .GKInputGS2 import GKInputGS2
from ..constants import sqrt2
from ..typing import PathLike


class GKOutputReaderGS2(GKOutputReader):
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
    def _init_dataset(raw_data: xr.Dataset, gk_input: GKInputGS2) -> xr.Dataset:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a GS2 dataset
        """
        # ky coords
        ky = raw_data["ky"].data / sqrt2

        # time coords
        time_divisor = sqrt2
        try:
            if gk_input.data["knobs"]["wstar_units"]:
                time_divisor = ky
        except KeyError:
            pass
        time = raw_data["t"].data / time_divisor

        # kx coords
        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(raw_data["kx"].data) / sqrt2

        # theta coords
        theta = raw_data["theta"].data

        # energy coords
        try:
            energy = raw_data["energy"].data  # new diagnostics
        except KeyError:
            energy = raw_data["egrid"].data  # old diagnostics

        # pitch coords
        pitch = raw_data["lambda"].data

        # moment coords
        moment = ["particle", "energy", "momentum"]

        # field coords
        # If fphi/fapar/fbpar not in 'knobs', or they equal zero, skip the field
        # TODO is there some way to get this info without looking at the input data?
        field_vals = {}
        for field, default in zip(["phi", "apar", "bpar"], [1.0, 0.0, -1.0]):
            try:
                field_vals[field] = gk_input.data["knobs"][f"f{field}"]
            except KeyError:
                field_vals[field] = default
        # By default, fbpar = -1, which tells gs2 to try reading faperp instead.
        # faperp is deprecated, but is treated as a synonym for fbpar
        # It has a default value of 0.0
        if field_vals["bpar"] == -1:
            try:
                field_vals["bpar"] = gk_input.data["knobs"]["faperp"]
            except KeyError:
                field_vals["bpar"] = 0.0
        fields = [field for field, val in field_vals.items() if val > 0]

        # species coords
        # TODO is there some way to get this info without looking at the input data?
        species = []
        ion_num = 0
        for idx in range(gk_input.data["species_knobs"]["nspec"]):
            if gk_input.data[f"species_parameters_{idx+1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        # Store grid data as xarray DataSet
        return xr.Dataset(
            coords={
                "time": time,
                "kx": kx,
                "ky": ky,
                "theta": theta,
                "energy": energy,
                "pitch": pitch,
                "moment": moment,
                "field": fields,
                "species": species,
            },
            attrs={
                "ntime": len(time),
                "nkx": len(kx),
                "nky": len(ky),
                "ntheta": len(theta),
                "nenergy": len(energy),
                "npitch": len(pitch),
                "nmoment": len(moment),
                "nfield": len(fields),
                "nspecies": len(species),
                "linear": gk_input.is_linear(),
                "time_divisor": time_divisor,
            },
        )

    @staticmethod
    def _set_fields(
        data: xr.Dataset, raw_data: xr.Dataset, gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)

        For GS2 to print fields, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also instruct GS2 to print each field
        individually in the gs2_diagnostics_knobs using:
        - write_phi_over_time = .true.
        - write_apar_over_time = .true.
        - write_bpar_over_time = .true.
        - write_fields = .true.
        """
        # Check to see if there's anything to do
        field_names = [f"{field}_t" for field in data["field"].data]
        if not np.any(np.isin(field_names, raw_data.data_vars)):
            return data

        coords = ["field", "theta", "kx", "ky", "time"]
        fields = np.empty([data.dims[coord] for coord in coords], dtype=complex)

        # Loop through all fields and add field if it exists
        for ifield, field_name in enumerate(field_names):

            if field_name not in raw_data:
                logging.warning(
                    f"Field data over time {field_name} not written to netCDF file. "
                    "Setting this field to 0"
                )
                # Note: we could instead set this to the field at the last time step
                fields[ifield, :, :, :, :] = 0
                continue

            # raw_field has coords (t,ky,kx,theta,real/imag).
            # We wish to transpose that to (real/imag,theta,kx,ky,t)
            field_data = raw_data[field_name].transpose("ri", "theta", "kx", "ky", "t")
            fields[ifield, :, :, :, :] = sqrt2 * (
                field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]
            )

        # Shift kx=0 to middle of axis
        fields = np.fft.fftshift(fields, axes=2)

        data["fields"] = (coords, fields)
        return data

    @staticmethod
    def _set_fluxes(
        data: xr.Dataset, raw_data: xr.Dataset, gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Set flux data over time.
        The flux coordinates should be (species, moment, field, ky, time)

        For GS2 to print fluxes, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also set the following in
        gs2_diagnostics_knobs:
        - write_fluxes = .true. (default if nonlinear)
        - write_fluxes_by_mode = .true. (default if nonlinear)
        """
        # field names change from ["phi", "apar", "bpar"] to ["es", "apar", "bpar"]
        # Take whichever fields are present in data, relabelling "phi" to "es"
        fields = [("es" if f == "phi" else f) for f in data["field"].data]
        moments = ["part", "heat", "mom"]
        suffixes = ["flux", "by_k", "flux_by_mode"]

        # Check to see if there's anything to do
        flux_names = [f"{x}_{y}_{z}" for x, y, z in product(fields, moments, suffixes)]
        if not np.any(np.isin(flux_names, raw_data.data_vars)):
            return data

        coords = ["species", "moment", "field", "ky", "time"]
        fluxes = np.empty([data.dims[coord] for coord in coords])

        for idx in product(enumerate(fields), enumerate(moments)):
            (ifield, field), (imoment, moment) = idx

            flux_key = f"{field}_{moment}_flux"
            by_k_key = f"{field}_{moment}_by_k"  # old diagnostics
            by_mode_key = f"{field}_{moment}_flux_by_mode"  # new diagnostics

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
                logging.warning(
                    f"Flux data {flux_key}/{by_k_key} not written to netCDF file. "
                    "Setting this flux to 0."
                )
                flux = 0

            fluxes[:, imoment, ifield, :, :] = flux

        data["fluxes"] = (coords, fluxes)
        return data

    @staticmethod
    def _set_eigenvalues(
        data: xr.Dataset, raw_data: Optional[Any] = None, gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        if "fields" in data:
            return GKOutputReader._set_eigenvalues(data, raw_data, gk_input)

        warnings.warn(
            "'fields' not set in data, falling back to 'omega_average' -- 'eigenvalues' will not be set!"
        )

        frequency = raw_data.omega_average.isel(ri=0).data / data.time_divisor
        growth_rate = raw_data.omega_average.isel(ri=1).data / data.time_divisor

        data["mode_frequency"] = (("time", "ky", "kx"), frequency)
        data["growth_rate"] = (("time", "ky", "kx"), growth_rate)

        return data
