import numpy as np
import xarray as xr
import logging
from itertools import product

from .GKOutputReader import GKOutputReader
from .GKInputGS2 import GKInputGS2
from ..constants import sqrt2
from ..typing import PathLike

class GKOutputReaderGS2(GKOutputReader):
    def read(self, filename: PathLike, grt_time_range: float = 0.8) -> xr.Dataset:
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
        
        data = self._build_dataset( raw_data, gk_input = gk_input, is_linear = gk_input.is_linear(), grt_time_range = grt_time_range)
        data.assign_attrs({"input_str" : input_str })
        return data

    def verify(self, filename: PathLike):
        data = xr.open_dataset(filename)
        if data.attrs["software_name"] != "GS2":
            raise RuntimeError

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
            energy = raw_data["egrid"].data
        except KeyError:
            energy = raw_data["energy"].data

        # pitch coords
        pitch = raw_data["lambda"].data

        # moment coords
        moment = ["particle", "energy", "momentum"]

        # field coords
        # If fphi > 0.0, phi is set. Similar for apar and bpar
        gs2_knobs = gk_input.data["knobs"]
        fields = ["phi", "apar", "bpar"]
        fields = [field for field in fields if gs2_knobs.get(f"f{field}",0.0) > 0.0]

        # species coords
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
                "nfield": len(fields),
                "nspecies": len(species)
            }
        )

    @staticmethod
    def _set_fields(data: xr.Dataset, raw_data: xr.Dataset) -> xr.Dataset:
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
        field_names = ["phi_t", "apar_t", "bpar_t"]
        if not np.any( np.isin(field_names, raw_data.data_vars)):
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

            # Switch to np.array, ran into bugs when transposing with xarray directly
            raw_field = np.asarray(raw_data[field_name]) * sqrt2
            # raw_field has coords (t,ky,kx,theta,real/imag).
            # We wish to transpose that to (real/imag,theta,kx,ky,t)
            field_data = raw_field.transpose()

            fields[ifield, :, :, :, :] = (
                field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]
            )

        # Shift kx=0 to middle of axis
        fields = np.fft.fftshift(fields, axes=2)

        data["fields"] = (coords, fields)
        return data

    @staticmethod
    def _set_fluxes(data: xr.Dataset, raw_data: xr.Dataset) -> xr.Dataset:
        """
        Set flux data over time.
        The flux coordinates should be (species, moment, field, ky, time)

        For GS2 to print fluxes, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also set the following in  
        gs2_diagnostics_knobs:
        - write_fluxes = .true. (default if nonlinear)
        - write_fluxes_by_mode = .true. (default if nonlinear)
        """
        #TODO What should be set to get GS2 to output 'es_part_by_k' etc?
        fields = ["es", "apar", "bpar"]
        moments = ["part", "heat", "mom"]
        coords = ["species", "moment", "field", "ky", "time"]

        # Check to see if there's anything to do
        flux_names = [
            f"{x}_{y}_{z}" for x in fields for y in moments for z in ("flux", "by_k")
        ]
        if not np.any( np.isin(flux_names, raw_data.data_vars)):
            return data

        fluxes = np.empty([data.dims[coord] for coord in coords])

        for idx in product(enumerate(fields), enumerate(moments)):
            (ifield, field), (imoment, moment) = idx

            by_k_key = f"{field}_{moment}_by_k"
            flux_key = f"{field}_{moment}_flux"

            if by_k_key in raw_data.data_vars:
                # Sum over kx
                flux = np.sum(raw_data[by_k_key], axis=-1)
                flux = flux.transpose((1,2,0))
                # Divide non-zonal components by 2 due to reality condition
                flux[:, 1:, :] *= 0.5
            elif flux_key in raw_data.data_vars:
                # coordinates from raw are (t,species)
                # convert to (species, ky, t)
                # GS2 output is averaged over ky, so we can broadcast that dim
                flux = np.swapaxes(raw_data[flux_key], 0, 1)
                flux = flux.data[:, np.newaxis, :]
            else:
                logging.warning(
                    f"Flux data {flux_key}/{by_k_key} not written to netCDF file. "
                    "Setting this flux to 0."
                )
                flux = 0

            fluxes[:, imoment, ifield, :, :] = flux

        data["fluxes"] = (("species", "moment", "field", "ky", "time"), fluxes)
        return data
