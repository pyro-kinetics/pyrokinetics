import numpy as np
import xarray as xr
import logging
from itertools import product

from .GKOutputReader import GKOutputReader
from .GKInputGS2 import GKInputGS2
from ..constants import sqrt2
from ..typing import PathLike


class GKOutputReaderGS2(GKOutputReader):
    def read(self, filename: PathLike) -> xr.Dataset:
        self.raw_data = xr.open_dataset(filename)
        # Read input file from netcdf, store as GKInputGS2
        input_file = self.raw_data["input_file"]
        if input_file.shape == ():
            # New diagnostics, input file stored as bytes
            # - Stored within numpy 0D array, use [()] syntax to extract
            # - Convert bytes to str by decoding
            # - \n is represented as character literals '\' 'n'. Replace with '\n'.
            self.input_str = input_file.data[()].decode("utf-8").replace(r"\n", "\n")
        else:
            # Old diagnostics (and eventually the single merged diagnostics)
            # input file stored as array of bytes
            self.input_str = "\n".join((line.decode("utf-8") for line in input_file.data))

        self.input = GKInputGS2()
        self.input.read_str(self.input_str)
        
        # Set components of self.data
        self._set_grids()  # Adds coordinates
        self._set_fields() # Adds fields over time, not activated by default with GS2
        self._set_fluxes() # Adds fluxes
        if self.input.is_nonlinear():
            self._set_eigenvalues()
            self._set_eigenfunctions()
        return self.data

    def verify(self, filename: PathLike):
        data = xr.open_dataset(filename)
        if data.attrs["software_name"] != "GS2":
            raise RuntimeError

    def _set_grids(self):
        """
        Loads GS2 grids to GKOutput
        """
        # ky coords
        ky = self.raw_data["ky"][:] / sqrt2

        # time coords
        time_divisor = sqrt2
        try:
            if self.input.data["knobs"]["wstar_units"]:
                time_divisor = ky
        except KeyError:
            pass
        time = self.raw_data["t"][:] / time_divisor

        # kx coords
        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(self.raw_data["kx"][:]) / sqrt2

        # theta coords
        theta = self.raw_data["theta"][:]

        # energy coords
        try:
            energy = self.raw_data["egrid"][:]
        except KeyError:
            energy = self.raw_data["energy"][:]

        # pitch coords
        pitch = self.raw_data["lambda"][:]

        # moment coords
        moment = ["particle", "energy", "momentum"]

        # field coords
        # If fphi > 0.0, phi is set. Similar for apar and bpar
        gs2_knobs = self.input.data["knobs"]
        fields = ["phi", "apar", "bpar"]
        fields = [field for field in fields if gs2_knobs.get(f"f{field}",0.0) > 0.0]

        # species coords
        species = []
        ion_num = 0
        for idx in range(self.input.data["species_knobs"]["nspec"]):
            if self.input.data[f"species_parameters_{idx+1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")


        # Store grid data as xarray DataSet
        self.data = xr.Dataset(
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
                "input_file": self.input_str,
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

    def _set_fields(self):
        """
        Loads 3D fields into GKOutput.data DataSet
        gk_output.data['fields'] = fields(field, theta, kx, ky, time)
        """

        # For GS2 to print fields, we must have fphi, fapar and fbpar set to 1.0
        # Each field must also be individually told to print using:
        # - write_phi_over_time = .true.
        # - write_apar_over_time = .true.
        # - write_bpar_over_time = .true.
        # write_fields must also be .true., or else these params are ignored.

        # Check to see if there's anything to do
        field_names = ["phi_t", "apar_t", "bpar_t"]
        if not np.any( np.isin(field_names, self.raw_data)):
            return

        coords = ["field", "theta", "kx", "ky", "time"]
        fields = np.empty([self.data.dims[coord] for coord in coords], dtype=np.complex)

        # Loop through all fields and add field if it exists
        for ifield, field_name in enumerate(field_names):

            if field_name not in self.raw_data:
                logging.warning(
                    f"Field data over time {field_name} not written to netCDF file. "
                    "Setting this field to 0"
                )
                # Note: we could instead set this to the field at the last time step
                fields[ifield, :, :, :, :] = 0
                continue

            # Switch to np.array, bugs found when transposing with xarray directly
            raw_field = np.asarray(self.raw_data[field_name]) * sqrt2
            # raw_field has coords (kx,ky,theta,t,real/imag).
            # We wish to transpose that to (real/imag,theta,kx,ky,t)
            field_data = raw_field.transpose([4,2,0,1,3])

            fields[ifield, :, :, :, :] = (
                field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]
            )

        # Shift kx=0 to middle of axis
        fields = np.fft.fftshift(fields, axes=2)

        self.data["fields"] = (coords, fields)

    def _set_fluxes(self):
        """
        Loads fluxes into GKOutput.data DataSet
        pyro.gk_output.data['fluxes'] = fluxes(species, moment, field, ky, time)
        """
        is_nonlinear = self.input.is_nonlinear()
        fields = ["es", "apar", "bpar"]
        if is_nonlinear:
            # FIXME I can't figure out how to get GS2 to output these for testing
            moments = ["part_by_k", "heat_by_k", "mom_by_k"]
        else:
            moments = ["part_flux", "heat_flux", "mom_flux"]
        coords = ["species", "moment", "field", "ky", "time"]

        fluxes = np.empty([self.data.dims[coord] for coord in coords])

        for idx in product(enumerate(fields), enumerate(moments)):
            (ifield, field), (imoment, moment) = idx

            key = f"{field}_{moment}"
            if key not in self.raw_data.data_vars:
                logging.warning(
                    f"Flux data {key} not written to netCDF file, setting fluxes to 0"
                )
                fluxes[:, imoment, ifield, :, :] = 0
                continue

            if is_nonlinear:
                # Sum over kx
                flux = np.sum(self.raw_data[key], axis=-1)
                flux = flux.transpose((1,2,0))

                # Divide non-zonal components by 2 due to reality condition
                flux[:, 1:, :] *= 0.5

            else:
                # coordinates from raw are (t,species)
                # convert to (species, ky, t)
                # GS2 output is averaged over ky, so we can simply broadcast that dim
                flux = np.swapaxes(self.raw_data[key], 0, 1)
                flux = flux[:, np.newaxis, :]

            fluxes[:, imoment, ifield, :, :] = flux

        self.data["fluxes"] = (("species", "moment", "field", "ky", "time"), fluxes)
