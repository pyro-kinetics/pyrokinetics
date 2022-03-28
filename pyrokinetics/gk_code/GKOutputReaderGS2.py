import numpy as np
import xarray as xr
import logging

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
            input_str = input_file.data[()].decode("utf-8").replace(r"\n", "\n")
        else:
            # Old diagnostics (and eventually the single merged diagnostics)
            # input file stored as array of bytes
            input_str = "\n".join((line.decode("utf-8") for line in input_file.data))

        self.input = GKInputGS2()
        self.input.read_str(input_str)
        # Set components of self
        self.set_grids()
        self.set_fields()
        self.set_fluxes()
        if self.input.is_nonlinear():
            self.set_eigenvalues()
            self.set_eigenfunctions()
        return self.data

    def verify(self, filename: PathLike):
        data = xr.open_dataset(filename)
        if data.attrs["software_name"] != "GS2":
            raise RuntimeError

    def set_grids(self):
        """
        Loads GS2 grids to GKOutput
        """
        ky = self.raw_data["ky"][:] / sqrt2
        self.ky = ky
        self.nky = len(ky)

        try:
            if self.input.data["knobs"]["wstar_units"]:
                time = self.raw_data["t"][:] / ky
            else:
                time = self.raw_data["t"][:] / sqrt2

        except KeyError:
            time = self.raw_data["t"][:] / sqrt2

        self.time = time
        self.ntime = len(time)

        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(self.raw_data["kx"][:]) / sqrt2

        self.kx = kx
        self.nkx = len(kx)

        nspecies = self.raw_data.dims["species"]
        self.nspecies = nspecies

        theta = self.raw_data["theta"][:]
        self.theta = theta
        self.ntheta = len(theta)

        try:
            energy = self.raw_data["egrid"][:]
        except KeyError:
            energy = self.raw_data["energy"][:]

        self.energy = energy
        self.nenergy = len(energy)

        pitch = self.raw_data["lambda"][:]
        self.pitch = pitch
        self.npitch = len(pitch)

        gs2_knobs = self.input.data["knobs"]
        nfield = 0
        if gs2_knobs["fphi"] > 0.0:
            nfield += 1
        if gs2_knobs["fapar"] > 0.0:
            nfield += 1
        if gs2_knobs["fbpar"] > 0.0:
            nfield += 1

        self.nfield = nfield

        field = ["phi", "apar", "bpar"]
        field = field[:nfield]

        moment = ["particle", "energy", "momentum"]

        # get species names
        species = []
        ion_num = 0
        for idx in range(self.input.data["species_knobs"]["nspec"]):
            if self.input.data[f"species_parameters_{idx+1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        # Store grid data as xarray DataSet
        ds = xr.Dataset(
            coords={
                "time": time,
                "field": field,
                "moment": moment,
                "species": species,
                "kx": kx,
                "ky": ky,
                "theta": theta,
            }
        )

        self.data = ds

    def set_fields(self):
        """
        Loads 3D fields into GKOutput.data DataSet
        gk_output.data['fields'] = fields(field, theta, kx, ky, time)
        """
        fields = np.empty(
            (
                self.nfield,
                self.nkx,
                self.ntheta,
                self.nky,
                self.ntime,
            ),
            dtype=np.complex,
        )

        field_appendices = ["phi_t", "apar_t", "bpar_t"]

        # Loop through all fields and add field in it exists
        # FIXME This assumes there are three fields, but load_grids did not.
        #       None of these field_appendices are in the example output file I've used.
        for ifield, field_appendix in enumerate(field_appendices):

            raw_field = self.raw_data[field_appendix][:] * sqrt2
            field_data = np.moveaxis(raw_field, [0, 1, 2, 3, 4], [4, 3, 1, 2, 0])

            fields[ifield, :, :, :, :] = (
                field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]
            )

        # Shift kx=0 to middle of axis
        fields = np.fft.fftshift(fields, axes=1)

        # FIXME kx, theta, ky here. theta, kx, ky in the docstring. Which is it?
        self.data["fields"] = (("field", "kx", "theta", "ky", "time"), fields)

    def set_fluxes(self):
        """
        Loads fluxes into GKOutput.data DataSet
        pyro.gk_output.data['fluxes'] = fluxes(species, moment, field, ky, time)
        """
        nonlinear = self.input.is_nonlinear()

        field_keys = ["es", "apar", "bpar"]
        if nonlinear:
            moment_keys = ["part_by_k", "heat_by_k", "mom_by_k"]
        else:
            moment_keys = ["part_flux", "heat_flux", "mom_flux"]

        fluxes = np.empty((self.nspecies, 3, self.nfield, self.nky, self.ntime))

        if f"{field_keys[0]}_{moment_keys[0]}" not in self.raw_data.data_vars:
            logging.warning("Flux data not written to netCDF file, setting fluxes to 0")

        else:
            for ifield, field in enumerate(field_keys):
                for imoment, moment in enumerate(moment_keys):
                    key = f"{field}_{moment}"

                    if nonlinear:
                        # Sum over kx
                        flux = np.sum(self.raw_data[key], axis=-1)
                        flux = np.moveaxis(flux, [1, 2, 0], [0, 1, 2])

                        # Divide non-zonal components by 2 due to reality condition
                        flux[:, 1:, :] *= 0.5

                    else:
                        flux = np.swapaxes(self.raw_data[key], 0, 1)
                        flux = flux[:, np.newaxis, :]

                    fluxes[:, imoment, ifield, :, :] = flux

        self.data["fluxes"] = (("species", "moment", "field", "ky", "time"), fluxes)
