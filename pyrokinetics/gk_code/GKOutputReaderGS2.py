from itertools import product
from typing import Any, Dict, Tuple
from pathlib import Path
import warnings

import numpy as np
import xarray as xr

from .gk_output import GKOutput, Coords, Fields, Fluxes, Moments, Eigenvalues
from .GKInputGS2 import GKInputGS2
from ..typing import PathLike
from ..readers import Reader
from ..normalisation import SimulationNormalisation


@GKOutput.reader("GS2")
class GKOutputReaderGS2(Reader):
    def read(
        self,
        filename: PathLike,
        norm: SimulationNormalisation,
        downsize: int = 1,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
    ) -> GKOutput:
        raw_data, gk_input, input_str = self._get_raw_data(filename)
        coords = self._get_coords(raw_data, gk_input, downsize)
        fields = self._get_fields(raw_data) if load_fields else None
        fluxes = self._get_fluxes(raw_data, gk_input, coords) if load_fluxes else None
        moments = (
            self._get_moments(raw_data, gk_input, coords) if load_moments else None
        )

        if fields or coords["linear"]:
            # Rely on gk_output to generate eigenvalues
            eigenvalues = None
        else:
            eigenvalues = self._get_eigenvalues(raw_data, coords["time_divisor"])

        # Assign units and return GKOutput
        convention = norm.gs2
        field_dims = ("theta", "kx", "ky", "time")
        flux_dims = (("field", "species", "ky", "time"),)
        moment_dims = (("field", "species", "ky", "time"),)
        return GKOutput(
            coords=Coords(
                time=coords["time"],
                kx=coords["kx"],
                ky=coords["ky"],
                theta=coords["theta"],
                pitch=coords["pitch"],
                energy=coords["energy"],
                species=coords["species"],
            ).with_units(convention),
            norm=norm,
            fields=Fields(**fields, dims=field_dims).with_units(convention)
            if fields
            else None,
            fluxes=Fluxes(**fluxes, dims=flux_dims).with_units(convention)
            if fluxes
            else None,
            moments=Moments(**moments, dims=moment_dims).with_units(convention)
            if moments
            else None,
            eigenvalues=Eigenvalues(**eigenvalues).with_units(convention)
            if eigenvalues
            else None,
            linear=coords["linear"],
            gk_code="GS2",
            input_file=input_str,
            normalise_flux_moment=True,
        )

    def verify(self, filename: PathLike):
        try:
            warnings.filterwarnings("error")
            data = xr.open_dataset(filename)
        except RuntimeWarning:
            warnings.resetwarnings()
            raise RuntimeError
        warnings.resetwarnings()

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
    def _get_coords(
        raw_data: xr.Dataset, gk_input: GKInputGS2, downsize: int
    ) -> Dict[str, Any]:
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

        # moment coords
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]

        # field coords
        # If fphi/fapar/fbpar not in 'knobs', or they equal zero, skip the field
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
            if gk_input.data[f"species_parameters_{idx + 1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "linear": gk_input.is_linear(),
            "time_divisor": time_divisor,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "downsize": downsize,
        }

    @staticmethod
    def _get_fields(raw_data: xr.Dataset) -> Dict[str, np.ndarray]:
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
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputGS2,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

    @staticmethod
    def _get_fluxes(
        raw_data: xr.Dataset,
        gk_input: GKInputGS2,
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
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
        fluxes_dict = {"particle": "part", "heat": "heat", "momentum": "mom"}

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

        coord_names = ["flux", "field", "species", "ky", "time"]
        fluxes = np.zeros([len(coords[name]) for name in coord_names])

        for (ifield, (field, gs2_field)), (iflux, gs2_flux) in product(
            enumerate(fields.items()), enumerate(fluxes_dict.values())
        ):
            flux_key = f"{gs2_field}_{gs2_flux}_flux"
            # old diagnostics
            by_k_key = f"{gs2_field}_{gs2_flux}_by_k"
            # new diagnostics
            by_mode_key = f"{gs2_field}_{gs2_flux}_flux_by_mode"

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

            fluxes[iflux, ifield, ...] = flux

        for iflux, flux in enumerate(coords["flux"]):
            if not np.all(fluxes[iflux, ...] == 0):
                results[flux] = fluxes[iflux, ...]

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: xr.Dataset, time_divisor: float
    ) -> Dict[str, np.ndarray]:
        # should only be called if no field data were found
        mode_frequency = raw_data.omega_average.isel(ri=0).transpose("kx", "ky", "time")
        growth_rate = raw_data.omega_average.isel(ri=1).transpose("kx", "ky", "time")
        return {
            "mode_frequency": mode_frequency.data / time_divisor,
            "growth_rate": growth_rate.data / time_divisor,
        }
