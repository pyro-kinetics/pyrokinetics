from itertools import product
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from ast import literal_eval

from .gk_output import (
    GKOutput,
    get_flux_units,
    get_field_units,
    get_coord_units,
    get_eigenvalues_units,
    FieldDict,
    FluxDict,
)
from .GKInputGS2 import GKInputGS2
from ..typing import PathLike
from ..readers import Reader
from ..normalisation import SimulationNormalisation, ureg


@GKOutput.reader("GS2")
class GKOutputReaderGS2(Reader):
    def read(
        self, filename: PathLike, norm: SimulationNormalisation, downsize: int = 1
    ) -> GKOutput:
        raw_data, gk_input, input_str = self._get_raw_data(filename)
        coords = self._get_coords(raw_data, gk_input, downsize)
        fields = self._get_fields(raw_data)
        fluxes = self._get_fluxes(raw_data, gk_input, coords)

        # Assign units and return GKOutput
        convention = norm.gs2
        coord_units = get_coord_units(convention)
        field_units = get_field_units(convention)
        flux_units = get_flux_units(convention)
        eig_units = get_eigenvalues_units(convention)

        for field_name, field in fields.items():
            fields[field_name] = field * field_units[field_name]

        for flux_type, flux in fluxes.items():
            fluxes[flux_type] = flux * flux_units[flux_type]

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
            field_dim=coords["field"],
            moment=coords["moment"],
            species=coords["species"],
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
        moment = ["particle", "heat", "momentum"]

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
            "moment": moment,
            "species": species,
            "downsize": downsize,
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
    def _get_fluxes(
        raw_data: xr.Dataset,
        gk_input: GKInputGS2,
        coords: Dict,
    ) -> FluxDict:
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

        coord_names = ["moment", "field", "species", "ky", "time"]
        fluxes = np.zeros([len(coords[name]) for name in coord_names])

        for (ifield, (field, gs2_field)), (imoment, (moment, gs2_moment)) in product(
            enumerate(fields.items()), enumerate(moments.items())
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

            fluxes[imoment, ifield, ...] = flux

        for imoment, moment in enumerate(moments):
            if not np.all(fluxes[imoment, ...] == 0):
                results[moment] = fluxes[imoment, ...]

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

    @staticmethod
    def to_netcdf(self, *args, **kwargs) -> None:
        """Writes self.data to disk. Forwards all args to xarray.Dataset.to_netcdf."""
        data = self.data.expand_dims('ReIm', axis=-1)  # Add ReIm axis at the end
        data = xr.concat([data.real, data.imag], dim='ReIm')

        data.pint.dequantify().to_netcdf(*args, **kwargs)

    @staticmethod
    def from_netcdf(
        path: PathLike,
        *args,
        overwrite_metadata: bool = False,
        overwrite_title: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialise self.data from a netCDF file.

        Parameters
        ----------

        path: PathLike
            Path to the netCDF file on disk.
        *args:
            Positional arguments forwarded to xarray.open_dataset.
        overwrite_metadata: bool, default False
            Take ownership of the netCDF data, overwriting attributes such as 'title',
            'software_name', 'date_created', etc.
        overwrite_title: Optional[str]
            If ``overwrite_metadata`` is ``True``, this is used to set the ``title``
            attribute in ``self.data``. If unset, the derived class name is used.
        **kwargs:
            Keyword arguments forwarded to xarray.open_dataset.

        Returns
        -------
        Derived
            Instance of a derived class with self.data initialised. Derived classes
            which need to do more than this should override this method with their
            own implementation.
        """
        instance = GKOutput.__new__(GKOutput)

        with xr.open_dataset(Path(path), *args, **kwargs) as dataset:
            if overwrite_metadata:
                if overwrite_title is None:
                    title = GKOutput.__name__
                else:
                    title = str(overwrite_title)
                for key, val in GKOutput._metadata(title).items():
                    dataset.attrs[key] = val
            instance.data = dataset

        # Set up attr_units
        attr_units_as_str = literal_eval(dataset.attribute_units)
        instance._attr_units = {k: ureg(v).units for k, v in attr_units_as_str.items()}
        attrs = instance.attrs

        # isel drops attrs so need to add back in
        instance.data = instance.data.isel(ReIm=0) + 1j * instance.data.isel(ReIm=1)
        instance.data.attrs = attrs

        return instance


