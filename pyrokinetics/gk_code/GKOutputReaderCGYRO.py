import numpy as np
import pint  # noqa
import pint_xarray  # noqa
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import xarray as xr
from ast import literal_eval

from .gk_output import (
    GKOutput,
    get_flux_units,
    get_field_units,
    get_coord_units,
    get_eigenvalues_units,
    get_eigenfunctions_units,
    get_moment_units,
    FieldDict,
    FluxDict,
    MomentDict,
)
from .GKInputCGYRO import GKInputCGYRO
from ..constants import pi
from ..typing import PathLike

from ..readers import Reader
from ..normalisation import SimulationNormalisation, ureg


class CGYROFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


@GKOutput.reader("CGYRO")
class GKOutputReaderCGYRO(Reader):
    fields = ["phi", "apar", "bpar"]
    moments = ["n", "e", "v"]

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
        if load_fields:
            fields = self._get_fields(raw_data, gk_input, coords)
        else:
            fields = {}

        if load_fluxes:
            fluxes = self._get_fluxes(raw_data, coords)
        else:
            fluxes = {}

        if load_moments:
            moments = self._get_moments(raw_data, gk_input, coords)
        else:
            moments = {}

        # Assign units and return GKOutput
        convention = norm.cgyro
        coord_units = get_coord_units(convention)
        field_units = get_field_units(convention)
        moments_units = get_moment_units(convention)
        flux_units = get_flux_units(convention)
        eig_units = get_eigenvalues_units(convention)
        eigfunc_units = get_eigenfunctions_units(convention)

        for field_name, field in fields.items():
            fields[field_name] = field * field_units[field_name]

        for moment_name, moment in moments.items():
            moments[moment_name] = moment * moments_units[moment_name]

        for flux_type, flux in fluxes.items():
            fluxes[flux_type] = flux * flux_units[flux_type]

        if coords["linear"] and (
            coords["ntheta_plot"] != coords["ntheta_grid"] or not fields
        ):
            eigenvalues = self._get_eigenvalues(raw_data, coords, gk_input)
            growth_rate = eigenvalues["growth_rate"] * eig_units["growth_rate"]
            mode_frequency = eigenvalues["mode_frequency"] * eig_units["mode_frequency"]
            eigenfunctions = (
                self._get_eigenfunctions(raw_data, coords)
                * eigfunc_units["eigenfunctions"]
            )

        else:
            # Rely on gk_output to generate eigenvalues
            growth_rate = None
            mode_frequency = None
            eigenfunctions = None

        return GKOutput(
            time=coords["time"] * coord_units["time"],
            kx=coords["kx"] * coord_units["kx"],
            ky=coords["ky"] * coord_units["ky"],
            theta=coords["theta"] * coord_units["theta"],
            pitch=coords["pitch"] * coord_units["pitch"],
            energy=coords["energy"] * coord_units["energy"],
            field_dim=coords["field"],
            flux_dim=coords["flux"],
            moment_dim=coords["moment"],
            species=coords["species"],
            fields=fields,
            fluxes=fluxes,
            moments=moments,
            norm=norm,
            linear=coords["linear"],
            gk_code="CGYRO",
            input_file=input_str,
            growth_rate=growth_rate,
            mode_frequency=mode_frequency,
            eigenfunctions=eigenfunctions,
        )

    def verify(self, dirname: PathLike):
        dirname = Path(dirname)
        for f in self._required_files(dirname).values():
            if not f.path.exists():
                raise RuntimeError

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        For CGYRO, simply returns dir of the path.
        """
        return Path(filename).parent

    @staticmethod
    def _required_files(dirname: PathLike):
        dirname = Path(dirname)
        return {
            "input": CGYROFile(dirname / "input.cgyro", required=True),
            "time": CGYROFile(dirname / "out.cgyro.time", required=True),
            "grids": CGYROFile(dirname / "out.cgyro.grids", required=True),
            "equilibrium": CGYROFile(dirname / "out.cgyro.equilibrium", required=True),
        }

    @classmethod
    def _get_raw_data(
        cls, dirname: PathLike
    ) -> Tuple[Dict[str, Any], GKInputCGYRO, str]:
        dirname = Path(dirname)
        if not dirname.exists():
            raise RuntimeError(
                f"GKOutputReaderCGYRO: Provided path {dirname} does not exist. "
                "Please supply the name of a directory containing CGYRO output files."
            )
        if not dirname.is_dir():
            raise RuntimeError(
                f"GKOutputReaderCGYRO: Provided path {dirname} is not a directory. "
                "Please supply the name of a directory containing CGYRO output files."
            )

        # The following list of CGYRO files may exist
        expected_files = {
            **cls._required_files(dirname),
            "flux": CGYROFile(dirname / "bin.cgyro.ky_flux", required=False),
            "cflux": CGYROFile(dirname / "bin.cgyro.ky_cflux", required=False),
            "eigenvalues_bin": CGYROFile(dirname / "bin.cgyro.freq", required=False),
            "eigenvalues_out": CGYROFile(dirname / "out.cgyro.freq", required=False),
            **{
                f"field_{f}": CGYROFile(dirname / f"bin.cgyro.kxky_{f}", required=False)
                for f in cls.fields
            },
            **{
                f"moment_{m}": CGYROFile(
                    dirname / f"bin.cgyro.kxky_{m}", required=False
                )
                for m in cls.moments
            },
            **{
                f"eigenfunctions_{f}": CGYROFile(
                    dirname / f"bin.cgyro.{f}b", required=False
                )
                for f in cls.fields
            },
        }
        # Read in files
        raw_data = {}
        for key, cgyro_file in expected_files.items():
            if not cgyro_file.path.exists():
                if cgyro_file.required:
                    raise RuntimeError(
                        f"GKOutputReaderCGYRO: The file {cgyro_file.path.name} is needed"
                    )
                continue
            # Read in file according to format
            if cgyro_file.fmt == "input":
                with open(cgyro_file.path, "r") as f:
                    raw_data[key] = f.read()
            if cgyro_file.fmt == "out":
                raw_data[key] = np.loadtxt(cgyro_file.path)
            if cgyro_file.fmt == "bin":
                raw_data[key] = np.fromfile(cgyro_file.path, dtype="float32")
        input_str = raw_data["input"]
        gk_input = GKInputCGYRO()
        gk_input.read_str(input_str)
        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: Dict[str, Any], gk_input: GKInputCGYRO, downsize: int = 1
    ) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of CGYRO
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing CGYRO output.
            gk_input (GKInputCGYRO): Processed CGYRO input file.

        Returns:
            Dict:  Dictionary with coords
        """
        bunit_over_b0 = gk_input.get_local_geometry().bunit_over_b0

        # Process time data
        time = raw_data["time"][:, 0]

        # Process grid data
        grid_data = raw_data["grids"]
        nky = int(grid_data[0])
        nspecies = int(grid_data[1])
        nfield = int(grid_data[2])
        nkx = int(grid_data[3])
        ntheta_grid = int(grid_data[4])
        nenergy = int(grid_data[5])
        npitch = int(grid_data[6])
        box_size = int(grid_data[7])
        length_x = grid_data[8]
        ntheta_plot = int(grid_data[10])

        # Iterate through grid_data in chunks, starting after kx
        pos = 11 + nkx

        theta_grid = grid_data[pos : pos + ntheta_grid]
        pos += ntheta_grid

        energy = grid_data[pos : pos + nenergy]
        pos += nenergy

        pitch = grid_data[pos : pos + npitch]
        pos += npitch

        ntheta_ballooning = ntheta_grid * int(nkx / box_size)
        theta_ballooning = grid_data[pos : pos + ntheta_ballooning]
        pos += ntheta_ballooning

        ky = grid_data[pos : pos + nky] / bunit_over_b0

        if gk_input.is_linear():
            # Convert to ballooning co-ordinate so only 1 kx
            theta = theta_ballooning
            ntheta = ntheta_ballooning
            kx = [0.0]
            nkx = 1
        else:
            # Output data actually given on theta_plot grid
            ntheta = ntheta_plot
            theta = [0.0] if ntheta == 1 else theta_grid[:: ntheta_grid // ntheta]
            kx = (
                2
                * pi
                * np.linspace(-int(nkx / 2), int((nkx + 1) / 2) - 1, nkx)
                / length_x
            ) / bunit_over_b0

        # Get rho_star from equilibrium file
        if len(raw_data["equilibrium"]) == 54 + 7 * nspecies:
            rho_star = raw_data["equilibrium"][35]
        else:
            rho_star = raw_data["equilibrium"][23]

        fields = ["phi", "apar", "bpar"][:nfield]
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]
        species = gk_input.get_local_species().names
        if nspecies != len(species):
            raise RuntimeError(
                "GKOutputReaderCGYRO: Different number of species in input and output."
            )

        # Store grid data as xarray DataSet
        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "ntheta_plot": ntheta_plot,
            "ntheta_grid": ntheta_grid,
            "nradial": int(gk_input.data["N_RADIAL"]),
            "rho_star": rho_star,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "linear": gk_input.is_linear(),
            "downsize": downsize,
        }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, Any],
        gk_input: GKInputCGYRO,
        coords: Dict[str, Any],
    ) -> FieldDict:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """
        field_names = ("phi", "apar", "bpar")

        nkx = len(coords["kx"])
        nradial = coords["nradial"]
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntheta_plot = coords["ntheta_plot"]
        ntheta_grid = coords["ntheta_grid"]
        ntime = len(coords["time"])

        raw_field_data = {f: raw_data.get(f"field_{f}", None) for f in field_names}

        results = {}

        # Check to see if there's anything to do
        if not raw_field_data:
            return results

        # Loop through all fields and add field in if it exists
        for ifield, (field_name, raw_field) in enumerate(raw_field_data.items()):
            if raw_field is None:
                logging.warning(
                    f"Field data {field_name} over time not found, expected the file "
                    f"bin.cygro.kxky_{field_name} to exist. Setting this field to 0."
                )
                continue

            # If linear, convert from kx to ballooning space.
            # Use nradial instead of nkx, ntheta_plot instead of ntheta
            if gk_input.is_linear():
                shape = (2, nradial, ntheta_plot, nky, ntime)
            else:
                shape = (2, nkx, ntheta, nky, ntime)

            field_data = raw_field[: np.prod(shape)].reshape(shape, order="F")
            # Adjust sign to match pyrokinetics frequency convention
            # (-ve is electron direction)
            mode_sign = -np.sign(
                np.sign(gk_input.data.get("Q", 2.0)) * -gk_input.data.get("BTCCW", -1)
            )

            field_data = (field_data[0] + mode_sign * 1j * field_data[1]) / coords[
                "rho_star"
            ]

            # If nonlinear, we can simply save the fields and continue
            if gk_input.is_nonlinear():
                fields = field_data.swapaxes(0, 1)
            else:
                # If theta_plot != theta_grid, we get eigenfunction data and multiply by the
                # field amplitude
                if ntheta_plot != ntheta_grid:
                    # Get eigenfunction data
                    raw_eig_data = raw_data.get(f"eigenfunctions_{field_name}", None)
                    if raw_eig_data is None:
                        logging.warning(
                            f"When setting fields, eigenfunction data for {field_name} not "
                            f"found, expected the file bin.cygro.{field_name}b to exist. "
                            f"Not setting the field {field_name}."
                        )
                        continue
                    eig_shape = [2, ntheta, ntime]
                    eig_data = raw_eig_data[: np.prod(eig_shape)].reshape(
                        eig_shape, order="F"
                    )
                    eig_data = eig_data[0] + 1j * eig_data[1]
                    # Get field amplitude
                    middle_kx = (nradial // 2) + 1
                    field_amplitude = np.abs(field_data[middle_kx, 0, 0, :])
                    # Multiply together
                    # FIXME We only set kx=ky=0 here, any other values are left undefined
                    #       as fields is created using np.empty. Should we instead set
                    #       all kx and ky to these values? Should we expect that nx=ny=1?
                    field_data = np.reshape(
                        eig_data * field_amplitude, (nradial, ntheta_grid, nky, ntime)
                    )

                # Poisson Sum (no negative in exponent to match frequency convention)
                q = gk_input.get_local_geometry_miller().q
                for i_radial in range(nradial):
                    nx = -nradial // 2 + (i_radial - 1)
                    field_data[i_radial, ...] *= np.exp(2j * pi * nx * q)

                fields = field_data.reshape([ntheta, nkx, nky, ntime])

            results[field_name] = fields

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputCGYRO,
        coords: Dict[str, Any],
    ) -> MomentDict:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        moment_names = {"n": "density", "e": "temperature", "v": "velocity"}

        nkx = len(coords["kx"])
        nradial = coords["nradial"]
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntheta_plot = coords["ntheta_plot"]
        ntime = len(coords["time"])
        nspec = len(coords["species"])

        raw_moment_data = {
            value: raw_data.get(f"moment_{key}", None)
            for key, value in moment_names.items()
        }
        results = {}

        # Check to see if there's anything to do
        if not raw_moment_data:
            return results

        # Loop through all moments and add moment in if it exists
        for imoment, (moment_name, raw_moment) in enumerate(raw_moment_data.items()):
            if raw_moment is None:
                logging.warning(
                    f"moment data {moment_name} over time not found, expected the file "
                    f"bin.cygro.kxky_{moment_name} to exist. Setting this moment to 0."
                )
                continue

            # If linear, convert from kx to ballooning space.
            # Use nradial instead of nkx, ntheta_plot instead of ntheta
            if gk_input.is_linear():
                shape = (2, nradial, ntheta_plot, nspec, nky, ntime)
            else:
                shape = (2, nkx, ntheta, nspec, nky, ntime)

            moment_data = raw_moment[: np.prod(shape)].reshape(shape, order="F")
            # Adjust sign to match pyrokinetics frequency convention
            # (-ve is electron direction)
            mode_sign = -np.sign(
                np.sign(gk_input.data.get("Q", 2.0)) * -gk_input.data.get("BTCCW", -1)
            )

            moment_data = (moment_data[0] + mode_sign * 1j * moment_data[1]) / coords[
                "rho_star"
            ]

            # If nonlinear, we can simply save the moments and continue
            if gk_input.is_nonlinear():
                moments = moment_data.swapaxes(0, 1)
            else:
                # Poisson Sum (no negative in exponent to match frequency convention)
                q = gk_input.get_local_geometry_miller().q
                for i_radial in range(nradial):
                    nx = -nradial // 2 + (i_radial - 1)
                    moment_data[i_radial, ...] *= np.exp(2j * pi * nx * q)

                moments = moment_data.reshape([ntheta, nkx, nspec, nky, ntime])

            results[moment_name] = moments

        temp_spec = np.ones(results["density"].shape)
        for i in range(nspec):
            temp_spec[:, :, i, :, :] = gk_input.data.get(f"TEMP_{i+1}", 1.0)

        # Convert CGYRO energy fluctuation to temperature
        results["temperature"] = (
            2 * results["temperature"] - results["density"] * temp_spec
        )
        return results

    @staticmethod
    def _get_fluxes(
        raw_data: Dict[str, Any],
        coords: Dict,
    ) -> FluxDict:
        """
        Set flux data over time.
        The flux coordinates should be (species, moment, field, ky, time)
        """

        results = {}

        # cflux is more appropriate for CGYRO simulations
        # with GAMMA_E > 0 and SHEAR_METHOD = 2.
        # However, for cross-code consistency, gflux is used for now.
        # if gk_input.data.get("GAMMA_E", 0.0) == 0.0:
        #     flux_key = "flux"
        # else:
        #     flux_key = "cflux"
        flux_key = "flux"

        if flux_key in raw_data:
            coord_names = ["species", "flux", "field", "ky", "time"]
            shape = [len(coords[coord_name]) for coord_name in coord_names]
            fluxes = raw_data[flux_key][: np.prod(shape)].reshape(shape, order="F")

        fluxes = np.swapaxes(fluxes, 0, 2)
        for iflux, flux in enumerate(coords["flux"]):
            results[flux] = fluxes[:, iflux, :, :, :]

        return results

    @classmethod
    def _get_eigenvalues(
        self, raw_data: Dict[str, Any], coords: Dict, gk_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:

        data['eigenvalues'] = eigenvalues(kx, ky, time)
        data['mode_frequency'] = mode_frequency(kx, ky, time)
        data['growth_rate'] = growth_rate(kx, ky, time)

        This should be called after _set_fields, and is only valid for linear runs.
        Unlike the version in the super() class, CGYRO may need to get extra info from
        an eigenvalue file.

        Args:
            data (xr.Dataset): The dataset to be modified.
            dirname (PathLike): Directory containing CGYRO output files.
        Returns:
            Dict: The modified dataset which was passed to 'data'.
        """

        ntime = len(coords["time"])
        nky = len(coords["ky"])
        nkx = len(coords["kx"])
        shape = (2, nky, ntime)

        if "eigenvalues_bin" in raw_data:
            eigenvalue_over_time = raw_data["eigenvalues_bin"][
                : np.prod(shape)
            ].reshape(shape, order="F")
        elif "eigenvalues_out" in raw_data:
            eigenvalue_over_time = (
                raw_data["eigenvalues_out"].transpose()[:, :ntime].reshape(shape)
            )
        else:
            raise RuntimeError(
                "Eigenvalues over time not found, expected the files bin.cgyro.freq or "
                "out.cgyro.freq to exist. Could not set data_vars 'growth_rate', "
                "'mode_frequency' and 'eigenvalue'."
            )
        mode_sign = -np.sign(
            np.sign(gk_input.data.get("Q", 2.0)) * -gk_input.data.get("BTCCW", -1)
        )

        mode_frequency = mode_sign * eigenvalue_over_time[0, :, :]

        growth_rate = eigenvalue_over_time[1, :, :]
        eigenvalue = mode_frequency + 1j * growth_rate
        # Add kx axis for compatibility with GS2 eigenvalues
        # FIXME Is this appropriate? Should we drop the kx coordinate?
        shape_with_kx = (nkx, nky, ntime)
        mode_frequency = np.ones(shape_with_kx) * mode_frequency
        growth_rate = np.ones(shape_with_kx) * growth_rate
        eigenvalue = np.ones(shape_with_kx) * eigenvalue

        result = {
            "growth_rate": growth_rate,
            "mode_frequency": mode_frequency,
            "eigenvalues": eigenvalue,
        }

        return result

    @staticmethod
    def _get_eigenfunctions(raw_data: Dict[str, Any], coords: Dict) -> Dict[str, Any]:
        """
        Loads eigenfunctions into data with the following coordinates:

        data['eigenfunctions'] = eigenfunctions(kx, ky, field, theta, time)

        This should be called after _set_fields, and is only valid for linear runs.
        """

        raw_eig_data = [
            raw_data.get(f"eigenfunctions_{f}", None) for f in coords["field"]
        ]

        ntime = len(coords["time"])
        ntheta = len(coords["theta"])
        nkx = len(coords["kx"])
        nky = len(coords["ky"])

        raw_shape = [2, ntheta, nkx, nky, ntime]

        # FIXME Currently using kx and ky for compatibility with GS2 results, but
        #       these coordinates are not used. Should we remove these coordinates?
        coord_names = ["field", "theta", "kx", "ky", "time"]
        eigenfunctions = np.empty(
            [len(coords[coord_name]) for coord_name in coord_names], dtype=complex
        )

        # Loop through all fields and add eigenfunction if it exists
        for ifield, raw_eigenfunction in enumerate(raw_eig_data):
            if raw_eigenfunction is not None:
                eigenfunction = raw_eigenfunction[: np.prod(raw_shape)].reshape(
                    raw_shape, order="F"
                )
                eigenfunctions[ifield, ...] = eigenfunction[0] + 1j * eigenfunction[1]

        square_fields = np.sum(np.abs(eigenfunctions) ** 2, axis=0)
        field_amplitude = np.sqrt(np.trapz(square_fields, coords["theta"], axis=0)) / (
            2 * np.pi
        )
        result = eigenfunctions / field_amplitude

        return result

    @staticmethod
    def to_netcdf(self, *args, **kwargs) -> None:
        """Writes self.data to disk. Forwards all args to xarray.Dataset.to_netcdf."""
        data = self.data.expand_dims("ReIm", axis=-1)  # Add ReIm axis at the end
        data = xr.concat([data.real, data.imag], dim="ReIm")

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
