import numpy as np
import xarray as xr
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .GKOutputReader import GKOutputReader
from .GKInputCGYRO import GKInputCGYRO
from ..constants import pi
from ..typing import PathLike


class CGYROFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


class GKOutputReaderCGYRO(GKOutputReader):

    fields = ["phi", "apar", "bpar"]

    @staticmethod
    def _required_files(dirname: PathLike):
        dirname = Path(dirname)
        return {
            "input": CGYROFile(dirname / "input.cgyro", required=True),
            "time": CGYROFile(dirname / "out.cgyro.time", required=True),
            "grids": CGYROFile(dirname / "out.cgyro.grids", required=True),
            "equilibrium": CGYROFile(dirname / "out.cgyro.equilibrium", required=True),
        }

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
            "eigenvalues_bin": CGYROFile(dirname / "bin.cgyro.freq", required=False),
            "eigenvalues_out": CGYROFile(dirname / "out.cgyro.freq", required=False),
            **{
                f"field_{f}": CGYROFile(dirname / f"bin.cgyro.kxky_{f}", required=False)
                for f in cls.fields
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

    @classmethod
    def _init_dataset(
        cls, raw_data: Dict[str, Any], gk_input: GKInputCGYRO
    ) -> xr.Dataset:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of CGYRO
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing CGYRO output.
            gk_input (GKInputCGYRO): Processed CGYRO input file.

        Returns:
            xr.Dataset: Dataset with coords and attrs set, but not data_vars
        """

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

        ky = grid_data[pos : pos + nky]

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
            )

        # Get rho_star from equilibrium file
        rho_star = raw_data["equilibrium"][23]

        field = cls.fields[:nfield]
        moment = ["particle", "energy", "momentum"]
        species = gk_input.get_local_species().names
        if nspecies != len(species):
            raise RuntimeError(
                "GKOutputReaderCGYRO: Different number of species in input and output."
            )

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
                "field": field,
                "species": species,
            },
            attrs={
                "ntime": len(time),
                "nkx": nkx,
                "nky": nky,
                "ntheta": ntheta,
                "nenergy": nenergy,
                "npitch": npitch,
                "nmoment": len(moment),
                "nfield": nfield,
                "nspecies": len(species),
                # The following attributes are specific to CGYRO, and are used later
                # to calculate fields/fluxes
                "rho_star": rho_star,
                "ntheta_plot": ntheta_plot,
                "ntheta_grid": ntheta_grid,
                "nradial": int(gk_input.data["N_RADIAL"]),
            },
        )

    @staticmethod
    def _set_fields(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: GKInputCGYRO
    ) -> xr.Dataset:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """
        coords = ["field", "theta", "kx", "ky", "time"]
        fields = np.empty([data.dims[coord] for coord in coords], dtype=complex)

        raw_field_data = {
            f: raw_data.get(f"field_{f}", None) for f in data["field"].data
        }
        # Check to see if there's anything to do
        if not raw_field_data:
            return data

        # Loop through all fields and add field in if it exists
        for ifield, (field_name, raw_field) in enumerate(raw_field_data.items()):
            if raw_field is None:
                logging.warning(
                    f"Field data {field_name} over time not found, expected the file "
                    f"bin.cygro.kxky_{field_name} to exist. Setting this field to 0."
                )
                fields[ifield, :, :, :, :] = 0
                continue

            # If linear, convert from kx to ballooning space.
            # Use nradial instead of nkx, ntheta_plot instead of ntheta
            if gk_input.is_linear():
                shape = (2, data.nradial, data.ntheta_plot, data.nky, data.ntime)
            else:
                shape = (2, data.nkx, data.ntheta, data.nky, data.ntime)
            field_data = raw_field[: np.prod(shape)].reshape(shape, order="F")
            # Using -1j here to match pyrokinetics frequency convention
            # (-ve is electron direction)
            field_data = (field_data[0] - 1j * field_data[1]) / data.rho_star

            # If nonlinear, we can simply save the fields and continue
            if gk_input.is_nonlinear():
                fields[ifield, :, :, :, :] = field_data.swapaxes(0, 1)
                continue

            # If theta_plot != theta_grid, we get eigenfunction data and multiply by the
            # field amplitude
            if data.ntheta_plot != data.ntheta_grid:
                # Get eigenfunction data
                raw_eig_data = raw_data.get(f"eigenfunctions_{field_name}", None)
                if raw_eig_data is None:
                    logging.warning(
                        f"When setting fields, eigenfunction data for {field_name} not "
                        f"found, expected the file bin.cygro.{field_name}b to exist. "
                        f"Setting the field {field_name} to 0."
                    )
                    fields[ifield, :, :, :, :] = 0
                    continue
                eig_shape = [2, data.ntheta, data.ntime]
                eig_data = raw_eig_data[: np.prod(eig_shape)].reshape(
                    eig_shape, order="F"
                )
                eig_data = eig_data[0] + 1j * eig_data[1]
                # Get field amplitude
                middle_kx = (data.nradial // 2) + 1
                field_amplitude = np.real(field_data[middle_kx, 0, 0, :])
                # Multiply together
                # FIXME We only set kx=ky=0 here, any other values are left undefined
                #       as fields is created using np.empty. Should we instead set
                #       all kx and ky to these values? Should we expect that nx=ny=1?
                fields[ifield, :, 0, 0, :] = eig_data * field_amplitude
                continue

            # Poisson Sum (no negative in exponent to match frequency convention)
            q = gk_input.get_local_geometry_miller().q
            for i_radial in range(data.nradial):
                nx = -data.nradial // 2 + (i_radial - 1)
                field_data[i_radial, :, :, :] *= np.exp(2j * pi * nx * q)

            fields[ifield, :, :, :, :] = field_data.reshape(
                [data.ntheta, data.nkx, data.nky, data.ntime]
            )

        data["fields"] = (coords, fields)
        return data

    @staticmethod
    def _set_fluxes(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Set flux data over time.
        The flux coordinates should be (species, moment, field, ky, time)
        """
        if "flux" in raw_data:
            coords = ["species", "moment", "field", "ky", "time"]
            shape = [data.dims[coord] for coord in coords]
            fluxes = raw_data["flux"][: np.prod(shape)].reshape(shape, order="F")
            data["fluxes"] = (coords, fluxes)
        return data

    @staticmethod
    def _set_eigenvalues(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: Optional[Any] = None
    ) -> xr.Dataset:
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
            xr.Dataset: The modified dataset which was passed to 'data'.
        """
        # Use default method to calculate growth/freq if possible
        fields_contains_nan = np.any(np.isnan(data["fields"].data))
        fields_exist = [(f"field_{f}" in raw_data) for f in data["field"].data]
        if np.all(fields_exist) and not fields_contains_nan:
            data = GKOutputReader._set_eigenvalues(data, raw_data, gk_input)
            return data

        shape = (2, data.nky, data.ntime)

        if "eigenvalues_bin" in raw_data:
            eigenvalue_over_time = raw_data["eigenvalues_bin"][
                : np.prod(shape)
            ].reshape(shape, order="F")
        elif "eigenvalues_out" in raw_data:
            eigenvalue_over_time = (
                raw_data["eigenvalues_out"].transpose()[:, : data.ntime].reshape(shape)
            )
        else:
            raise RuntimeError(
                "Eigenvalues over time not found, expected the files bin.cgyro.freq or "
                "out.cgyro.freq to exist. Could not set data_vars 'growth_rate', "
                "'mode_frequency' and 'eigenvalue'."
            )

        mode_frequency = eigenvalue_over_time[0, :, :]
        growth_rate = eigenvalue_over_time[1, :, :]
        eigenvalue = mode_frequency + 1j * growth_rate
        # Add kx axis for compatibility with GS2 eigenvalues
        # FIXME Is this appropriate? Should we drop the kx coordinate?
        shape_with_kx = (data.nkx, data.nky, data.ntime)
        mode_frequency = np.ones(shape_with_kx) * mode_frequency
        growth_rate = np.ones(shape_with_kx) * growth_rate
        eigenvalue = np.ones(shape_with_kx) * eigenvalue

        data["growth_rate"] = (("kx", "ky", "time"), growth_rate)
        data["mode_frequency"] = (("kx", "ky", "time"), mode_frequency)
        data["eigenvalues"] = (("kx", "ky", "time"), eigenvalue)

        return data

    @staticmethod
    def _set_eigenfunctions(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Loads eigenfunctions into data with the following coordinates:

        data['eigenfunctions'] = eigenfunctions(kx, ky, field, theta, time)

        This should be called after _set_fields, and is only valid for linear runs.
        """

        # Use default method to calculate growth/freq if possible
        all_ballooning = data.ntheta_plot == data.ntheta_grid
        fields_contains_nan = np.any(np.isnan(data["fields"].data))
        fields_exist = [(f"field_{f}" in raw_data) for f in data["field"].data]
        if all_ballooning and np.all(fields_exist) and not fields_contains_nan:
            data = GKOutputReader._set_eigenfunctions(data, raw_data, gk_input)
            return data

        raw_eig_data = [
            raw_data.get(f"eigenfunctions_{f}", None) for f in data["field"].data
        ]
        raw_shape = [2, data.ntheta, data.ntime]

        # FIXME Currently using kx and ky for compatibility with GS2 results, but
        #       these coordinates are not used. Should we remove these coordinates?
        coords = ["kx", "ky", "field", "theta", "time"]
        eigenfunctions = np.empty([data.dims[coord] for coord in coords], dtype=complex)

        # Loop through all fields and add eigenfunction if it exists
        for ifield, raw_eigenfunction in enumerate(raw_eig_data):
            if raw_eigenfunction is not None:
                eigenfunction = raw_eigenfunction[: np.prod(raw_shape)].reshape(
                    raw_shape, order="F"
                )
                eigenfunctions[:, :, ifield, :, :] = (
                    eigenfunction[0] + 1j * eigenfunction[1]
                )

        data["eigenfunctions"] = (coords, eigenfunctions)
        return data
