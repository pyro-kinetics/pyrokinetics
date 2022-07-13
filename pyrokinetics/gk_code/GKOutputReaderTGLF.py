import numpy as np
import xarray as xr
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .GKOutputReader import GKOutputReader
from .GKInputTGLF import GKInputTGLF
from ..typing import PathLike


class TGLFFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


class GKOutputReaderTGLF(GKOutputReader):

    fields = ["phi", "apar", "bpar"]

    @staticmethod
    def _required_files(dirname: PathLike):
        dirname = Path(dirname)
        return {
            "input": TGLFFile(dirname / "input.tglf", required=True),
            "run": TGLFFile(dirname / "out.tglf.run", required=True),
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
        For TGLF, simply returns dir of the path.
        """
        return Path(filename).parent

    @classmethod
    def _get_raw_data(
        cls, dirname: PathLike
    ) -> Tuple[Dict[str, Any], GKInputTGLF, str]:
        dirname = Path(dirname)
        if not dirname.exists():
            raise RuntimeError(
                f"GKOutputReaderTGLF: Provided path {dirname} does not exist. "
                "Please supply the name of a directory containing TGLF output files."
            )
        if not dirname.is_dir():
            raise RuntimeError(
                f"GKOutputReaderTGLF: Provided path {dirname} is not a directory. "
                "Please supply the name of a directory containing TGLF output files."
            )

        # The following list of TGLF files may exist
        expected_files = {
            **cls._required_files(dirname),
            "wavefunction": TGLFFile(dirname / "out.tglf.wavefunction", required=False),
            "field": TGLFFile(dirname / "out.tglf.field_spectrum", required=False),
            "ky": TGLFFile(dirname / "out.tglf.ky_spectrum", required=False),
            "ql_flux": TGLFFile(dirname / "out.tglf.QL_flux_spectrum", required=False),
            "sum_flux": TGLFFile(
                dirname / "out.tglf.sum_flux_spectrum", required=False
            ),
            "eigenvalues": TGLFFile(
                dirname / "out.tglf.eigenvalue_spectrum", required=False
            ),
        }
        # Read in files
        raw_data = {}
        for key, tglf_file in expected_files.items():
            if not tglf_file.path.exists():
                if tglf_file.required:
                    raise RuntimeError(
                        f"GKOutputReaderTGLF: The file {tglf_file.path.name} is needed"
                    )
                continue
            # Read in file according to format
            if key == "ky":
                raw_data[key] = np.loadtxt(tglf_file.path, skiprows=2)

            else:
                with open(tglf_file.path, "r") as f:
                    raw_data[key] = f.read()

        input_str = raw_data["input"]
        gk_input = GKInputTGLF()
        gk_input.read_str(input_str)
        return raw_data, gk_input, input_str

    @classmethod
    def _init_dataset(
        cls, raw_data: Dict[str, Any], gk_input: GKInputTGLF
    ) -> xr.Dataset:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of TGLF
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing TGLF output.
            gk_input (GKInputTGLF): Processed TGLF input file.

        Returns:
            xr.Dataset: Dataset with coords and attrs set, but not data_vars
        """

        if gk_input.is_linear():
            f = raw_data["wavefunction"].splitlines()
            grid = f[0].strip().split(" ")
            grid = [x for x in grid if x]

            nmode_data = int(grid[0])
            nmode = gk_input.data.get("nmodes", 2)
            nfield = int(grid[1])
            ntheta = int(grid[2])

            full_data = " ".join(f[2:]).split(" ")
            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            full_data = np.reshape(full_data, (ntheta, (nmode_data * 2 * nfield) + 1))
            theta = full_data[:, 0]

            mode = list(range(1, 1 + nmode))
            field = cls.fields[:nfield]

            # Store grid data as xarray DataSet
            return xr.Dataset(
                coords={
                    "field": field,
                    "theta": theta,
                    "mode": mode,
                },
                attrs={
                    "nfield": nfield,
                    "ntheta": ntheta,
                    "nmode": nmode,
                },
            )

        else:
            raw_grid = raw_data["ql_flux"].splitlines()[3].split(" ")
            grids = [int(g) for g in raw_grid if g]

            nmoment = grids[0]
            nspecies = grids[1]
            nfield = grids[2]
            nky = grids[3]
            nmode = grids[4]

            moment = ["particle", "energy", "tor_momentum", "par_momentum", "exchange"][
                :nmoment
            ]
            species = gk_input.get_local_species().names
            if nspecies != len(species):
                raise RuntimeError(
                    "GKOutputReaderTGLF: Different number of species in input and output."
                )
            field = cls.fields[:nfield]
            ky = raw_data["ky"]
            mode = list(range(1, 1 + nmode))

            # Store grid data as xarray DataSet
            return xr.Dataset(
                coords={
                    "moment": moment,
                    "species": species,
                    "field": field,
                    "ky": ky,
                    "mode": mode,
                },
                attrs={
                    "nmoment": nmoment,
                    "nspecies": nspecies,
                    "nfield": nfield,
                    "nky": nky,
                    "nmode": nmode,
                },
            )

    @staticmethod
    def _set_fields(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: GKInputTGLF
    ) -> xr.Dataset:
        """
        Sets fields over  for eac ky.
        The field coordinates should be (ky, mode, field)
        """

        coords = ["ky", "mode", "field"]
        # Check to see if there's anything to do
        if "field" not in raw_data.keys():
            return data

        nky = data.nky
        nmode = data.nmode
        nfield = data.nfield

        f = raw_data["field"].splitlines()

        full_data = " ".join(f[6:]).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        fields = np.reshape(full_data, (nky, nmode, 4))
        fields = fields[:, :, 1 : nfield + 1]

        data["fields"] = (coords, fields)

        # FIXME Currently using "nonlinear" TGLF also generates the linear growth rates
        #      but it is not possible to call _set_eigenvalues from here and we
        #      shouldn't add this to the default reading from GKOutputReader so I
        #      manually load it in with the fields for now
        coords = ["ky", "mode"]

        nky = data.nky
        nmode = data.nmode

        f = raw_data["eigenvalues"].splitlines()

        full_data = " ".join(f).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        eigenvalues = np.reshape(full_data, (nky, nmode, 2))
        eigenvalues = -eigenvalues[:, :, 1] + 1j * eigenvalues[:, :, 0]

        data["eigenvalues"] = (coords, eigenvalues)
        data["growth_rate"] = (coords, np.imag(eigenvalues))
        data["mode_frequency"] = (coords, np.real(eigenvalues))

        return data

    @staticmethod
    def _set_fluxes(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Set flux data over time.
        The flux coordinates should be (species, field, ky, moment)
        """
        if "sum_flux" in raw_data:
            coords = ["species", "field", "ky", "moment"]
            nky = data.nky
            nspecies = data.nspecies
            nfield = data.nfield
            nmoment = data.nmoment

            f = raw_data["sum_flux"].splitlines()
            full_data = [x for x in f if "species" not in x]
            full_data = " ".join(full_data).split(" ")

            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            fluxes = np.reshape(full_data, (nspecies, nfield, nky, nmoment))

            data["fluxes"] = (coords, fluxes)

        return data

    @staticmethod
    def _set_eigenvalues(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:

        data['eigenvalues'] = eigenvalues(ky, mode)
        data['mode_frequency'] = mode_frequency(ky, mode)
        data['growth_rate'] = growth_rate(ky, mode)

        This is only valid for transport runs.
        Unlike the version in the super() class, TGLF needs to get extra info from
        an eigenvalue file.

        Args:
            data (xr.Dataset): The dataset to be modified.
            dirname (PathLike): Directory containing TGLF output files.
        Returns:
            xr.Dataset: The modified dataset which was passed to 'data'.
        """

        # Use default method to calculate growth/freq if possible
        if "eigenvalues" in raw_data and not gk_input.is_linear():

            coords = ["ky", "mode"]

            nky = data.nky
            nmode = data.nmode

            f = raw_data["eigenvalues"].splitlines()

            full_data = " ".join(f).split(" ")
            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            eigenvalues = np.reshape(full_data, (nky, nmode, 2))
            eigenvalues = -eigenvalues[:, :, 1] + 1j * eigenvalues[:, :, 0]

            data["eigenvalues"] = (coords, eigenvalues)
            data["growth_rate"] = (coords, np.imag(eigenvalues))
            data["mode_frequency"] = (coords, np.real(eigenvalues))

        elif gk_input.is_linear():
            coords = ["mode"]
            nmode = data.nmode

            f = raw_data["run"].splitlines()

            lines = f[-nmode:]

            eigenvalues = np.array(
                [
                    list(filter(None, eig.strip().split(":")[-1].split("  ")))
                    for eig in lines
                ],
                dtype="float",
            )

            mode_frequency = -eigenvalues[:, 0]
            growth_rate = eigenvalues[:, 1]
            eigenvalues = -eigenvalues[:, 0] + 1j * eigenvalues[:, 1]

            data["eigenvalues"] = (coords, eigenvalues)
            data["growth_rate"] = (coords, growth_rate)
            data["mode_frequency"] = (coords, mode_frequency)

        return data

    @staticmethod
    def _set_eigenfunctions(
        data: xr.Dataset, raw_data: Dict[str, Any], gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Loads eigenfunctions into data with the following coordinates:

        data['eigenfunctions'] = eigenfunctions(mode, field, theta)

        Only possible with single ky runs (USE_TRANSPORT_MODEL=False)
        """

        # Load wavefunction if file exists

        if "wavefunction" in raw_data:
            coords = ["theta", "mode", "field"]

            f = raw_data["wavefunction"].splitlines()
            grid = f[0].strip().split(" ")
            grid = [x for x in grid if x]

            # In case no unstable modes are found
            nmode_data = int(grid[0])
            nmode = data.nmode
            nfield = data.nfield
            ntheta = data.ntheta

            eigenfunctions = np.zeros((ntheta, nmode, nfield), dtype="complex")

            full_data = " ".join(f[1:]).split(" ")
            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            full_data = np.reshape(full_data, (ntheta, (nmode_data * 2 * nfield) + 1))

            reshaped_data = np.reshape(
                full_data[:, 1:], (ntheta, nmode_data, nfield, 2)
            )

            eigenfunctions[:, :nmode_data, :] = (
                reshaped_data[:, :, :, 1] + 1j * reshaped_data[:, :, :, 0]
            )
            data["eigenfunctions"] = (coords, eigenfunctions)

        return data


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
