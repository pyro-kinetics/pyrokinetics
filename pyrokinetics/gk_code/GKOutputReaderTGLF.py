import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .gk_output import (
    GKOutput,
    Coords,
    Fields,
    Fluxes,
    Eigenvalues,
    Eigenfunctions,
    Moments,
)
from .GKInputTGLF import GKInputTGLF
from ..typing import PathLike
from ..normalisation import SimulationNormalisation
from ..readers import Reader


class TGLFFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


@GKOutput.reader("TGLF")
class GKOutputReaderTGLF(Reader):
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
        coords = self._get_coords(raw_data, gk_input)
        fields = self._get_fields(raw_data, coords) if load_fields else None
        fluxes = self._get_fluxes(raw_data, coords) if load_fluxes else None
        moments = self._get_moments(raw_data, coords) if load_moments else None
        eigenvalues = self._get_eigenvalues(raw_data, coords, gk_input)
        eigenfunctions = (
            self._get_eigenfunctions(raw_data, coords) if coords["linear"] else None
        )

        # Assign units and return GKOutput
        convention = norm.cgyro

        field_dims = (("ky", "mode"),)
        flux_dims = (("field", "species", "ky"),)
        moment_dims = (("field", "species", "ky"),)
        eigenvalues_dims = (("ky", "mode"),)
        eigenfunctions_dims = (("theta", "mode", "field"),)
        return GKOutput(
            coords=Coords(
                time=coords["time"],
                kx=coords["kx"],
                ky=coords["ky"],
                theta=coords["theta"],
                mode=coords["mode"],
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
            eigenvalues=Eigenvalues(**eigenvalues, dims=eigenvalues_dims).with_units(
                convention
            )
            if eigenvalues
            else None,
            eigenfunctions=Eigenfunctions(eigenfunctions, dims=eigenfunctions_dims)
            if eigenfunctions
            else None,
            linear=coords["linear"],
            gk_code="TGLF",
            input_file=input_str,
        )

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
            "run": TGLFFile(dirname / "out.tglf.run", required=False),
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

    @staticmethod
    def _get_coords(raw_data: Dict[str, Any], gk_input: GKInputTGLF) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of TGLF
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing TGLF output.
            gk_input (GKInputTGLF): Processed TGLF input file.

        Returns:
            Dict: Dict with coords
        """

        bunit_over_b0 = gk_input.get_local_geometry().bunit_over_b0

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
            field = ["phi", "apar", "bpar"][:nfield]
            species = gk_input.get_local_species().names

            run = raw_data["run"].splitlines()
            ky = (
                float([line for line in run if "ky" in line][0].split(":")[-1].strip())
                / bunit_over_b0
            )

            # Store grid data as Dict
            return {
                "flux": None,
                "moment": None,
                "species": species,
                "field": field,
                "theta": theta,
                "mode": mode,
                "ky": [ky],
                "kx": [0.0],
                "time": [0.0],
                "linear": gk_input.is_linear(),
            }
        else:
            raw_grid = raw_data["ql_flux"].splitlines()[3].split(" ")
            grids = [int(g) for g in raw_grid if g]

            nflux = grids[0]
            nspecies = grids[1]
            nfield = grids[2]
            nmode = grids[4]

            flux = ["particle", "heat", "momentum", "par_momentum", "exchange"][:nflux]
            species = gk_input.get_local_species().names
            if nspecies != len(species):
                raise RuntimeError(
                    "GKOutputReaderTGLF: Different number of species in input and output."
                )
            field = ["phi", "apar", "bpar"][:nfield]
            ky = raw_data["ky"] / bunit_over_b0
            mode = list(range(1, 1 + nmode))

            # Store grid data as xarray DataSet
            return {
                "flux": flux,
                "moment": None,
                "species": species,
                "field": field,
                "theta": None,
                "ky": ky,
                "kx": [0.0],
                "mode": mode,
                "time": [0.0],
                "linear": gk_input.is_linear(),
            }

    @staticmethod
    def _get_fields(raw_data: Dict[str, Any], coords: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Sets fields over  for eac ky.
        The field coordinates should be (ky, mode, field)
        """

        # Check to see if there's anything to do
        if "field" not in raw_data.keys():
            return {}

        nky = len(coords["ky"])
        nmode = len(coords["mode"])
        nfield = len(coords["field"])

        f = raw_data["field"].splitlines()

        full_data = " ".join(f[6:]).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        fields = np.reshape(full_data, (nky, nmode, 4))
        fields = fields[:, :, 1 : nfield + 1]

        results = {}
        for ifield, field_name in enumerate(coords["field"]):
            results[field_name] = fields[:, :, ifield]

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputTGLF,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

    @staticmethod
    def _get_fluxes(raw_data: Dict[str, Any], coords: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Set flux data over time.
        The flux coordinates should be (species, field, ky, moment)
        """

        results = {}

        if "sum_flux" in raw_data:
            nky = len(coords["ky"])
            nfield = len(coords["field"])
            nflux = len(coords["flux"])
            nspecies = len(coords["species"])

            f = raw_data["sum_flux"].splitlines()
            full_data = [x for x in f if "species" not in x]
            full_data = " ".join(full_data).split(" ")

            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            fluxes = np.reshape(full_data, (nspecies, nfield, nky, nflux))
            # Order should be (flux, field, species, ky)
            fluxes = fluxes.transpose((3, 1, 0, 2))

            # Pyro doesn't handle parallel/exchange fluxs yet
            pyro_fluxes = ["particle", "heat", "momentum"]

            for iflux, flux in enumerate(coords["flux"]):
                if flux in pyro_fluxes:
                    results[flux] = fluxes[iflux, ...]

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: Dict[str, Any],
        coords: Dict[str, Any],
        gk_input: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
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

        results = {}
        # Use default method to calculate growth/freq if possible
        if "eigenvalues" in raw_data and not gk_input.is_linear():
            nky = len(coords["ky"])
            nmode = len(coords["mode"])

            f = raw_data["eigenvalues"].splitlines()

            full_data = " ".join(f).split(" ")
            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            eigenvalues = np.reshape(full_data, (nky, nmode, 2))
            eigenvalues = -eigenvalues[:, :, 1] + 1j * eigenvalues[:, :, 0]

            results["eigenvalues"] = eigenvalues
            results["growth_rate"] = np.imag(eigenvalues)
            results["mode_frequency"] = np.real(eigenvalues)

        elif gk_input.is_linear():
            nmode = len(coords["mode"])
            f = raw_data["run"].splitlines()
            lines = f[-nmode:]

            eigenvalues = np.array(
                [
                    list(filter(None, eig.strip().split(":")[-1].split("  ")))
                    for eig in lines
                ],
                dtype="float",
            )

            eigenvalues = eigenvalues.reshape((1, nmode, 2))
            mode_frequency = eigenvalues[:, :, 0]
            growth_rate = eigenvalues[:, :, 1]

            eigenvalues = mode_frequency + 1j * growth_rate
            results["growth_rate"] = growth_rate
            results["mode_frequency"] = mode_frequency

        return results

    @staticmethod
    def _get_eigenfunctions(
        raw_data: Dict[str, Any],
        coords: Dict[str, Any],
    ) -> np.ndarray:
        """
        Returns eigenfunctions with the coordinates ``(mode, field, theta)``

        Only possible with single ky runs (USE_TRANSPORT_MODEL=False)
        """

        # Load wavefunction if file exists
        if "wavefunction" not in raw_data:
            return None

        f = raw_data["wavefunction"].splitlines()
        grid = f[0].strip().split(" ")
        grid = [x for x in grid if x]

        # In case no unstable modes are found
        nmode_data = int(grid[0])
        nmode = len(coords["mode"])
        nfield = len(coords["field"])
        ntheta = len(coords["theta"])

        eigenfunctions = np.zeros((ntheta, nmode, nfield), dtype="complex")

        full_data = " ".join(f[1:]).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]
        full_data = np.reshape(full_data, (ntheta, (nmode_data * 2 * nfield) + 1))

        reshaped_data = np.reshape(full_data[:, 1:], (ntheta, nmode_data, nfield, 2))

        eigenfunctions[:, :nmode_data, :] = (
            reshaped_data[:, :, :, 1] + 1j * reshaped_data[:, :, :, 0]
        )
        return eigenfunctions


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
