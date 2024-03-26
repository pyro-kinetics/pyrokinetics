from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from h5py import is_hdf5
from idspy_dictionaries import ids_gyrokinetics
from xmltodict import parse as xmltodict

from ..file_utils import FileReader
from ..normalisation import SimulationNormalisation, ureg
from ..typing import PathLike
from . import GKInput
from .gk_output import Coords, Eigenvalues, Fields, Fluxes, GKOutput, Moments


class IDSFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


class GKOutputReaderIDS(FileReader, file_type="IDS", reads=GKOutput):
    fields = ["phi", "apar", "bpar"]

    def read_from_file(
        self,
        filename: PathLike,
        norm: SimulationNormalisation,
        ids: ids_gyrokinetics,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
        original_theta_geo=None,
        mxh_theta_geo=None,
    ) -> GKOutput:
        gk_input = self._get_gk_input(ids)
        coords = self._get_coords(ids, gk_input, original_theta_geo, mxh_theta_geo)

        fields = self._get_fields(ids, coords) if load_fields else None
        fluxes = self._get_fluxes(ids, coords) if load_fluxes else None
        moments = self.__get_moments(ids, coords) if load_moments else None

        # Assign units and return GKOutput
        convention = norm.imas
        # Check dimensions of outputs
        if fluxes["particle"].ndim == 4:
            flux_dims = ("field", "species", "ky", "time")
        else:
            flux_dims = ("field", "species", "time")
        moment_dims = ("theta", "kx", "species", "ky", "time")
        field_dims = ("theta", "kx", "ky", "time")

        eigenvalues = {}

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
            fields=(
                Fields(**fields, dims=field_dims).with_units(convention)
                if fields
                else None
            ),
            fluxes=(
                Fluxes(**fluxes, dims=flux_dims).with_units(convention)
                if fluxes
                else None
            ),
            moments=(
                Moments(**moments, dims=moment_dims).with_units(convention)
                if moments
                else None
            ),
            eigenvalues=(
                Eigenvalues(**eigenvalues).with_units(convention)
                if eigenvalues
                else None
            ),
            linear=coords["linear"],
            gk_code=coords["gk_code"],
            normalise_flux_moment=False,
        )

    def verify_file_type(self, dirname: PathLike):
        dirname = Path(dirname)
        if not is_hdf5(dirname):
            raise RuntimeError

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        For CGYRO, simply returns dir of the path.
        """
        return Path(filename).parent

    @classmethod
    def _get_gk_input(cls, ids: ids_gyrokinetics) -> Tuple[GKInput, str]:
        gk_input_dict = xmltodict(ids.code.parameters)["root"]
        dict_to_numeric(gk_input_dict)

        gk_input = GKInput._factory(ids.code.name)
        gk_input.read_dict(gk_input_dict)

        return gk_input

    @staticmethod
    def _get_coords(
        ids: ids_gyrokinetics, gk_input: GKInput, original_theta_geo, mxh_theta_geo
    ) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of CGYRO
        files.

        Args:
            ids (Dict[str,Any]): Dict containing CGYRO output.
            gk_input (GKInput): Processed CGYRO input file.

        Returns:
            Dict:  Dictionary with coords
        """

        # Process time data
        time = ids.time

        kx = []
        ky = []
        for wv in ids.wavevector:
            kx.append(wv.radial_component_norm)
            ky.append(wv.binormal_component_norm)

        kx = np.sort(np.unique(kx))
        ky = np.sort(np.unique(ky))
        mxh_theta_output = ids.wavevector[0].eigenmode[0].poloidal_angle

        theta_interval = mxh_theta_output // (2 * np.pi)
        theta_norm = mxh_theta_output % (2 * np.pi)
        original_theta_output = np.interp(theta_norm, mxh_theta_geo, original_theta_geo)
        original_theta_output += theta_interval * 2 * np.pi

        theta = original_theta_output

        nfield = (
            1
            + int(ids.model.include_a_field_parallel)
            + int(ids.model.include_b_field_parallel)
        )
        nspecies = len(ids.species)

        fields = ["phi", "apar", "bpar"][:nfield]
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]

        species = gk_input.get_local_species().names
        if nspecies != len(species):
            raise RuntimeError(
                "GKOutputReaderCGYRO: Different number of species in input and output."
            )

        # Not currently stored
        pitch = [0.0]
        energy = [0.0]

        # TODO code dependant flux_loc
        flux_loc = "particle"
        gk_code = ids.code.name

        # Store grid data as xarray DataSet
        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "pitch": pitch,
            "energy": energy,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "linear": gk_input.is_linear(),
            "flux_loc": flux_loc,
            "gk_code": gk_code,
        }

    @staticmethod
    def _get_fields(
        ids: Dict[str, Any],
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """
        nkx = len(coords["kx"])
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntime = len(coords["time"])

        results = {
            field: np.empty((ntheta, nkx, nky, ntime), dtype=complex)
            for field in coords["field"]
        }

        # Loop through all wavevectors
        for wv in ids.wavevector:
            # TODO only handles one eigenmode at the minute (should I sum over eigemodes)
            eigenmode = wv.eigenmode[0]
            ikx = np.argwhere(coords["kx"] == wv.radial_component_norm).flatten()[0]
            iky = np.argwhere(coords["ky"] == wv.binormal_component_norm).flatten()[0]
            for field, imas_field in zip(
                coords["field"], imas_pyro_field_names.values()
            ):
                results[field][:, ikx, iky, :] = getattr(
                    eigenmode, f"{imas_field}_perturbed_norm"
                )

        return results

    @staticmethod
    def _get_fluxes(
        ids: Dict[str, Any],
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Set flux data over time.
        The flux coordinates should be (species, flux, field, ky, time)
        """

        flux_loc = coords["flux_loc"]
        nky = len(coords["ky"])
        ntime = len(coords["time"])
        nspecies = len(coords["species"])
        nfield = len(coords["field"])

        # TODO Does imas store ky flux spectrum
        results = {
            flux: np.empty((nfield, nspecies, nky, ntime), dtype=complex)
            for flux in coords["flux"]
        }

        for wv in ids.wavevector:
            eigenmode = wv.eigenmode[0]
            iky = np.argwhere(coords["ky"] == wv.binormal_component_norm).flatten()[0]
            for isp, fm in enumerate(eigenmode.fluxes_moments):
                flux_data = getattr(fm, f"fluxes_norm_{flux_loc}")
                for imom, (flux, imas_flux) in enumerate(
                    zip(coords["flux"], imas_pyro_flux_names.values())
                ):
                    for ifield, (pyro_field, imas_field) in enumerate(
                        zip(coords["field"], imas_pyro_field_names.values())
                    ):
                        results[flux][ifield, isp, iky, :] = getattr(
                            flux_data, f"{imas_flux}_{imas_field}"
                        )

        # GENE does have flux as a function of ky
        if coords["gk_code"] == "GENE":
            for key, flux in results.items():
                results[key] = flux[:, :, 0, :]

        return results

    @staticmethod
    def _get_moments(
        ids: Dict[str, Any],
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

    @staticmethod
    def to_netcdf(self, *args, **kwargs) -> None:
        """Writes self.data to disk. Forwards all args to xarray.Dataset.to_netcdf."""
        import pint_xarray  # noqa
        import xarray as xr

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
        import pint_xarray  # noqa
        import xarray as xr

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


def dict_to_numeric(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str):
                try:
                    o[k] = literal_eval(v)
                except (SyntaxError, ValueError):
                    pass
                if v == "true":
                    o[k] = True
                if v == "false":
                    o[k] = False
            else:
                dict_to_numeric(v)
    elif isinstance(o, list):
        for v in o:
            dict_to_numeric(v)


imas_pyro_field_names = {
    "phi": "phi_potential",
    "apar": "a_field_parallel",
    "bpar": "b_field_parallel",
}

imas_pyro_flux_names = {
    "particle": "particles",
    "heat": "energy",
    "momentum": "momentum_tor_perpendicular",
}

imas_pyro_moment_names = {
    "density": "density",
    "temperature": "heat_flux",
    "velocity": "velocity_parallel",
}
