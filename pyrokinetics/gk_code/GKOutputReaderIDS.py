import numpy as np
import pint  # noqa
import pint_xarray  # noqa
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import xarray as xr
from ast import literal_eval
from h5py import is_hdf5
from idspy_dictionaries import ids_gyrokinetics
from xmltodict import parse as xmltodict

from .gk_output import (
    GKOutput,
    get_flux_units,
    get_field_units,
    get_moment_units,
    get_coord_units,
    FieldDict,
    FluxDict,
    MomentDict,
)
from . import gk_inputs
from . import GKInput
from ..typing import PathLike

from ..readers import Reader
from ..normalisation import SimulationNormalisation, ureg


class IDSFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


@GKOutput.reader("IDS")
class GKOutputReaderIDS(Reader):
    fields = ["phi", "apar", "bpar"]

    def read(
        self,
        filename: PathLike,
        norm: SimulationNormalisation,
        ids: ids_gyrokinetics,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
    ) -> GKOutput:
        gk_input = self._get_gk_input(ids)
        coords = self._get_coords(ids, gk_input)

        if load_fields:
            fields = self._get_fields(ids, coords)
        else:
            fields = {}

        if load_fluxes:
            fluxes = self._get_fluxes(ids, coords)
        else:
            fluxes = {}

        if load_moments:
            moments = self.__get_moments(ids, coords)
        else:
            moments = {}

        # Check dimensions of outputs
        if fluxes["particle"].ndim == 4:
            flux_shape = ("field", "species", "ky", "time")
        else:
            flux_shape = ("field", "species", "time")

        # Assign units and return GKOutput
        convention = norm.imas
        coord_units = get_coord_units(convention)
        field_units = get_field_units(convention)
        flux_units = get_flux_units(convention)
        moment_units = get_moment_units(convention)

        for field_name, field in fields.items():
            fields[field_name] = field * field_units[field_name]

        for flux_type, flux in fluxes.items():
            fluxes[flux_type] = flux * flux_units[flux_type]

        for moment_type, moment in moments.items():
            moments[moment_type] = moment * moment_units[moment_type]

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
            moment_dim=coords["moment"],
            flux_dim=coords["flux"],
            field_var=("theta", "kx", "ky", "time"),
            flux_var=flux_shape,
            moment_var=("theta", "kx", "species", "ky", "time"),
            species=coords["species"],
            gk_code=coords["gk_code"],
            fields=fields,
            moments=moments,
            fluxes=fluxes,
            norm=norm,
            linear=coords["linear"],
            growth_rate=growth_rate,
            mode_frequency=mode_frequency,
            eigenfunctions=eigenfunctions,
        )

    def verify(self, dirname: PathLike):
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

        gk_input = gk_inputs[ids.code.name]
        gk_input.read_dict(gk_input_dict)

        return gk_input

    @staticmethod
    def _get_coords(ids: ids_gyrokinetics, gk_input: GKInput) -> Dict[str, Any]:
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
        theta = ids.wavevector[0].eigenmode[0].poloidal_angle

        nfield = (
            1
            + int(ids.model.include_a_field_parallel)
            + 1
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
    ) -> FieldDict:
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
    ) -> FluxDict:
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
    ) -> MomentDict:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

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
