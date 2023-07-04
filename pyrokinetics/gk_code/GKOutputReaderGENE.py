import numpy as np
import xarray as xr
import f90nml
import logging
import struct
import csv
import re
import h5py
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from ast import literal_eval

from .gk_output import (
    GKOutput,
    get_flux_units,
    get_field_units,
    get_coord_units,
    get_moment_units,
    get_eigenvalues_units,
    FieldDict,
    FluxDict,
    MomentDict,
)
from .GKInputGENE import GKInputGENE
from ..constants import pi
from ..typing import PathLike
from ..readers import Reader
from ..normalisation import SimulationNormalisation
from ..units import ureg


@GKOutput.reader("GENE")
class GKOutputReaderGENE(Reader):
    fields = ["phi", "apar", "bpar"]

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

        # Determine normalisation used
        nml = gk_input.data
        if nml["geometry"].get("minor_r", 0.0) == 1.0:
            convention = norm.pyrokinetics
        elif gk_input.data["geometry"].get("major_R", 1.0) == 1.0:
            convention = norm.gene
        else:
            raise NotImplementedError(
                "Pyro does not handle GENE cases where neither major_R and minor_r are 1.0"
            )

        # Assign units and return GKOutput
        coord_units = get_coord_units(convention)
        field_units = get_field_units(convention)
        moments_units = get_moment_units(convention)
        flux_units = get_flux_units(convention)
        eig_units = get_eigenvalues_units(convention)

        for field_name, field in fields.items():
            fields[field_name] = field * field_units[field_name]

        for moment_name, moment in moments.items():
            moments[moment_name] = moment * moments_units[moment_name]

        for flux_type, flux in fluxes.items():
            fluxes[flux_type] = flux * flux_units[flux_type]

        if coords["linear"] and not fields:
            eigenvalues = self._get_eigenvalues(raw_data, coords)
            growth_rate = eigenvalues["growth_rate"] * eig_units["growth_rate"]
            mode_frequency = eigenvalues["mode_frequency"] * eig_units["mode_frequency"]
        else:
            # Rely on gk_output to generate eigenvalues
            growth_rate = None
            mode_frequency = None

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
            field_var=("theta", "kx", "ky", "time"),
            flux_var=("field", "species", "time"),
            moment_var=("field", "species", "time"),
            species=coords["species"],
            fields=fields,
            fluxes=fluxes,
            moments=moments,
            norm=norm,
            linear=coords["linear"],
            gk_code="GENE",
            input_file=input_str,
            growth_rate=growth_rate,
            mode_frequency=mode_frequency,
            normalise_flux_moment=True,
        )

    @staticmethod
    def _get_gene_files(filename: PathLike) -> Dict[str, Path]:
        """
        Given a directory name, looks for the files filename/parameters_0000,
        filename/field_0000 and filename/nrg_0000.
        If instead given any of the files parameters_####, field_#### or nrg_####,
        looks up the rest of the files in the same directory.
        """
        filename = Path(filename)
        prefixes = ["parameters", "field", "nrg", "omega"]
        if filename.is_dir():
            # If given a dir name, looks for dir/parameters_0000
            dirname = filename
            dat_matches = np.any(
                [Path(filename / f"{p}.dat").is_file() for p in prefixes]
            )
            if dat_matches:
                suffix = "dat"
                delimiter = "."
            else:
                suffix = "0000"
                delimiter = "_"
        else:
            # If given a file, searches for all similar GENE files in that file's dir
            dirname = filename.parent
            # Ensure provided file is a GENE file (fr"..." means raw format str)
            matches = [re.search(rf"^{p}_\d{{4}}$", filename.name) for p in prefixes]
            if not np.any(matches):
                raise RuntimeError(
                    f"GKOutputReaderGENE: The provided file {filename} is not a GENE "
                    "output file."
                )
            suffix = filename.name.split("_")[1]
            delimiter = "_"

        # Get all files in the same dir
        files = {
            prefix: dirname / f"{prefix}{delimiter}{suffix}"
            for prefix in prefixes
            if (dirname / f"{prefix}{delimiter}{suffix}").exists()
        }

        if not files:
            raise RuntimeError(
                "GKOutputReaderGENE: Could not find GENE output files in the "
                f"directory '{dirname}'."
            )
        if "parameters" not in files:
            raise RuntimeError(
                "GKOutputReaderGENE: Could not find GENE output file 'parameters_"
                f"{suffix}' when provided with the file/directory '{filename}'."
            )
        # If binary field file absent, adds .h5 field file,
        # if present, to 'files'
        if "field" not in files:
            if (dirname / f"field{delimiter}{suffix}.h5").exists():
                files.update({"field": dirname / f"field{delimiter}{suffix}.h5"})
        return files

    def verify(self, filename: PathLike):
        self._get_gene_files(filename)

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        """
        # If the input file is of the form name_####, get the numbered part and
        # search for 'parameters_####' in the run directory. If not, simply return
        # the directory.
        filename = Path(filename)
        num_part_regex = re.compile(r"(\d{4})")
        num_part_match = num_part_regex.search(filename.name)

        if num_part_match is None:
            return Path(filename).parent
        else:
            return Path(filename).parent / f"parameters_{num_part_match[0]}"
        pass

    @classmethod
    def _get_raw_data(
        cls, filename: PathLike
    ) -> Tuple[Dict[str, Any], GKInputGENE, str]:
        files = cls._get_gene_files(filename)
        # Read parameters_#### as GKInputGENE and into plain string
        with open(files["parameters"], "r") as f:
            input_str = f.read()
        gk_input = GKInputGENE()
        gk_input.read_str(input_str)
        # Defer processing field and flux data until their respective functions
        # Simply return files in place of raw data
        return files, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: Dict[str, Any], gk_input: GKInputGENE, downsize: int
    ) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a GENE parameters file.

        Args:
            raw_data (Dict[str,Any]): Dict containing GENE output. Ignored.
            gk_input (GKInputGENE): Processed GENE input file.

        Returns:
            xr.Dataset: Dataset with coords and attrs set, but not data_vars
        """
        nml = gk_input.data

        # The last time step is not always written, but depends on
        # whatever condition is met first between simtimelim and timelim
        species = gk_input.get_local_species().names
        with open(raw_data["nrg"], "r") as f:
            full_data = f.readlines()
            ntime = len(full_data) // (len(species) + 1)
            lasttime = float(full_data[-(len(species) + 1)])

        ntime = (
            int(ntime * nml["in_out"]["istep_nrg"] / nml["in_out"]["istep_field"]) + 1
        )

        if lasttime == nml["general"]["simtimelim"]:
            ntime = ntime + 1

        # Set time to index for now, gets overwritten by field data
        time = np.linspace(0, ntime - 1, ntime)

        nfield = nml["info"]["n_fields"]
        field = ["phi", "apar", "bpar"][:nfield]

        nky = nml["box"]["nky0"]
        nkx = nml["box"]["nx0"]
        ntheta = nml["box"]["nz0"]
        theta = np.linspace(-pi, pi, ntheta, endpoint=False)

        nenergy = nml["box"]["nv0"]
        energy = np.linspace(-1, 1, nenergy)

        npitch = nml["box"]["nw0"]
        pitch = np.linspace(-1, 1, npitch)

        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]

        if gk_input.is_linear():
            # Set up ballooning angle
            single_theta_loop = theta
            single_ntheta_loop = ntheta

            ntheta = ntheta * (nkx - 1)
            theta = np.empty(ntheta)
            start = 0
            for i in range(nkx - 1):
                pi_segment = i - nkx // 2 + 1
                theta[start : start + single_ntheta_loop] = (
                    single_theta_loop + pi_segment * 2 * pi
                )
                start += single_ntheta_loop

            ky = [nml["box"]["kymin"]]
            kx = [0.0]
            nkx = 1
            # TODO should we not also set nky=1?

        else:
            kymin = nml["box"]["kymin"]
            ky = np.linspace(0, kymin * (nky - 1), nky)
            lx = nml["box"]["lx"]
            dkx = 2 * np.pi / lx
            kx = np.empty(nkx)
            for i in range(nkx):
                if i < (nkx / 2 + 1):
                    kx[i] = i * dkx
                else:
                    kx[i] = (i - nkx) * dkx

            kx = np.roll(np.fft.fftshift(kx), -1)

        # Convert to Pyro coordinate (need magnitude to set up Dataset)

        # Store grid data as xarray DataSet
        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "moment": moments,
            "flux": fluxes,
            "field": field,
            "species": species,
            "downsize": downsize,
            "linear": gk_input.is_linear(),
            "lasttime": lasttime,
        }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, Any],
        gk_input: GKInputGENE,
        coords: Dict[str, Any],
    ) -> FieldDict:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """

        if "field" not in raw_data:
            return {}

        # Time data stored as binary (int, double, int)
        time = []
        time_data_fmt = "=idi"
        time_data_size = struct.calcsize(time_data_fmt)

        int_size = 4
        complex_size = 16

        downsize = coords["downsize"]

        nx = gk_input.data["box"]["nx0"]
        nz = gk_input.data["box"]["nz0"]

        nkx = len(coords["kx"])
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntime = len(coords["time"])
        nfield = len(coords["field"])

        field_size = nx * nz * nky * complex_size

        sliced_field = np.empty((nfield, nx, nky, nz, ntime), dtype=complex)
        fields = np.empty((nfield, nkx, nky, ntheta, ntime), dtype=complex)
        # Read binary file if present
        if ".h5" not in str(raw_data["field"]):
            with open(raw_data["field"], "rb") as file:
                for i_time in range(ntime):
                    # Read in time data (stored as int, double int)
                    time_value = float(
                        struct.unpack(time_data_fmt, file.read(time_data_size))[1]
                    )
                    time.append(time_value)
                    for i_field in range(nfield):
                        file.seek(int_size, 1)
                        binary_field = file.read(field_size)
                        raw_field = np.frombuffer(binary_field, dtype=np.complex128)
                        sliced_field[i_field, :, :, :, i_time] = raw_field.reshape(
                            (nx, nky, nz),
                            order="F",
                        )
                        file.seek(int_size, 1)
                    if i_time < ntime - 1:
                        file.seek(
                            (downsize - 1)
                            * (time_data_size + nfield * (2 * int_size + field_size)),
                            1,
                        )

        # Read .h5 file if binary file absent
        else:
            h5_field_subgroup_names = ["phi", "A_par", "B_par"]
            fields = np.empty(
                (nfield, nkx, nky, ntheta, ntime),
                dtype=complex,
            )
            with h5py.File(raw_data["field"], "r") as file:
                # Read in time data
                time.extend(list(file.get("field/time")))
                for i_field in range(nfield):
                    h5_subgroup = "field/" + h5_field_subgroup_names[i_field] + "/"
                    h5_dataset_names = list(file[h5_subgroup].keys())
                    for i_time in range(ntime):
                        h5_dataset = h5_subgroup + h5_dataset_names[i_time]
                        raw_field = np.array(file.get(h5_dataset))
                        raw_field = np.array(
                            raw_field["real"] + raw_field["imaginary"] * 1j,
                            dtype="complex128",
                        )
                        sliced_field[i_field, :, :, :, i_time] = np.swapaxes(
                            raw_field, 0, 2
                        )

        # Match pyro convention for ion/electron direction
        sliced_field = np.conjugate(sliced_field)

        if not gk_input.is_linear():
            nl_shape = (nfield, nkx, nky, ntheta, ntime)
            fields = sliced_field.reshape(nl_shape, order="F")

        # Convert from kx to ballooning space
        else:
            try:
                n0_global = gk_input.data["box"]["n0_global"]
                q0 = gk_input.data["geometry"]["q0"]
                phase_fac = -np.exp(-2 * np.pi * 1j * n0_global * q0)
            except KeyError:
                phase_fac = -1
            i_ball = 0

            for i_conn in range(-int(nx / 2) + 1, int((nx - 1) / 2) + 1):
                fields[:, 0, :, i_ball : i_ball + nz, :] = (
                    sliced_field[:, i_conn, :, :, :] * (phase_fac) ** i_conn
                )
                i_ball += nz

        # =================================================

        # Overwrite 'time' coordinate as determined in _init_dataset
        coords["time"] = time

        # Original method coords: (field, kx, ky, theta, time)
        # New coords: (field, theta, kx, ky, time)
        fields = fields.transpose(0, 3, 1, 2, 4)

        # Shift kx component to middle of array
        fields = np.roll(np.fft.fftshift(fields, axes=2), -1, axis=-2)

        result = {}

        for ifield, field_name in enumerate(coords["field"]):
            result[field_name] = fields[ifield, ...]

        return result

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputGENE,
        coords: Dict[str, Any],
    ) -> MomentDict:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

    @staticmethod
    def _get_fluxes(raw_data: Dict[str, Any], coords: Dict[str, Any]) -> FluxDict:
        """
        Set flux data over time.
        The flux coordinates should  be (species, flux, field, ky, time)
        """

        # ky data not available in the nrg file so no ky coords here
        coord_names = ["species", "flux", "field", "time"]
        shape = [len(coords[coord_name]) for coord_name in coord_names]
        fluxes = np.empty(shape)

        nfield = len(coords["field"])
        nspecies = len(coords["species"])
        ntime = len(coords["time"])

        if "nrg" not in raw_data:
            logging.warning("Flux data not found, setting all fluxes to zero")
            fluxes[...] = 0
            result = {"fluxes": fluxes}
            return result

        nml = f90nml.read(raw_data["parameters"])
        flux_istep = nml["in_out"]["istep_nrg"]
        field_istep = nml["in_out"]["istep_field"]

        ntime_flux = nml["info"]["steps"][0] // flux_istep + 1
        if nml["info"]["steps"][0] % flux_istep > 0:
            ntime_flux += 1

        downsize = coords["downsize"]

        if nml["general"]["simtimelim"] == coords["lasttime"]:
            final_time = True
        else:
            final_time = False

        if flux_istep < field_istep:
            time_skip = int(field_istep * downsize / flux_istep) - 1
        else:
            time_skip = downsize - 1

        with open(raw_data["nrg"], "r") as csv_file:
            nrg_data = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)

            if nfield == 3:
                logging.warning(
                    "GENE combines Apar and Bpar fluxes, setting Bpar fluxes to zero"
                )
                fluxes[:, :, 2, :] = 0.0
                field_size = 2
            else:
                field_size = nfield

            for i_time in range(ntime):
                time = next(nrg_data)  # noqa

                for i_species in range(nspecies):
                    nrg_line = np.array(next(nrg_data), dtype=float)

                    # Particle
                    fluxes[i_species, 0, :field_size, i_time] = nrg_line[
                        4 : 4 + field_size,
                    ]

                    # Heat
                    fluxes[i_species, 1, :field_size, i_time] = nrg_line[
                        6 : 6 + field_size,
                    ]

                    # Momentum
                    fluxes[i_species, 2, :field_size, i_time] = nrg_line[
                        8 : 8 + field_size,
                    ]

                # Skip time/data values in field print out is less
                if i_time < ntime - 2:
                    for skip_t in range(time_skip):
                        for skip_s in range(nspecies + 1):
                            next(nrg_data)
                elif i_time == ntime - 2:
                    if not final_time:
                        final_skip = time_skip
                    else:
                        final_skip = ntime_flux - (i_time * (time_skip + 1)) - 2
                    for skip_t in range(final_skip):
                        for skip_s in range(nspecies + 1):
                            next(nrg_data)

        results = {}

        fluxes = fluxes.transpose(1, 2, 0, 3)

        for iflux, flux in enumerate(coords["flux"]):
            results[flux] = fluxes[iflux, ...]

        return results

    @staticmethod
    def _get_eigenvalues(raw_data: Dict[str, Any], coords: Dict) -> Dict[str, Any]:
        """

        Parameters
        ----------
        raw_data
        coords

        Returns
        -------
        Dict of eigenvalues with coords (kx, ky, time)
            Only final time is output so we set that to all the times
        """

        nky = len(coords["ky"])
        nkx = len(coords["kx"])
        ntime = len(coords["time"])
        mode_frequency = np.empty((nkx, nky, ntime))
        growth_rate = np.empty((nkx, nky, ntime))

        with open(raw_data["omega"], "r") as csv_file:
            omega_data = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)
            for iky, line in enumerate(omega_data):
                ky, growth, frequency = line

                mode_frequency[:, iky, :] = float(frequency)
                growth_rate[:, iky, :] = float(growth)

        results = {"growth_rate": growth_rate, "mode_frequency": mode_frequency}

        return results
