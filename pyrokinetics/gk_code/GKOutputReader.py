import numpy as np
import xarray as xr
from abc import abstractmethod
from typing import Optional, Tuple, Any
from pathlib import Path
import warnings

from .GKInput import GKInput
from ..typing import PathLike
from ..constants import pi
from ..readers import Reader, create_reader_factory


def get_growth_rate_tolerance(data: xr.Dataset, time_range: float = 0.8):
    """
    Given a pyrokinetics output dataset with eigenvalues determined, calculate the
    growth rate tolerance. This is calculated starting at the time given by
    time_range * max_time.
    """
    if "growth_rate" not in data:
        raise ValueError(
            "Provided Dataset does not have growth rate. The dataset should be "
            "associated with a linear gyrokinetics runs"
        )
    growth_rate = data["growth_rate"]
    final_growth_rate = growth_rate.isel(time=-1)
    difference = np.abs((growth_rate - final_growth_rate) / final_growth_rate)
    final_time = difference["time"].isel(time=-1).data
    # Average over the end of the simulation, starting at time_range*final_time
    within_time_range = difference["time"].data > time_range * final_time
    tolerance = np.sum(np.where(within_time_range, difference, 0), axis=-1) / np.sum(
        within_time_range, axis=-1
    )
    return tolerance


class GKOutputReader(Reader):
    """
    A GKOutputReader reads in output data from gyrokinetics codes, and converts it to
    a standardised schema to allow for easier cross-code comparisons. Using the read
    method, it takes in ouput data typically expressed as a .cdf file, and converts it
    to an xarray Dataset. The functions _set_grids, _set_fields, _set_fluxes,
    _set_eigenvalues, _set_eigenfunctions, and _set_growth_rate_tolerance are used to
    build up the Dataset, and need not be called by the user.

    The produced xarray Dataset should have the following:

    coords
        time        1D array of floats
        kx          1D array of floats
        ky          1D array of floats
        theta       1D array of floats
        energy      1D array of floats
        pitch       1D array of floats
        moment      ["particle", "energy", "momentum"]
        field       ["phi", "apar", "bpar"] (the number appearing depends on nfield)
        species     list of species names (e.g. "electron", "ion1", "deuterium", etc)

    data_vars
        fields      (field, theta, kx, ky, time) complex array, may be zeros
        fluxes      (species, moment, field, ky, time) float array, may be zeros
        growth_rate            (kx, ky, time) float array, linear only
        mode_frequency         (kx, ky, time) float array, linear only
        eigenvalues            (kx, ky, time) float array, linear only
        eigenfunctions         (field, theta, kx, ky, time) float array, linear only
        growth_rate_tolerance  (kx, ky) float array, linear only

    attrs
        input_file   gk input file expressed as a string
        ntime        length of time coords
        nkx          length of kx coords
        nky          length of ky coords
        ntheta       length of theta coords
        nenergy      length of energy coords
        npitch       length of pitch coords
        nfield       length of field coords
        nspecies     length of species coords
    """

    def read(self, filename: PathLike, grt_time_range: float = 0.8) -> xr.Dataset:
        """
        Reads in GK output file to xarray Dataset
        """
        raw_data, gk_input, input_str = self._get_raw_data(filename)
        data = (
            self._init_dataset(raw_data, gk_input)
            .pipe(self._set_fields, raw_data, gk_input)
            .pipe(self._set_fluxes, raw_data, gk_input)
        )
        if gk_input.is_linear():
            data = data.pipe(self._set_eigenvalues, raw_data, gk_input).pipe(
                self._set_eigenfunctions, raw_data, gk_input
            )
            if "fields" in data:
                data = data.pipe(self._set_growth_rate_tolerance, grt_time_range)

        data.attrs.update(input_file=input_str)
        return data

    @abstractmethod
    def verify(self, filename: PathLike):
        """
        Ensure file is valid for a given GK output type.
        """
        pass

    @staticmethod
    @abstractmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_raw_data(filename: PathLike) -> Tuple[Any, GKInput, str]:
        """
        Read raw data from disk. The first returned value may take any type, depending
        on how the GK code handles output. For example, GS2 outputs a single NetCDF
        file, which can be handled with a single xarray Dataset, while CGYRO outputs
        multiple files with various formats, which are simply dumped in a dict.
        """
        pass

    @staticmethod
    @abstractmethod
    def _init_dataset(raw_data: Any, gk_input: GKInput) -> xr.Dataset:
        """
        Create a new dataset with coordinates and attrs set. Later functions
        will be tasked with filling in data_vars.
        """
        pass

    @staticmethod
    @abstractmethod
    def _set_fields(data: xr.Dataset, raw_data: Any, gk_input: GKInput) -> xr.Dataset:
        """
        Processes 3D field data over time, sets data["fields"] with the following
        coordinates:

        data['fields'] = fields(field, theta, kx, ky, time)

        This should be called after _init_dataset.

        Beyond the requirement to pass the incomplete Dataset as the first argument,
        the function signature is undefined for derived classes, as each gyrokinetics
        code has an idiosyncratic way of handling its output.
        """
        pass

    @staticmethod
    @abstractmethod
    def _set_fluxes(data: xr.Dataset, raw_data: Any, gk_input: GKInput) -> xr.Dataset:
        """
        Processes 3D flux data over time from raw_data, sets data["fluxes"] with
        the following coordinates:

        data['fluxes'] = fluxes(species, moment, field, ky, time)

        This should be called after _set_fields.

        Beyond the requirement to pass the incomplete Dataset as the first argument,
        the function signature is undefined for derived classes, as each gyrokinetics
        code has an idiosyncratic way of handling its output.
        """
        pass

    @staticmethod
    def _set_eigenvalues(
        data: xr.Dataset, raw_data: Optional[Any] = None, gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:

        data['eigenvalues'] = eigenvalues(kx, ky, time)
        data['mode_frequency'] = mode_frequency(kx, ky, time)
        data['growth_rate'] = growth_rate(kx, ky, time)

        This should be called after _set_fields, and is only valid for linear runs.

        Args:
            data (xr.Dataset): The dataset to be modified.
            raw_data (Optional): The raw data as produced by the GK code. May be needed
                by functions in derived classes, unused here.
            gk_input (Optional): The input file used to generate the data, expressed as
                a GKInput object. May be needed by functions in derived classes, unused
                here.
        Returns:
            xr.Dataset: The modified dataset which was passed to 'data'.

        """
        square_fields = np.abs(data["fields"]) ** 2
        field_amplitude = np.sqrt(
            square_fields.sum(dim="field").integrate(coord="theta")
        )

        growth_rate = np.log(field_amplitude).differentiate(coord="time").data

        field_angle = np.angle(data["fields"].sum(dim="field").integrate(coord="theta"))

        # Change angle by 2pi for every rotation so gradient is easier to calculate
        pi_change = np.cumsum(
            np.where(
                field_angle[:, :, :-1] * field_angle[:, :, 1:] < -pi,
                -np.sign(field_angle[:, :, 1:]),
                0,
            ),
            axis=-1,
        )
        field_angle[:, :, 1:] += 2 * pi * pi_change

        mode_frequency = -np.gradient(field_angle, axis=-1) / np.gradient(
            data["time"].data
        )

        eigenvalue = mode_frequency + 1j * growth_rate

        data["growth_rate"] = (("kx", "ky", "time"), growth_rate.data)
        data["mode_frequency"] = (("kx", "ky", "time"), mode_frequency)
        data["eigenvalues"] = (("kx", "ky", "time"), eigenvalue)
        return data

    @staticmethod
    def _set_eigenfunctions(
        data: xr.Dataset, raw_data: Optional[Any] = None, gk_input: Optional[Any] = None
    ) -> xr.Dataset:
        """
        Loads eigenfunctions into data with the following coordinates:

        data['eigenfunctions'] = eigenfunctions(kx, ky, field, theta, time)

        This should be called after _set_fields, and is only valid for linear runs.
        """
        if "fields" not in data:
            warnings.warn("'fields' not set in data, unable to compute eigenfunctions")
            return data

        eigenfunctions = data["fields"]

        square_fields = np.abs(data["fields"]) ** 2

        field_amplitude = np.sqrt(
            square_fields.sum(dim="field").integrate(coord="theta") / (2 * pi)
        )

        eigenfunctions = eigenfunctions / field_amplitude

        data["eigenfunctions"] = (
            ("field", "theta", "kx", "ky", "time"),
            eigenfunctions.data,
        )
        return data

    @staticmethod
    def _set_growth_rate_tolerance(data: xr.Dataset, time_range: float = 0.8):
        """
        Takes dataset that has already had growth_rate set. Sets the growth rate
        tolerance.

        Args:
            data (xr.Dataset): The dataset to be modified.
            time_range (float): Time range above which growth rate tolerance
                is calculated, as a fraction of the total time range. Takes values
                between 0.0 (100% of time steps used) and 1.0 (0% of time steps used).
        Returns:
            xr.Dataset: The modified dataset which was passed to 'data'.

        """
        data["growth_rate_tolerance"] = (
            ("kx", "ky"),
            get_growth_rate_tolerance(data, time_range=time_range),
        )
        return data


gk_output_readers = create_reader_factory(BaseReader=GKOutputReader)
