import numpy as np
import xarray as xr
from abc import abstractmethod
from typing import Optional

from .GKInput import GKInput
from ..typing import PathLike
from ..constants import pi
from ..readers import Reader, create_reader_factory

def get_growth_rate_tolerance( data: xr.Dataset, time_range=0.8):
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
    final_growth_rate = growth_rate.isel(time=-1).isel(ky=0)
    difference = abs((growth_rate - final_growth_rate) / final_growth_rate)
    final_time = difference["time"].isel(time=-1)
    # Average over the end of the simulation, starting at time_range*final_time
    tolerance = np.mean(difference.where(difference.time > time_range * final_time))
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
        growth_rate            (ky, time) float array, linear only 
        mode_frequency         (ky, time) float array, linear only
        eigenvalues            (ky, time) float array, linear only
        eigenfunctions         (field, theta, time) float array, linear only
        growth_rate_tolerance  (ZeroD) float, linear only

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
    @abstractmethod
    def read(self, filename: PathLike, grt_time_range : float = 0.8) -> xr.Dataset:
        """
        Reads in GK output file to xarray Dataset
        """
        pass

    @abstractmethod
    def verify(self, filename: PathLike):
        """
        Ensure file is valid for a given GK output type.
        """
        pass

    @staticmethod
    @abstractmethod
    def _init_dataset(raw_data: xr.Dataset, gk_input: Optional[GKInput] = None) -> xr.Dataset:
        """
        Given an xarray dataset containing the raw output data of a given gyrokinetics
        code, create a new dataset with coordinates and attrs set. Later functions
        will be tasked with filling in data_vars.
        This function may also be passed the input data of a gyrokinetics run, as this
        may be necessary to understand the output.
        """
        pass

    @staticmethod
    @abstractmethod
    def _set_fields(data: xr.Dataset, raw_data: xr.Dataset) -> xr.Dataset:
        """
        Processes 3D field data over time from raw_data, sets data["fields"]

        data['fields'] = fields(field, theta, kx, ky, time)
        """
        pass

    @staticmethod
    @abstractmethod
    def _set_fluxes(data: xr.Dataset, raw_data: xr.Dataset) -> xr.Dataset:
        """
        Processes 3D flux data over time from raw_data, sets data["fluxes"]

        data['fluxes'] = fluxes(species, moment, field, ky, time)
        """
        pass

    @staticmethod
    def _set_eigenvalues(data: xr.Dataset, grt_time_range: float) -> xr.Dataset:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:
        
        data['eigenvalues'] = eigenvalues(ky, time)
        data['mode_frequency'] = mode_frequency(ky, time)
        data['growth_rate'] = growth_rate(ky, time)
        
        Args:
            data (xr.Dataset): The dataset to be modified.
            grt_time_range (float): Time range above which growth rate tolerance
                is calculated, as a fraction of the total time range. Takes values
                between 0.0 (100% of time steps used) and 1.0 (0% of time steps used).

        Returns:
            xr.Dataset: The modified dataset which was passed to 'data'.

        """

        square_fields = np.abs(data["fields"]) ** 2
        field_amplitude = np.sqrt(
            square_fields.sum(dim="field").integrate(coord="theta") / (2 * pi)
        )

        growth_rate = (
            np.log(field_amplitude)
            .differentiate(coord="time")
            .squeeze(dim="kx", drop=True)
            .data
        )

        field_angle = np.angle(
            data["fields"]
            .sum(dim="field")
            .integrate(coord="theta")
            .squeeze(dim=["kx", "ky"], drop=True)
        )

        # Change angle by 2pi for every rotation so gradient is easier to calculate
        pi_change = field_angle * 0
        rotation_number = 0
        for i in range(len(field_angle) - 1):
            if field_angle[i] * field_angle[i + 1] < -pi:
                rotation_number -= field_angle[i + 1] / abs(field_angle[i + 1])

            pi_change[i + 1] = rotation_number

        field_angle = field_angle + (pi * 2) * pi_change

        mode_frequency = -np.gradient(field_angle) / np.gradient(data["time"].data)
        mode_frequency = mode_frequency[np.newaxis, :]

        eigenvalue = mode_frequency + 1j * growth_rate

        data["growth_rate"] = (("ky", "time"), growth_rate.data)
        data["mode_frequency"] = (("ky", "time"), mode_frequency)
        data["eigenvalues"] = (("ky", "time"), eigenvalue)
        data["growth_rate_tolerance"] = get_growth_rate_tolerance(data, grt_time_range)
        return data

    @staticmethod
    def _set_eigenfunctions(data : xr.Dataset) -> xr.Dataset:
        """
        Loads eigenfunctions into data
        data['eigenfunctions'] = eigenfunctions(field, theta, time)

        """
        eigenfunctions = data["fields"].isel({"ky": 0}).isel({"kx": 0})

        square_fields = np.abs(data["fields"]) ** 2

        field_amplitude = np.sqrt(
            square_fields.sum(dim="field").integrate(coord="theta") / (2 * pi)
        )

        eigenfunctions = eigenfunctions / field_amplitude
        eigenfunctions = eigenfunctions.squeeze(dim=["kx", "ky"], drop=True)

        data["eigenfunctions"] = (("field", "theta", "time"), eigenfunctions.data)
        return data

    @classmethod
    def _build_dataset(cls, raw_data: xr.Dataset, gk_input: Optional[GKInput] = None, is_linear: bool = True, grt_time_range: float = 0.8) -> xr.Dataset:
        """
        Given an xarray dataset containing the raw output data of a given gyrokinetics
        code, builds a dataset in the standard form determined by Pyrokinetics. Calls
        the other static methods of this class in the correct sequence.
        """
        data = cls._init_dataset(raw_data, gk_input=gk_input)
        data = cls._set_fields(data, raw_data)
        data = cls._set_fluxes(data, raw_data)
        if is_linear and data.dims["kx"] == 1 and data.dims["ky"] == 1:
            data = cls._set_eigenvalues(data, grt_time_range=grt_time_range)
            data = cls._set_eigenfunctions(data)
        return data

gk_output_readers = create_reader_factory(BaseReader=GKOutputReader)
