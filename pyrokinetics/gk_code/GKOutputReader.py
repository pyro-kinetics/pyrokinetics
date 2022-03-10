import numpy as np
import xarray as xr
from abc import abstractmethod

from ..typing import PathLike
from ..constants import pi
from ..readers import Reader, create_reader_factory


class GKOutputReader(Reader):
    @abstractmethod
    def read(self, filename: PathLike) -> xr.Dataset:
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

    @abstractmethod
    def set_grids(self):
        """reads in numerical grids"""
        pass

    @abstractmethod
    def set_fields(self):
        """reads in 3D fields"""
        pass

    @abstractmethod
    def set_fluxes(self):
        """reads in fields"""
        pass

    def set_eigenvalues(self):
        """
        Loads eigenvalues into self.data
        gk_output.data['eigenvalues'] = eigenvalues(ky, time)
        gk_output.data['mode_frequency'] = mode_frequency(ky, time)
        gk_output.data['growth_rate'] = growth_rate(ky, time)

        """
        square_fields = np.abs(self.data["fields"]) ** 2
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
            self.data["fields"]
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

        mode_frequency = -np.gradient(field_angle) / np.gradient(self.data["time"].data)
        mode_frequency = mode_frequency[np.newaxis, :]

        eigenvalue = mode_frequency + 1j * growth_rate

        self.data["growth_rate"] = (("ky", "time"), growth_rate.data)
        self.data["mode_frequency"] = (("ky", "time"), mode_frequency)
        self.data["eigenvalues"] = (("ky", "time"), eigenvalue)

        self.set_growth_rate_tolerance()

    def set_eigenfunctions(self):
        """
        Loads eigenfunctions into self.data
        gk_output.data['eigenfunctions'] = eigenvalues(field, theta, time)

        """

        eigenfunctions = self.data["fields"].isel({"ky": 0}).isel({"kx": 0})

        square_fields = np.abs(self.data["fields"]) ** 2

        field_amplitude = np.sqrt(
            square_fields.sum(dim="field").integrate(coord="theta") / (2 * pi)
        )

        eigenfunctions = eigenfunctions / field_amplitude
        eigenfunctions = eigenfunctions.squeeze(dim=["kx", "ky"], drop=True)

        self.data["eigenfunctions"] = (("field", "theta", "time"), eigenfunctions.data)

    def set_growth_rate_tolerance(self, time_range=0.8):
        """
        Calculate tolerance on the growth rate

        time_range: time range above which tolerance is calculated
        """

        growth_rate = self.data["growth_rate"]

        final_growth_rate = growth_rate.isel(time=-1).isel(ky=0)

        difference = abs((growth_rate - final_growth_rate) / final_growth_rate)

        final_time = difference["time"].isel(time=-1)

        # Average over final 20% of simulation

        tolerance = np.mean(difference.where(difference.time > time_range * final_time))

        self.data["growth_rate_tolerance"] = tolerance


gk_output_readers = create_reader_factory(BaseReader=GKOutputReader)
