from ..decorators import not_implemented
from ..constants import pi
from ..readers import Reader, create_reader_factory
import numpy as np


class GKCode(Reader):
    """
    Basic GK code object
    """

    def __init__(self):

        self.code_name = None

        pass

    @not_implemented
    def read(self, pyro):
        """
        Reads in GK input file into Pyro object
        as a dictionary
        """
        pass

    @not_implemented
    def verify(self, filename):
        """
        Ensure file is valid for a given GK input type.
        """
        pass

    @not_implemented
    def load_pyro(self, pyro):
        """
        Loads GK dictionary into Pyro object
        """
        pass

    @not_implemented
    def write(self, pyro):
        """
        For a given pyro object write a GK code input file

        """
        pass

    @not_implemented
    def load_local_geometry(self, pyro, code):
        """
        Load local geometry object from a GK code input file
        """
        pass

    @not_implemented
    def load_miller(self, pyro, code):
        """
        Load Miller object from a GK code input file
        """
        pass

    @not_implemented
    def load_local_species(self, pyro, code):
        """
        Load local species object from a GK code input file
        """
        pass

    @not_implemented
    def add_flags(self, pyro, flags):
        """
        Add extra flags to a GK code input file

        """
        pass

    @not_implemented
    def load_numerics(self, pyro, code):
        """
        Load Numerics object from a GK code input file
        """
        pass

    @not_implemented
    def pyro_to_code_miller(self):
        """
        Generates dictionary of equivalent pyro and gk code parameter names
        for miller parameters
        """
        pass

    @not_implemented
    def pyro_to_code_species(self):
        """
        Generates dictionary of equivalent pyro and gk code parameter names
        for miller parameters
        """
        pass

    @not_implemented
    def run(self):
        """
        Runs GK code
        """
        pass

    @not_implemented
    def load_gk_output(self):
        """
        Loads GKOutput object with simulation data
        """
        pass

    def load_eigenvalues(self, pyro):
        """
        Loads eigenvalues into GKOutput.data DataSet
        pyro.gk_output.data['eigenvalues'] = eigenvalues(ky, time)
        pyro.gk_output.data['mode_frequency'] = mode_frequency(ky, time)
        pyro.gk_output.data['growth_rate'] = growth_rate(ky, time)

        """

        gk_output = pyro.gk_output
        data = gk_output.data

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

        self.get_growth_rate_tolerance(pyro)

    def load_eigenfunctions(self, pyro):
        """
        Loads eigenfunctions into GKOutput.data Dataset
        pyro.gk_output.data['eigenfunctions'] = eigenvalues(field, theta, time)

        """

        gk_output = pyro.gk_output
        data = gk_output.data

        eigenfunctions = data["fields"].isel({"ky": 0}).isel({"kx": 0})

        square_fields = np.abs(data["fields"]) ** 2

        field_amplitude = np.sqrt(
            square_fields.sum(dim="field").integrate(coord="theta") / (2 * pi)
        )

        eigenfunctions = eigenfunctions / field_amplitude
        eigenfunctions = eigenfunctions.squeeze(dim=["kx", "ky"], drop=True)

        data["eigenfunctions"] = (("field", "theta", "time"), eigenfunctions.data)

    def get_growth_rate_tolerance(self, pyro, time_range=0.8):
        """
        Calculate tolerance on the growth rate

        time_range: time range above which tolerance is calculated
        """

        growth_rate = pyro.gk_output.data["growth_rate"]

        final_growth_rate = growth_rate.isel(time=-1).isel(ky=0)

        difference = abs((growth_rate - final_growth_rate) / final_growth_rate)

        final_time = difference["time"].isel(time=-1)

        # Average over final 20% of simulation

        tolerance = np.mean(difference.where(difference.time > time_range * final_time))

        pyro.gk_output.data["growth_rate_tolerance"] = tolerance


gk_codes = create_reader_factory(BaseReader=GKCode)
