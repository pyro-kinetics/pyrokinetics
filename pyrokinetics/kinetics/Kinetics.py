from scipy.interpolate import InterpolatedUnivariateSpline
from ..constants import electron_mass, deuterium_mass, hydrogen_mass
import netCDF4 as nc
from ..species import Species
from cleverdict import CleverDict


class Kinetics:
    """
    Contains all the kinetic data
    made up of Species object

    CleverDict of Species with key being species name

    kinetics['electron'] = Species(electron)

    Need to do this for all species

    Needs:
    psi_n
    r/a
    Temperature
    Density
    Rotation
    """

    # Define class level info
    supported_kinetics_types = ["SCENE", "JETTO", "TRANSP", None]

    def __init__(self, kinetics_file=None, kinetics_type=None, nspec=None, **kwargs):

        self.kinetics_file = kinetics_file
        self.kinetics_type = kinetics_type
        self.nspec = nspec

        self.species_data = CleverDict()
        self.species_names = []

        if self.kinetics_type == "SCENE":
            self.read_scene()
        elif self.kinetics_type == "JETTO":
            self.read_jetto()
        elif self.kinetics_type == "TRANSP":
            self.read_transp(**kwargs)
        elif self.kinetics_type is None:
            pass
        else:
            raise ValueError(
                f"{self.kinetics_type} as a source of Kinetic profiles not currently supported"
            )

    def read_scene(self):
        """

        Read NetCDF file from SCENE code
        Assumes 3 species: e, D, T

        """

        kinetics_data = nc.Dataset(self.kinetics_file)

        self.nspec = 3

        psi = kinetics_data["Psi"][::-1]
        psi_n = psi / psi[-1]

        rho = kinetics_data["TGLF_RMIN"][::-1]
        rho_func = InterpolatedUnivariateSpline(psi_n, rho)

        electron_temp_data = kinetics_data["Te"][::-1]
        electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

        electron_density_data = kinetics_data["Ne"][::-1]
        electron_density_func = InterpolatedUnivariateSpline(
            psi_n, electron_density_data
        )

        electron_rotation_data = electron_temp_data * 0.0
        electron_rotation_func = InterpolatedUnivariateSpline(
            psi_n, electron_rotation_data
        )

        electron = Species(
            species_type="electron",
            charge=-1,
            mass=electron_mass,
            dens=electron_density_func,
            temp=electron_temp_func,
            rot=electron_rotation_func,
            rho=rho_func,
        )

        self.species_data.electron = electron

        ion_temperature_func = electron_temp_func
        ion_rotation_func = electron_rotation_func

        ion_density_func = InterpolatedUnivariateSpline(
            psi_n, electron_density_data / 2
        )

        deuterium = Species(
            species_type="deuterium",
            charge=1,
            mass=deuterium_mass,
            dens=ion_density_func,
            temp=ion_temperature_func,
            rot=ion_rotation_func,
            rho=rho_func,
        )

        self.species_data.deuterium = deuterium

        tritium = Species(
            species_type="tritium",
            charge=1,
            mass=1.5 * deuterium_mass,
            dens=ion_density_func,
            temp=ion_temperature_func,
            rot=ion_rotation_func,
            rho=rho_func,
        )

        self.species_data.tritium = tritium

        self.species_names = [*self.species_data.keys()]

    def read_transp(self, time_index: int = -1, time: float = None):
        """
        Reads in TRANSP profiles NetCDF file


        """
        import numpy as np

        kinetics_data = nc.Dataset(self.kinetics_file)

        time_cdf = kinetics_data["TIME3"][:]

        if time_index != -1 and time is not None:
            raise ValueError("Cannot set both `time` and `time_index`")

        if time is not None:
            time_index = np.argmin(np.abs(time_cdf - time))

        psi = kinetics_data["PLFLX"][time_index, :].data
        psi = psi - psi[0]
        psi_n = psi / psi[-1]

        rho = kinetics_data["RMNMP"][time_index, :].data
        rho = rho / rho[-1]

        rho_func = InterpolatedUnivariateSpline(psi_n, rho)

        electron_temp_data = kinetics_data["TE"][time_index, :].data
        electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

        electron_dens_data = kinetics_data["NE"][time_index, :].data * 1e6
        electron_dens_func = InterpolatedUnivariateSpline(psi_n, electron_dens_data)

        try:
            omega_data = kinetics_data["OMEG_VTR"][time_index, :].data
        except IndexError:
            omega_data = electron_dens_data * 0.0

        omega_func = InterpolatedUnivariateSpline(psi_n, omega_data)

        self.species_data.electron = Species(
            species_type="electron",
            charge=-1,
            mass=electron_mass,
            dens=electron_dens_func,
            temp=electron_temp_func,
            ang=omega_func,
            rho=rho_func,
        )

        # TRANSP only has one ion temp
        ion_temp_data = kinetics_data["TI"][time_index, :].data
        ion_temp_func = InterpolatedUnivariateSpline(psi_n, ion_temp_data)

        # Go through each species output in TRANSP
        try:
            impurity_charge = int(kinetics_data["XZIMP"][time_index].data)
            impurity_mass = int(kinetics_data["AIMP"][time_index].data) * hydrogen_mass
        except IndexError:
            impurity_charge = 0
            impurity_mass = 0

        possible_species = [
            {
                "species_name": "deuterium",
                "transp_name": "ND",
                "charge": 1,
                "mass": deuterium_mass,
            },
            {
                "species_name": "tritium",
                "transp_name": "NT",
                "charge": 1,
                "mass": 1.5 * deuterium_mass,
            },
            {
                "species_name": "helium",
                "transp_name": "NI4",
                "charge": 2,
                "mass": 4 * hydrogen_mass,
            },
            {
                "species_name": "helium3",
                "transp_name": "NI4",
                "charge": 2,
                "mass": 3 * hydrogen_mass,
            },
            {
                "species_name": "impurity",
                "transp_name": "NIMP",
                "charge": impurity_charge,
                "mass": impurity_mass,
            },
        ]

        for species in possible_species:
            if species["transp_name"] not in kinetics_data.variables:
                continue

            density_data = (
                kinetics_data[species["transp_name"]][time_index, :].data * 1e6
            )
            density_func = InterpolatedUnivariateSpline(psi_n, density_data)

            self.species_data[species["species_name"]] = Species(
                species_type=species["species_name"],
                charge=species["charge"],
                mass=species["mass"],
                dens=density_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

        self.nspec = len(self.species_data)
        self.species_names = [*self.species_data.keys()]

    def read_jetto(self):
        """
        Reads in JETTO profiles NetCDF file
        Loads Kinetics object

        """

        kinetics_data = nc.Dataset(self.kinetics_file)

        psi = kinetics_data["PSI"][-1, :].data
        psi = psi - psi[0]
        psi_n = psi / psi[-1]

        rho = kinetics_data["RMNMP"][-1, :].data
        rho_func = InterpolatedUnivariateSpline(psi_n, rho)

        # Electron data
        electron_temp_data = kinetics_data["TE"][-1, :].data
        electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

        electron_dens_data = kinetics_data["NE"][-1, :].data
        electron_dens_func = InterpolatedUnivariateSpline(psi_n, electron_dens_data)

        rotation_data = kinetics_data["VTOR"][-1, :].data
        rotation_func = InterpolatedUnivariateSpline(psi_n, rotation_data)

        electron = Species(
            species_type="electron",
            charge=-1,
            mass=electron_mass,
            dens=electron_dens_func,
            temp=electron_temp_func,
            rot=rotation_func,
            rho=rho_func,
        )

        self.species_data.electron = electron

        # JETTO only has one ion temp
        ion_temp_data = kinetics_data["TI"][-1, :].data
        ion_temp_func = InterpolatedUnivariateSpline(psi_n, ion_temp_data)

        # Go through each species output in JETTO
        try:
            impurity_charge = int(kinetics_data["ZIA1"][-1, 0].data)
            impurity_mass = get_impurity_mass(impurity_charge) * hydrogen_mass
        except IndexError:
            impurity_charge = 0
            impurity_mass = 0

        possible_species = [
            {
                "species_name": "deuterium",
                "jetto_name": "NID",
                "charge": 1,
                "mass": deuterium_mass,
            },
            {
                "species_name": "tritium",
                "jetto_name": "NIT",
                "charge": 1,
                "mass": 1.5 * deuterium_mass,
            },
            {
                "species_name": "helium",
                "jetto_name": "NALF",
                "charge": 2,
                "mass": 4 * hydrogen_mass,
            },
            {
                "species_name": "impurity",
                "jetto_name": "NIMP",
                "charge": impurity_charge,
                "mass": impurity_mass,
            },
        ]

        for species in possible_species:
            density_data = kinetics_data[species["jetto_name"]][-1, :].data
            if not any(density_data):
                continue

            density_func = InterpolatedUnivariateSpline(psi_n, density_data)

            self.species_data[species["species_name"]] = Species(
                species_type=species["species_name"],
                charge=species["charge"],
                mass=species["mass"],
                dens=density_func,
                temp=ion_temp_func,
                rot=rotation_func,
                rho=rho_func,
            )

        self.nspec = len(self.species_data)
        self.species_names = [*self.species_data.keys()]

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Kinetics object

        Returns
        -------
        Copy of kinetics object
        """

        new_kinetics = Kinetics()

        for key, value in self.__dict__.items():
            if key == "species_data":
                pass
            else:
                setattr(new_kinetics, key, value)

        # Manually add each species
        for name, species in self.species_data.items():
            new_kinetics.species_data["name"] = species
        return new_kinetics


def get_impurity_mass(Z=None):
    """Get impurity mass from charge"""

    Zlist = [2, 6, 8, 10, 18, 54, 74]
    Mlist = [4, 12, 16, 20, 40, 132, 184]

    return Mlist[Zlist.index(Z)]


def get_impurity_charge(M=None):
    """Get impurity charge from mass"""

    Zlist = [2, 6, 8, 10, 18, 54, 74]
    Mlist = [4, 12, 16, 20, 40, 132, 184]

    return Zlist[Mlist.index(M)]
