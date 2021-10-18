from scipy.interpolate import InterpolatedUnivariateSpline
from .constants import electron_mass, deuterium_mass, hydrogen_mass
import netCDF4 as nc
from .species import Species
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

    def __init__(self, kinetics_file=None, kinetics_type=None, nspec=None):

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
            self.read_transp()
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

    def read_transp(self):
        """
        Reads in TRANSP profiles NetCDF file


        """

        kinetics_data = nc.Dataset(self.kinetics_file)

        psi = kinetics_data["PLFLX"][-1, :].data
        psi = psi - psi[0]
        psi_n = psi / psi[-1]

        rho = kinetics_data["RMNMP"][-1, :].data
        rho = rho / rho[-1]

        rho_func = InterpolatedUnivariateSpline(psi_n, rho)

        nspec = 1

        electron_temp_data = kinetics_data["TE"][-1, :].data
        electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

        electron_dens_data = kinetics_data["NE"][-1, :].data * 1e6
        electron_dens_func = InterpolatedUnivariateSpline(psi_n, electron_dens_data)

        omega_data = kinetics_data["OMEG_VTR"][-1, :].data
        omega_func = InterpolatedUnivariateSpline(psi_n, omega_data)

        electron = Species(
            species_type="electron",
            charge=-1,
            mass=electron_mass,
            dens=electron_dens_func,
            temp=electron_temp_func,
            ang=omega_func,
            rho=rho_func,
        )

        self.species_data.electron = electron

        # TRANSP only has one ion temp
        ion_temp_data = kinetics_data["TI"][-1, :].data
        ion_temp_func = InterpolatedUnivariateSpline(psi_n, ion_temp_data)

        # Go through each species output in TRANSP

        # Deuterium
        try:
            deuterium_dens_data = kinetics_data["ND"][-1, :].data * 1e6

            nspec += 1
            deuterium_dens_func = InterpolatedUnivariateSpline(
                psi_n, deuterium_dens_data
            )

            deuterium = Species(
                species_type="deuterium",
                charge=1,
                mass=deuterium_mass,
                dens=deuterium_dens_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

            self.species_data.deuterium = deuterium

        except IndexError:
            pass

        # Tritium
        try:
            tritium_dens_data = kinetics_data["NT"][-1, :].data * 1e6

            nspec += 1
            tritium_dens_func = InterpolatedUnivariateSpline(psi_n, tritium_dens_data)

            tritium = Species(
                species_type="tritium",
                charge=1,
                mass=1.5 * deuterium_mass,
                dens=tritium_dens_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

            self.species_data.tritium = tritium

        except IndexError:
            pass

        # Helium 4
        try:
            helium4_dens_data = kinetics_data["NI4"][-1, :].data * 1e6

            nspec += 1
            helium_dens_func = InterpolatedUnivariateSpline(psi_n, helium4_dens_data)

            helium = Species(
                species_type="helium",
                charge=2,
                mass=4 * hydrogen_mass,
                dens=helium_dens_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

            self.species_data.helium = helium
        except IndexError:
            pass

        # Helium 3
        try:
            helium3_dens_data = kinetics_data["NI3"][-1, :].data * 1e6

            nspec += 1
            helium3_dens_func = InterpolatedUnivariateSpline(psi_n, helium3_dens_data)

            helium3 = Species(
                species_type="helium3",
                charge=2,
                mass=4 * hydrogen_mass,
                dens=helium3_dens_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

            self.species_data.helium3 = helium3
        except IndexError:
            pass

        try:
            impurity_dens_data = kinetics_data["NIMP"][-1, :].data * 1e6

            nspec += 1
            impurity_dens_func = InterpolatedUnivariateSpline(psi_n, impurity_dens_data)

            Z = int(kinetics_data["XZIMP"][-1].data)
            M = int(kinetics_data["AIMP"][-1].data)

            impurity = Species(
                species_type="impurity",
                charge=Z,
                mass=M * hydrogen_mass,
                dens=impurity_dens_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

            self.species_data.impurity = impurity

        except IndexError:
            pass

        self.nspec = nspec
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

        nspec = 1

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

        # Deuterium data
        deuterium_dens_data = kinetics_data["NID"][-1, :].data

        if any(deuterium_dens_data):
            nspec += 1
            deuterium_dens_func = InterpolatedUnivariateSpline(
                psi_n, deuterium_dens_data
            )

            deuterium = Species(
                species_type="deuterium",
                charge=1,
                mass=deuterium_mass,
                dens=deuterium_dens_func,
                temp=ion_temp_func,
                rot=rotation_func,
                rho=rho_func,
            )

            self.species_data.deuterium = deuterium

        # Tritium data
        tritium_dens_data = kinetics_data["NIT"][-1, :].data

        if any(tritium_dens_data):
            nspec += 1
            tritium_dens_func = InterpolatedUnivariateSpline(psi_n, tritium_dens_data)

            tritium = Species(
                species_type="tritium",
                charge=1,
                mass=3 * hydrogen_mass,
                dens=tritium_dens_func,
                temp=ion_temp_func,
                rot=rotation_func,
                rho=rho_func,
            )

            self.species_data.tritium = tritium

        # Helium data
        alpha_dens_data = kinetics_data["NALF"][-1, :].data

        if any(alpha_dens_data):
            nspec += 1
            alpha_dens_func = InterpolatedUnivariateSpline(psi_n, alpha_dens_data)

            helium = Species(
                species_type="helium",
                charge=2,
                mass=2 * deuterium_mass,
                dens=alpha_dens_func,
                temp=ion_temp_func,
                rot=rotation_func,
                rho=rho_func,
            )

            self.species_data.helium = helium

        # Impurity data
        impurity_dens_data = kinetics_data["NIMP"][-1, :].data

        if any(impurity_dens_data):
            nspec += 1
            impurity_dens_func = InterpolatedUnivariateSpline(psi_n, impurity_dens_data)

            Z = int(kinetics_data["ZIA1"][-1, 0].data)
            M = get_impurity_mass(Z)

            impurity = Species(
                species_type="impurity",
                charge=Z,
                mass=M * hydrogen_mass,
                dens=impurity_dens_func,
                temp=ion_temp_func,
                rot=rotation_func,
                rho=rho_func,
            )

            self.species_data.impurity = impurity

        self.nspec = nspec
        self.species_names = [*self.species_data.keys()]


def get_impurity_mass(Z=None):
    """Get impurity mass from charge"""

    Zlist = [2, 6, 8, 10, 18, 54, 74]
    Mlist = [4, 12, 16, 20, 40, 132, 184]

    M = Mlist[Zlist.index(Z)]

    return M


def get_impurity_charge(M=None):
    """Get impurity charge from mass"""

    Zlist = [2, 6, 8, 10, 18, 54, 74]
    Mlist = [4, 12, 16, 20, 40, 132, 184]

    Z = Zlist[Mlist.index(M)]
