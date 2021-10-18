class Species:
    """
    Contains all species data as a function of psiN

    Charge
    Mass
    Density
    Temperature
    Rotation
    Omega

    Also need r/a (rho) as a function of psi for a/Lt etc.
    May need to add psi_toroidal
    """

    def __init__(
        self,
        species_type=None,
        charge=None,
        mass=None,
        dens=None,
        temp=None,
        rot=None,
        rho=None,
        ang=None,
    ):

        self.species_type = species_type
        self.charge = charge
        self.mass = mass
        self.dens = dens
        self.temp = temp
        self.rotation = rot
        self.omega = ang
        self.rho = rho
        self.grad_rho = self.rho.derivative()

    def get_mass(
        self,
    ):

        return self.mass

    def get_charge(
        self,
    ):

        return self.charge

    def get_dens(self, psi_n=None):

        return self.dens(psi_n)

    def get_norm_dens_gradient(self, psi_n=None):
        """
        - 1/n dn/rho
        """

        dens = self.get_dens(psi_n)
        grad_n = self.dens.derivative()(psi_n)
        grad_rho = self.grad_rho(psi_n)

        a_ln = -1 / dens * grad_n / grad_rho

        return a_ln

    def get_temp(self, psi_n=None):

        return self.temp(psi_n)

    def get_norm_temp_gradient(self, psi_n=None):
        """
        - 1/T dT/drho
        """

        temp = self.get_temp(psi_n)
        grad_t = self.temp.derivative()(psi_n)
        grad_rho = self.grad_rho(psi_n)

        a_lt = -1 / temp * grad_t / grad_rho

        return a_lt

    def get_velocity(self, psi_n=None):

        if self.rotation is None:
            vel = 0.0
        else:
            vel = self.rotation(psi_n)

        return vel

    def get_norm_vel_gradient(self, psi_n=None):
        """
        - 1/v dv/drho
        """
        vel = self.get_velocity(psi_n)

        if self.rotation is None:
            grad_v = 0
        else:
            grad_v = self.rotation.derivative()(psi_n)

        grad_rho = self.grad_rho(psi_n)

        if vel != 0.0:
            a_lv = -1 / vel * grad_v / grad_rho
        else:
            a_lv = 0.0

        return a_lv
