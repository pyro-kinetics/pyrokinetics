from numpy import sqrt
import numpy as np


class Equilibrium:
    """
    Partially adapted from equilibrium option developed by
    B. Dudson in FreeGS

    Equilibrium object containing

    R
    Z
    Psi(R,Z)
    q (Psi)
    f (Psi)
    ff'(Psi)
    B
    Bp
    get_b_toroidal
    """

    def __init__(self, eq_file=None, eq_type=None, **kwargs):

        self.eq_file = eq_file
        self.eq_type = eq_type

        self.nr = None
        self.nz = None
        self.psi = None
        self.psi_RZ = None
        self.pressure = None
        self.p_prime = None
        self.f_psi = None
        self.ff_prime = None
        self.q = None
        self.Bp = None
        self.Bt = None
        self.a_minor = None
        self.current = None

        if self.eq_file is not None:
            if self.eq_type == "GEQDSK":
                self.read_geqdsk(**kwargs)

            elif self.eq_type is None:
                raise ValueError("Please specify the type of equilibrium")
            else:
                raise NotImplementedError(
                    f"Equilibrium type {self.eq_type} not yet implemented"
                )

    def get_b_radial(self, R, Z):

        b_radial = -1 / R * self.psi_RZ(R, Z, dy=1, grid=False)

        return b_radial

    def get_b_vertical(self, R, Z):

        b_vertical = 1 / R * self.psi_RZ(R, Z, dx=1, grid=False)

        return b_vertical

    def get_b_poloidal(self, R, Z):

        b_radial = self.get_b_radial(R, Z)
        b_vertical = self.get_b_vertical(R, Z)

        b_poloidal = sqrt(b_radial ** 2 + b_vertical ** 2)

        return b_poloidal

    def get_b_toroidal(self, R, Z):

        psi = self.psi_RZ(R, Z, grid=False)

        psi_n = (psi - self.psi_axis) / (self.psi_bdry - self.psi_axis)
        f = self.f_psi(psi_n)

        b_tor = f / R

        return b_tor

    def read_geqdsk(self, psi_n_lcfs=1.0):
        """

        Read in GEQDSK file and populates Equilibrium object

        """

        from freegs import _geqdsk
        from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
        from numpy import linspace

        f = open(self.eq_file)
        gdata = _geqdsk.read(f)
        f.close()

        # Assign gdata to Equilibriun object

        self.nr = gdata["nx"]
        self.nz = gdata["ny"]

        psi_RZ = gdata["psi"]
        self.bcentr = gdata["bcentr"]

        self.psi_axis = gdata["simagx"]
        self.psi_bdry = gdata["sibdry"]

        psi_n = linspace(0.0, psi_n_lcfs, self.nr)

        # Set up 1D profiles as interpolated functions
        self.f_psi = InterpolatedUnivariateSpline(psi_n, gdata["fpol"])

        self.ff_prime = InterpolatedUnivariateSpline(psi_n, gdata["ffprime"])

        self.q = InterpolatedUnivariateSpline(psi_n, gdata["qpsi"])

        self.pressure = InterpolatedUnivariateSpline(psi_n, gdata["pres"])

        self.p_prime = self.pressure.derivative()

        # Set up 2D psi_RZ grid
        self.R = linspace(gdata["rleft"], gdata["rleft"] + gdata["rdim"], self.nr)

        self.Z = linspace(
            gdata["zmid"] - gdata["zdim"] / 2,
            gdata["zmid"] + gdata["zdim"] / 2,
            self.nz,
        )

        self.psi_RZ = RectBivariateSpline(self.R, self.Z, psi_RZ)

        rho = psi_n * 0.0
        R_major = psi_n * 0.0

        for i, i_psiN in enumerate(psi_n[1:]):

            surface_R, surface_Z = self.get_flux_surface(psi_n=i_psiN)

            rho[i + 1] = (max(surface_R) - min(surface_R)) / 2
            R_major[i + 1] = (max(surface_R) + min(surface_R)) / 2

        self.lcfs_R = surface_R
        self.lcfs_Z = surface_Z

        self.a_minor = rho[-1]

        rho = rho / rho[-1]

        R_major[0] = R_major[1] + psi_n[1] * (R_major[2] - R_major[1]) / (
            psi_n[2] - psi_n[1]
        )

        self.rho = InterpolatedUnivariateSpline(psi_n, rho)

        self.R_major = InterpolatedUnivariateSpline(psi_n, R_major)

    def get_flux_surface(
        self,
        psi_n=None,
    ):

        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        if psi_n is None:
            raise ValueError("get_flux_surface needs a psi_n")

        # Generate 2D mesh of normalised psi
        psi_2d = np.transpose(self.psi_RZ(self.R, self.Z))

        psin_2d = (psi_2d - self.psi_axis) / (self.psi_bdry - self.psi_axis)

        # Returns a list of list of contours for psi_n
        con = plt.contour(self.R, self.Z, psin_2d, levels=[0, psi_n])

        paths = con.collections[1].get_paths()

        if psi_n == 1.0 and len(paths) == 0:
            raise ValueError(
                "PsiN=1.0 for LCFS isn't well defined. Try lowering psi_n_lcfs"
            )

        path = paths[np.argmax(len(paths))]

        R_con, Z_con = path.vertices[:, 0], path.vertices[:, 1]

        # Start from OMP
        Z_con = np.flip(np.roll(Z_con, -np.argmax(R_con) - 1))
        R_con = np.flip(np.roll(R_con, -np.argmax(R_con) - 1))

        return R_con, Z_con

    def generate_local(
        self,
        geometry_type=None,
    ):

        raise NotImplementedError
