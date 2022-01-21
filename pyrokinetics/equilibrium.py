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
        self.rho = None
        self.R_major = None

        if self.eq_file is not None:
            if self.eq_type == "GEQDSK":
                self.read_geqdsk(**kwargs)
            elif self.eq_type == "TRANSP":
                self.read_transp_cdf(**kwargs)
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

        with open(self.eq_file) as f:
            gdata = _geqdsk.read(f)

        # Assign gdata to Equilibriun object

        self.nr = gdata["nx"]
        self.nz = gdata["ny"]

        psi_RZ = gdata["psi"]
        self.bcentr = gdata["bcentr"]

        self.psi_axis = gdata["simagx"]
        self.psi_bdry = gdata["sibdry"]

        self.R_axis = gdata["rmagx"]
        self.Z_axis = gdata["zmagx"]

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

    def read_transp_cdf(self, time=-1):
        """

        Read in TRANSP netCDF and populates Equilibrium object

        """

        import netCDF4 as nc
        import numpy as np
        from scipy.interpolate import (
            RBFInterpolator,
            InterpolatedUnivariateSpline,
            RectBivariateSpline,
        )

        data = nc.Dataset(self.eq_file)

        nr = len(data["XB"][-1, :])

        time_cdf = data["TIME3"][:]

        if isinstance(time, int):
            itime = time
        elif isinstance(time, float):
            itime = np.argmin(abs(time_cdf - time))
        else:
            raise ValueError("time input needs to be float or int")

        ntheta = 256
        theta = np.linspace(0, 2 * np.pi, ntheta)
        theta = theta[:, np.newaxis]
        
        # No. of moments stored in TRANSP
        nmoments = 17

        # Calculate flux surfaces from moments up to 17
        R_mom_cos = np.empty((nmoments, ntheta, nr))
        R_mom_sin = np.empty((nmoments, ntheta, nr))
        Z_mom_cos = np.empty((nmoments, ntheta, nr))
        Z_mom_sin = np.empty((nmomemts, ntheta, nr))

        for i in range(nmoments):
            try:
                R_mom_cos[i, :, :] = (
                    np.cos(i * theta) * data[f"RMC{i:02d}"][itime, :] * 1e-2
                )
            except IndexError:
                break
            Z_mom_cos[i, :, :] = (
                np.cos(i * theta) * data[f"YMC{i:02d}"][itime, :] * 1e-2
            )
            
            # TRANSP doesn't stored 0th sin moment = 0.0 by defn
            if i == 0:
                R_mom_sin[i, :, :] = 0.0
                Z_mom_sin[i, :, :] = 0.0
            else:
                R_mom_sin[i, :, :] = (
                    np.sin(i * theta) * data[f"RMS{i:02d}"][itime, :] * 1e-2
                )
                Z_mom_sin[i, :, :] = (
                    np.sin(i * theta) * data[f"YMS{i:02d}"][itime, :] * 1e-2
                )

        Rsur = np.sum(R_mom_cos, axis=0) + np.sum(R_mom_sin, axis=0)
        Zsur = np.sum(Z_mom_cos, axis=0) + np.sum(Z_mom_sin, axis=0)

        psi_axis = data["PSI0_TR"][itime]
        psi_bdry = data["PLFLXA"][itime]

        current = data["PCUR"][itime]

        # Load in 1D profiles
        q = data["Q"][itime, :]
        press = data["PMHD_IN"][itime, :]

        # F is on a different grid and need to remove duplicated HFS points
        psi_rmajm = data["PLFMP"][itime, :]
        rmajm_ax = np.argmin(psi_rmajm)
        psi_n_rmajm = psi_rmajm[rmajm_ax:] / psi_rmajm[-1]

        # f = (Bt / |B|) * |B| *  R
        f = (
            data["FBTX"][itime, rmajm_ax:]
            * data["BTX"][itime, rmajm_ax:]
            * data["RMAJM"][itime, rmajm_ax:]
            * 1e-2
        )

        psi = data["PLFLX"][itime, :]
        psi_n = psi / psi[-1]

        rbdry = Rsur[:, -1]
        zbdry = Zsur[:, -1]

        Rmin = min(rbdry)
        Rmax = max(rbdry)
        Zmin = min(zbdry)
        Zmax = max(zbdry)

        Rgrid = np.linspace(Rmin, Rmax, nr)
        Zgrid = np.linspace(Zmin, Zmax, nr)

        # Set up 1D profiles

        # Using interpolated splines
        q_interp = InterpolatedUnivariateSpline(psi_n, q)
        press_interp = InterpolatedUnivariateSpline(psi_n, press)
        f_interp = InterpolatedUnivariateSpline(psi_n_rmajm, f)
        f2_interp = InterpolatedUnivariateSpline(psi_n_rmajm, f ** 2)
        ffprime_interp = f2_interp.derivative()

        # Set up 2D profiles
        # Re-map from R(theta, psi), Z (theta, psi) to psi(R, Z)
        Rmesh, Zmesh = np.meshgrid(Rgrid, Zgrid)
        Rmesh_flat = np.ravel(Rmesh)
        Zmesh_flat = np.ravel(Zmesh)

        Rflat = np.ravel(Rsur.T)
        Zflat = np.ravel(Zsur.T)
        psiflat = np.repeat(psi, ntheta)

        RZflat = np.stack([Rflat, Zflat], -1)
        RZmesh_flat = np.stack([Rmesh_flat, Zmesh_flat], -1)

        # Interpolate using flat data
        psiRZ_interp = RBFInterpolator(RZflat, psiflat, kernel="cubic")

        # Map data to new grid and reshape
        psiRZ_data = np.reshape(psiRZ_interp(RZmesh_flat), np.shape(Rmesh)).T

        # Load in Eq object
        self.psi = psi

        # Set up 1D profiles as interpolated functions
        self.f_psi = f_interp

        self.ff_prime = ffprime_interp

        self.q = q_interp

        self.pressure = press_interp

        self.p_prime = self.pressure.derivative()

        # Set up 2D psi_RZ grid
        self.R = Rgrid

        self.Z = Zgrid

        self.psi_RZ = RectBivariateSpline(self.R, self.Z, psiRZ_data)

        rho = (np.max(Rsur, axis=0) - np.min(Rsur, axis=0)) / 2
        R_major = (np.max(Rsur, axis=0) + np.min(Rsur, axis=0)) / 2

        self.lcfs_R = rbdry
        self.lcfs_Z = zbdry

        self.a_minor = rho[-1]
        self.current = current
        self.psi_axis = psi_axis
        self.psi_bdry = psi_bdry

        rho = rho / rho[-1]

        R_major[0] = R_major[1] + psi_n[1] * (R_major[2] - R_major[1]) / (
            psi_n[2] - psi_n[1]
        )

        self.rho = InterpolatedUnivariateSpline(psi_n, rho)
        self.R_major = InterpolatedUnivariateSpline(psi_n, R_major)

    def get_flux_surface(self, psi_n):

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if psi_n > 1.0 or psi_n < 0.0:
            raise ValueError("You must have 0.0 <= psi_n <= 1.0")

        # Generate 2D mesh of normalised psi
        psi_2d = np.transpose(self.psi_RZ(self.R, self.Z))

        psin_2d = (psi_2d - self.psi_axis) / (self.psi_bdry - self.psi_axis)

        # Returns a list of list of contours for psi_n, resets backend to original value
        original_backend = mpl.get_backend()
        mpl.use("Agg")

        con = plt.contour(self.R, self.Z, psin_2d, levels=[0, psi_n])

        mpl.use(original_backend)

        paths = con.collections[1].get_paths()

        if psi_n == 1.0:
            if len(paths) == 0:
                raise ValueError(
                    "PsiN=1.0 for LCFS isn't well defined. Try lowering psi_n_lcfs"
                )
            elif len(paths) > 1:
                paths = [np.concatenate([path for path in paths])]

        # Find smallest path integral to find closed loop
        closest_path = np.argmin(
            [
                np.mean(
                    np.sqrt(
                        (path.vertices[:, 0] - self.R_axis) ** 2
                        + (path.vertices[:, 1] - self.Z_axis) ** 2
                    )
                )
                for path in paths
            ]
        )

        path = paths[closest_path]

        R_con, Z_con = path.vertices[:, 0], path.vertices[:, 1]

        # Start from OMP
        Z_con = np.flip(np.roll(Z_con, -np.argmax(R_con) - 1))
        R_con = np.flip(np.roll(R_con, -np.argmax(R_con) - 1))

        mpl.use(original_backend)
        return R_con, Z_con

    def generate_local(
        self,
        geometry_type=None,
    ):

        raise NotImplementedError
