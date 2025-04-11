import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from ..pyroscan import PyroScan


class SaturationRules:
    r"""
    Contains all the different saturation rules that can be applied

    Need a PyroScan object to apply the rule to
    """

    def __init__(self, pyro_scan: PyroScan):

        self.pyro_scan = pyro_scan

    def mg_saturation(
        self,
        Q0: float = 25.0,
        alpha: float = 2.5,
        gamma_exb: float = 0.0,
        output_convention: str = "pyrokinetics",
        gamma_tolerance: float = 0.001,
        equal_arc_theta: bool = True,
        ky_dim: str = "ky",
        theta0_dim: str = "theta0",
    ):
        """
        Please see doi:10.1017/S0022377824001107 for details on this quasi-linear model.

        Parameters
        ----------
        Q0: float
            Heat flux absolute magnitude from fitted model
        alpha: float
            Power to raise QL metric Lambda
        gamma_exb: float
            Rate of ExB shear. Note must be in units matching output_convention
        output_convention: str
            Choice of output convention
        gamma_tolerance: float
            Tolerance of growth rate to be included in calculation
        equal_arc_theta: bool
            Whether theta grid is equally spaced in arc length
        ky_dim: str
            Name of the dimension in PyroScan object corresponding to ky
        theta0_dim
            Name of the dimension in PyroScan object corresponding to theta0

        Returns
        -------
        gk_output: xr.Dataset
            Dataset containing heat and particle flux with dimensions of Species and
            other in original PyroScan object

        """
        if not hasattr(self.pyro_scan, "gk_output"):
            self.pyro_scan.load_gk_output(output_convention=output_convention)

        data = self.pyro_scan.gk_output
        pyro = self.pyro_scan.base_pyro

        kys = data[ky_dim]

        if theta0_dim in data.dims:
            theta0s = data[theta0_dim]
        else:
            theta0s = [0.0]

        shat = pyro.local_geometry.shat
        bunit_over_b0 = pyro.local_geometry.bunit_over_b0.m

        theta = data["theta"].data
        eigenfunctions = data["eigenfunctions"]
        growth_rate_tolerance = data["growth_rate_tolerance"]

        growth_rate = data["growth_rate"].where(
            growth_rate_tolerance < gamma_tolerance, 0.0
        )

        heat_tot = data["heat"].sum(dim=("field", "species"))
        heat = (
            data["heat"]
            .where(growth_rate_tolerance < gamma_tolerance, 0.0)
            .sum(dim="field")
            / heat_tot
        )
        particle = (
            data["particle"]
            .where(growth_rate_tolerance < gamma_tolerance, 0.0)
            .sum(dim="field")
            / heat_tot
        )

        field_squared = (
            np.abs(eigenfunctions.where(growth_rate_tolerance < gamma_tolerance, 0.0))
            ** 2
        )

        # Set up Jacobian and k_perp
        k_perp = field_squared.data * 0.0

        # Need to load MetricTerms
        pyro.load_metric_terms(ntheta=1024)
        theta_metric = pyro.metric_terms.regulartheta

        # Add one to cover additional theta grids (like in CGYRO)
        nperiod = pyro.numerics.nperiod + 1
        g_tt = pyro.metric_terms.field_aligned_covariant_metric("theta", "theta")

        # Can remove when we adjust outputs to handle equal_arc = True in GS2
        if equal_arc_theta:
            # Parallel gradient
            grho = np.sqrt(g_tt)

            # regulartheta foes from -pi to pi
            l_theta = cumulative_trapezoid(grho, theta_metric, initial=0.0)
            g_tt_eq = (l_theta[-1] / (2 * np.pi)) ** 2

            l_theta *= 1.0 / l_theta[-1] * 2 * np.pi
            l_theta += -np.pi
        else:
            l_theta = theta_metric

        # Geometric theta on grid on evenly spaced l(theta) grid
        even_l_theta = np.linspace(-np.pi, np.pi, len(l_theta))
        theta_geo = np.interp(even_l_theta, l_theta, theta_metric)

        m = np.linspace(-(nperiod - 1), nperiod - 1, 2 * nperiod - 1)
        ntheta = len(theta_geo) - 1
        m = np.repeat(m, ntheta)
        theta_geo_long = np.tile(theta_geo[:-1], 2 * nperiod - 1) + 2.0 * np.pi * m

        # Append final point
        theta_geo_final = 2 * np.pi * (m[-1]) + np.pi
        theta_geo_long = np.append(theta_geo_long, theta_geo_final)

        # L(theta) grid on regular theta grid long
        m = np.linspace(-(nperiod - 1), nperiod - 1, 2 * nperiod - 1)
        ntheta = len(l_theta) - 1
        m = np.repeat(m, ntheta)
        l_theta_long = np.tile(l_theta[:-1], 2 * nperiod - 1) + 2.0 * np.pi * m

        l_theta_final = l_theta[-1] + 2.0 * np.pi * m[-1]
        l_theta_long = np.append(l_theta_long, l_theta_final)

        for itheta0, theta0 in enumerate(theta0s.data):
            theta_long, k_perp_long = pyro.metric_terms.k_perp(
                ky=1.0, theta0=theta0, nperiod=nperiod
            )

            # Interp on to equal arc grid
            if equal_arc_theta:
                k_perp_interp = np.interp(theta, l_theta_long, k_perp_long)
            else:
                k_perp_interp = np.interp(theta, theta_long, k_perp_long)
            for iky, ky in enumerate(kys.data):
                # Technically k_perp / ky
                k_perp[iky, itheta0, :, :] = k_perp_interp.m * ky * bunit_over_b0

        bmag = pyro.metric_terms.B_magnitude

        if equal_arc_theta:
            bmag_eq_arc = np.interp(theta_geo, theta_metric, bmag)
            bmag_long = np.tile(bmag_eq_arc[:-1], 2 * nperiod - 1)
            bmag_long = np.append(bmag_long, bmag_eq_arc[-1])
            pyro_jacob = pyro.metric_terms.dpsidr * np.sqrt(g_tt_eq) / bmag_long
            bmag_balloon = np.interp(theta, l_theta_long, bmag_long)
        else:
            g_tt_long = np.tile(g_tt[:-1], 2 * nperiod - 1)
            bmag_long = np.tile(bmag[:-1], 2 * nperiod - 1)

            g_tt_long = np.append(g_tt_long, g_tt[-1])
            bmag_long = np.append(bmag_long, bmag[-1])
            pyro_jacob = pyro.metric_terms.dpsidr * np.sqrt(g_tt_long) / bmag_long
            bmag_balloon = np.interp(theta, theta_geo_long, bmag_long)

        jacobian_long = pyro_jacob

        # Extend onto ballooning space
        jacobian = np.interp(theta, theta_long, jacobian_long)[
            np.newaxis, np.newaxis, np.newaxis, :
        ]

        # Account for GS2 field normalisation used in training
        bmag = field_squared * 0.0 + bmag_balloon
        field_correction = xr.where(bmag.field == "bpar", bmag, 1.0)
        field_correction = xr.where(
            field_correction.field == "apar", field_correction * 0.5, field_correction
        )
        field_squared *= field_correction**-2

        # Field ratios
        field_factor = np.sqrt(
            field_squared.max(dim="theta")
            / field_squared.sel(field="phi").max(dim="theta")
        )

        # Numerator in Lambda
        numerator = (field_squared * jacobian).integrate(coord="theta")

        # Denominator
        denom = (field_squared * jacobian * k_perp**2).integrate(coord="theta")

        # Sum over fields
        ql_metric_full = (growth_rate * numerator * field_factor / denom).sum(
            dim="field"
        )

        # Q_ql = Q_ql * Lambda
        heat_ql = heat * ql_metric_full
        particle_ql = particle * ql_metric_full

        # Find theta0_max
        max_gam = growth_rate.max(dim=theta0_dim)
        thmax = gamma_exb / (shat * max_gam)

        thmax = thmax.where(thmax < np.pi, np.pi)
        if len(theta0s) > 1:
            thmax = thmax.where(thmax > theta0s[1], theta0s[1])
            # Select relevant theta0
            heat_theta0 = heat_ql.where(theta0s <= thmax, 0.0) / thmax
            particle_theta0 = particle_ql.where(theta0s <= thmax, 0.0) / thmax
            ql_metric_theta0 = ql_metric_full.where(theta0s <= thmax, 0.0) / thmax

            # Integrate up to theta0_max
            heat_ky = heat_theta0.integrate(coord=theta0_dim)
            particle_ky = particle_theta0.integrate(coord=theta0_dim)
            ql_metric_ky = ql_metric_theta0.integrate(coord=theta0_dim)

        else:
            # Select relevant theta0 only
            heat_ky = heat_ql.where(theta0s == theta0s.data[0], drop=True)
            particle_ky = particle_ql.where(theta0s == theta0s.data[0], drop=True)
            ql_metric_ky = ql_metric_full.where(theta0s == theta0s.data[0], drop=True)

        # Integrate over ky
        heat = heat_ky.integrate(coord=ky_dim)
        particle = particle_ky.integrate(coord=ky_dim)
        ql_metric = ql_metric_ky.integrate(coord=ky_dim)

        # Units factor to account for training done in pyro units
        pyro_units = pyro.norms.pyrokinetics
        units = getattr(pyro.norms, output_convention)

        if not hasattr(Q0, "units"):
            Q0 *= (
                units.nref * units.tref * units.vref * (units.rhoref / units.lref) ** 2
            )

        G0 = Q0 / units.tref

        Q_gb_pyro_units = (
            pyro_units.nref
            * pyro_units.tref
            * pyro_units.vref
            * (pyro_units.rhoref / pyro_units.lref) ** 2
        )

        units_factor = (
            (
                1
                * Q_gb_pyro_units
                / (pyro_units.vref * pyro_units.rhoref / pyro_units.lref) ** alpha
            )
            .to(units)
            .m
        )

        qflux = Q0 * heat * ql_metric ** (alpha - 1) * units_factor

        Gamma_gb_pyro_units = (
            pyro_units.nref
            * pyro_units.vref
            * (pyro_units.rhoref / pyro_units.lref) ** 2
        )

        units_factor = (
            (
                1
                * Gamma_gb_pyro_units
                / (pyro_units.vref * pyro_units.rhoref / pyro_units.lref) ** alpha
            )
            .to(units)
            .m
        )

        # Full flux calculation
        gflux = G0 * particle * ql_metric ** (alpha - 1) * units_factor

        gk_output = xr.Dataset()
        gk_output["heat"] = qflux
        gk_output["particle"] = gflux
        gk_output["lambda"] = ql_metric_full

        return gk_output
