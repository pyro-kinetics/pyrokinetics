import copy
import warnings
from itertools import product

import numpy as np
import xarray as xr

from ..pyroscan import PyroScan, integrate_over_valid_range
from ..units import ureg


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
        ky_dim: str = "ky",
        theta0_dim: str = "theta0",
    ):
        """
        Please see doi:10.1017/S0022377824001107 for details on this quasi-linear model.

        Equivalent to ``mg_flux(mg_ingredients(...))``: those two steps are
        public so that the linear inputs of the rule can be inspected or
        exchanged between scans before they are combined.

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
        ingredients = self.mg_ingredients(
            output_convention=output_convention,
            gamma_tolerance=gamma_tolerance,
            ky_dim=ky_dim,
            theta0_dim=theta0_dim,
        )
        return self.mg_flux(
            ingredients,
            Q0=Q0,
            alpha=alpha,
            gamma_exb=gamma_exb,
            output_convention=output_convention,
            ky_dim=ky_dim,
            theta0_dim=theta0_dim,
        )

    def mg_ingredients(
        self,
        output_convention: str = "pyrokinetics",
        gamma_tolerance: float = 0.001,
        ky_dim: str = "ky",
        theta0_dim: str = "theta0",
    ):
        r"""
        The linear inputs of the MG rule, per ky (and theta0, field and every
        other dimension of the scan output):

        - ``growth_rate``: :math:`\gamma`, zero where the run is not converged
          (``growth_rate_tolerance`` above ``gamma_tolerance``, if the output
          has one) or stable (:math:`\gamma\le0`): a stable mode does not
          saturate at a finite amplitude, and a negative :math:`\Lambda` has no
          real power :math:`\Lambda^{\alpha-1}`.
        - ``field_amplitude``: :math:`\sqrt{\max|f|^2/\max|\phi|^2}` per field.
        - ``field_extent``: :math:`\langle|f|^2\rangle/\langle k_\perp^2|f|^2\rangle`
          per field, with the Jacobian and :math:`k_\perp` of each scan point's
          own geometry (a scan of ``beta_prime``, say, changes both).
        - ``heat_fraction``, ``particle_fraction``: each species' share of the
          linear flux, normalised by the total heat flux, as in the rule. Only if
          the output has fluxes; the total heat flux of the rule does not use
          them.

        Then :math:`\Lambda_k=\gamma\sum_f` ``field_amplitude * field_extent``.
        """
        if not hasattr(self.pyro_scan, "gk_output"):
            self.pyro_scan.load_gk_output(output_convention=output_convention)

        data = self.pyro_scan.gk_output
        data = getattr(data, "data", data)
        if "sample" in data.dims:
            raise NotImplementedError(
                "mg_ingredients needs a gridded scan; a hypercube has no ky dimension"
            )
        added_theta0 = theta0_dim not in data.dims
        if added_theta0:
            data = data.expand_dims({theta0_dim: [0.0]})

        # The rule needs |eigenfunction|**2: loaded with time_mode="average" it
        # is already stored, averaged in time; otherwise take the final time
        if "eigenfunctions_squared" in data:
            eigenfunctions_squared = data["eigenfunctions_squared"]
        else:
            eigenfunctions_squared = np.abs(data["eigenfunctions"]) ** 2
            if "time" in eigenfunctions_squared.dims:
                eigenfunctions_squared = eigenfunctions_squared.isel(time=-1, drop=True)
        # The scan keeps kx; the rule uses one kx
        if "kx" in eigenfunctions_squared.dims:
            kx_min = abs(eigenfunctions_squared["kx"]).argmin().item()
            eigenfunctions_squared = eigenfunctions_squared.isel(kx=kx_min, drop=True)

        if "growth_rate_tolerance" in data:
            converged = data["growth_rate_tolerance"] < gamma_tolerance
        else:
            converged = xr.full_like(
                data["growth_rate"].pint.dequantify(), True, dtype=bool
            )

        growth_rate = data["growth_rate"].where(converged, 0.0)
        growth_rate = growth_rate.where(growth_rate > 0.0, 0.0)
        field_squared = eigenfunctions_squared.where(converged, 0.0)

        geometry = self._geometry_profiles(data, ky_dim, theta0_dim)
        k_perp = geometry["k_perp_over_ky"] * data[ky_dim] * data[ky_dim].units
        jacobian = geometry["jacobian"]

        theta_min, theta_max = geometry.attrs["theta_range"]
        outside = (data["theta"] < theta_min) | (data["theta"] > theta_max)
        if bool(field_squared.where(outside).notnull().any()):
            warnings.warn(
                "Eigenfunctions extend beyond the theta range of the base Pyro's "
                f"metric terms [{theta_min:.3g}, {theta_max:.3g}], where the "
                "Jacobian is not known; theta integrals are restricted to that "
                "range.",
                stacklevel=2,
            )

        # Account for GS2 field normalisation used in training
        bmag = geometry["bmag"]
        field_correction = xr.where(
            field_squared.field == "bpar", bmag, 1.0 * bmag.pint.units
        )
        field_correction = xr.where(
            field_correction.field == "apar", field_correction * 0.5, field_correction
        )
        field_squared = field_squared * field_correction**-2

        # Field ratios
        field_amplitude = np.sqrt(
            field_squared.max(dim="theta")
            / field_squared.sel(field="phi").max(dim="theta")
        )

        # Each run is integrated over its own valid theta range: gaps from a
        # coarser grid are filled exactly, and nothing is extrapolated
        numerator = integrate_over_valid_range(field_squared * jacobian)
        denominator = integrate_over_valid_range(field_squared * jacobian * k_perp**2)

        ingredients = xr.Dataset(
            {
                "growth_rate": growth_rate,
                "field_amplitude": field_amplitude,
                "field_extent": numerator / denominator,
            }
        )

        if "heat" in data:
            heat_total = data["heat"].sum(dim=("field", "species"))
            for flux in ("heat", "particle"):
                ingredients[f"{flux}_fraction"] = (
                    data[flux].where(converged, 0.0).sum(dim="field") / heat_total
                )

        ingredients.attrs["added_theta0"] = int(added_theta0)
        return ingredients

    def _geometry_profiles(self, data, ky_dim, theta0_dim):
        r"""
        :math:`k_\perp/k_y` (per theta0), the Jacobian and :math:`|B|` on the
        output's ballooning-angle grid, for each combination of the scan's
        other parameters: those can change the geometry, so each gets its own
        metric terms. Points that differ only in ky or theta0 share them.
        """
        scan = self.pyro_scan
        other_dims = [
            dim
            for dim in getattr(scan, "parameter_dict", {})
            if dim not in (ky_dim, theta0_dim) and dim in data.dims
        ]
        profiles = []
        for values in product(*(scan.parameter_dict[dim] for dim in other_dims)):
            pyro = copy.deepcopy(scan.base_pyro)
            parameters = dict(zip(other_dims, values))
            scan._apply_parameters(parameters, pyro)
            profile = self._geometry_profile(pyro, data["theta"], data[theta0_dim])
            coords = {
                dim: [
                    data[dim]
                    .sel({dim: getattr(value, "m", value)}, method="nearest")
                    .item()
                ]
                for dim, value in parameters.items()
            }
            profiles.append(profile.expand_dims(coords))
        if not profiles:
            pyro = copy.deepcopy(scan.base_pyro)
            profiles = [self._geometry_profile(pyro, data["theta"], data[theta0_dim])]
        theta_range = profiles[0].attrs["theta_range"]
        combined = xr.combine_by_coords(profiles, combine_attrs="drop_conflicts")
        combined.attrs["theta_range"] = theta_range
        return combined.pint.quantify(unit_registry=ureg)

    @staticmethod
    def _geometry_profile(pyro, theta, theta0s):
        """Metric profiles of one geometry; see ``_geometry_profiles``."""
        theta = theta.data

        # Need to load MetricTerms
        pyro.load_metric_terms(ntheta=1024)
        theta_metric = pyro.metric_terms.regulartheta

        # Add one to cover additional theta grids (like in CGYRO)
        nperiod = pyro.numerics.nperiod + 1
        g_tt = pyro.metric_terms.field_aligned_covariant_metric("theta", "theta")

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

        k_perp_over_ky = []
        for theta0 in np.atleast_1d(theta0s.data):
            theta_long, k_perp_long = pyro.metric_terms.k_perp(
                ky=1.0, theta0=theta0, nperiod=nperiod
            )
            # NaN beyond the metric's theta range, rather than its end value
            k_perp_over_ky.append(
                np.interp(theta, theta_long, k_perp_long.m, left=np.nan, right=np.nan)
            )

        bmag = pyro.metric_terms.B_magnitude

        g_tt_long = np.tile(g_tt[:-1], 2 * nperiod - 1)
        bmag_long = np.tile(bmag[:-1], 2 * nperiod - 1)

        g_tt_long = np.append(g_tt_long, g_tt[-1])
        bmag_long = np.append(bmag_long, bmag[-1])
        jacobian_long = pyro.metric_terms.dpsidr * np.sqrt(g_tt_long) / bmag_long

        # Extend onto ballooning space, NaN beyond the metric's theta range: the
        # Jacobian is not known there, and holding its end value would be wrong
        jacobian = np.interp(
            theta,
            theta_long,
            getattr(jacobian_long, "magnitude", jacobian_long),
            left=np.nan,
            right=np.nan,
        ) * getattr(jacobian_long, "units", 1.0)
        bmag_balloon = (
            np.interp(theta, theta_geo_long, bmag_long.m, left=np.nan, right=np.nan)
            * bmag_long.units
        )

        theta_long = getattr(theta_long, "magnitude", theta_long)
        profile = xr.Dataset(
            {
                "k_perp_over_ky": (
                    (theta0s.name, "theta"),
                    np.array(k_perp_over_ky) * k_perp_long.units,
                ),
                "jacobian": ("theta", jacobian),
                "bmag": ("theta", bmag_balloon),
            },
            coords={"theta": theta, theta0s.name: np.atleast_1d(theta0s.data)},
        )
        # Units go to attrs, so the profiles of several geometries can be combined
        profile = profile.pint.dequantify()
        profile.attrs["theta_range"] = (np.min(theta_long), np.max(theta_long))
        return profile

    def mg_flux(
        self,
        ingredients: xr.Dataset,
        Q0: float = 25.0,
        alpha: float = 2.5,
        gamma_exb: float = 0.0,
        output_convention: str = "pyrokinetics",
        ky_dim: str = "ky",
        theta0_dim: str = "theta0",
    ):
        r"""
        The MG rule from its linear inputs (``mg_ingredients``, possibly with
        some of them exchanged with another scan's):

        .. math::
            \Lambda_k=\gamma\sum_f A_f E_f,\qquad
            Q = Q_0\,\Lambda^{\alpha},\qquad \Lambda = \int \Lambda_k\,dk_y

        per species :math:`Q_s = Q_0\Lambda^{\alpha-1}\int w_s\Lambda_k\,dk_y`
        when ``heat_fraction`` :math:`w_s` is given. Returns ``heat`` (and
        ``particle``) with the species dimension if the fractions are given,
        and the total heat flux otherwise, plus ``lambda`` (:math:`\Lambda_k`).
        """
        pyro = self.pyro_scan.base_pyro
        pyro_units = pyro.norms.pyrokinetics
        units = getattr(pyro.norms, output_convention)

        shat = pyro.local_geometry.shat
        theta0s = ingredients[theta0_dim]

        # Sum over fields
        ql_metric_full = (
            ingredients["growth_rate"]
            * ingredients["field_amplitude"]
            * ingredients["field_extent"]
        ).sum(dim="field") / (units.vref / units.lref * units.rhoref)
        # A stable or unconverged point has no eigenfunction averages (0/0):
        # it contributes nothing, whatever the other inputs are
        ql_metric_full = ql_metric_full.where(ingredients["growth_rate"] > 0.0, 0.0)

        # Q_ql = Q_ql * Lambda; without species fractions the total heat
        # flux is Q0 * Lambda**alpha
        weights = {
            flux: ingredients[f"{flux}_fraction"]
            for flux in ("heat", "particle")
            if f"{flux}_fraction" in ingredients
        } or {"heat": 1.0}
        # Where Lambda_k is zero the fractions may be NaN (no flux output there):
        # those points contribute nothing either way
        flux_ql = {
            flux: (weight * ql_metric_full).where(ql_metric_full != 0.0, 0.0)
            for flux, weight in weights.items()
        }

        # Find theta0_max
        # gamma_exb is in the units of the growth rate
        max_gam = ingredients["growth_rate"].max(dim=theta0_dim).pint.dequantify()
        thmax = gamma_exb / (shat * max_gam)

        thmax = thmax.where(thmax.pint.dequantify() < np.pi, np.pi)

        if len(theta0s) > 1:
            thmax = thmax.where(thmax > theta0s[1], theta0s[1])
            # Select relevant theta0 and integrate up to theta0_max
            flux_ky = {
                flux: (value.where(theta0s <= thmax, 0.0) / thmax).integrate(
                    coord=theta0_dim
                )
                for flux, value in flux_ql.items()
            }
            ql_metric_ky = (
                ql_metric_full.where(theta0s <= thmax, 0.0) / thmax
            ).integrate(coord=theta0_dim)
        else:
            # Select relevant theta0 only
            flux_ky = {
                flux: value.isel({theta0_dim: 0}) for flux, value in flux_ql.items()
            }
            ql_metric_ky = ql_metric_full.isel({theta0_dim: 0})

        # Integrate over ky
        ky_units = ingredients[ky_dim].units
        flux = {
            name: value.integrate(coord=ky_dim) * ky_units
            for name, value in flux_ky.items()
        }
        ql_metric = ql_metric_ky.integrate(coord=ky_dim) * ky_units

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

        gk_output = xr.Dataset()
        gk_output["heat"] = Q0 * flux["heat"] * ql_metric ** (alpha - 1) * units_factor

        if "particle" in flux:
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
            gk_output["particle"] = (
                G0 * flux["particle"] * ql_metric ** (alpha - 1) * units_factor
            )

        gk_output["lambda"] = ql_metric_full
        if ingredients.attrs.get("added_theta0"):
            gk_output = gk_output.squeeze(theta0_dim, drop=True)

        return gk_output
