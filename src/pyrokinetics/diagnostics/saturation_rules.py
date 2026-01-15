import warnings

import numpy as np
import xarray as xr

from ..pyroscan import PyroScan
from . import get_sat_params, get_zonal_mixing, sum_ky_spectrum


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
        if not hasattr(self.pyro_scan, "gk_output"):
            self.pyro_scan.load_gk_output(output_convention=output_convention)

        data = self.pyro_scan.gk_output
        pyro = self.pyro_scan.base_pyro

        # Units factor to account for training done in pyro units
        pyro_units = pyro.norms.pyrokinetics
        units = getattr(pyro.norms, output_convention)

        kys = data[ky_dim] * data[ky_dim].units

        if theta0_dim in data.dims:
            theta0s = data[theta0_dim]
        else:
            theta0s = [0.0]

        shat = pyro.local_geometry.shat

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
        k_perp = field_squared.data * 0.0 * kys.data.units

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

            k_perp_interp = np.interp(theta, theta_long, k_perp_long)
            for iky, ky in enumerate(kys.data):
                # Technically k_perp / ky
                k_perp[iky, itheta0, :, :] = k_perp_interp.m * ky

        bmag = pyro.metric_terms.B_magnitude

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
        b_units = bmag_balloon.units
        bmag = field_squared * 0.0 * b_units + bmag_balloon
        field_correction = xr.where(bmag.field == "bpar", bmag, 1.0 * b_units)
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
        ) / (units.vref / units.lref * units.rhoref)

        # Q_ql = Q_ql * Lambda
        heat_ql = heat * ql_metric_full
        particle_ql = particle * ql_metric_full

        # Find theta0_max
        max_gam = growth_rate.max(dim=theta0_dim)
        thmax = gamma_exb / (shat * max_gam)

        thmax = thmax.where(thmax.pint.dequantify() < np.pi, np.pi)

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
        heat = heat_ky.integrate(coord=ky_dim) * kys.data.units
        particle = particle_ky.integrate(coord=ky_dim) * kys.data.units
        ql_metric = ql_metric_ky.integrate(coord=ky_dim) * kys.data.units

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

    def tglf_saturation(
        self,
        sat_rule: int = 2,
        output_convention: str = "pyrokinetics",
        gamma_tolerance: float = 0.001,
        ky_dim: str = "ky",
        theta0_dim: str = "theta0",
        time_avg_range: float = 0.8,
        units: str = "GYRO",
        alpha_zf: float = 1.0,
        rlnp_cutoff: float = 18.0,
        alpha_quench: float = 0.0,
        use_ave_ion_grid: bool = False,
        vexb_shear: float = 0.0,
        alpha_e: float = 1.0,
        **tglf_params,
    ):
        """
        Apply TGLF saturation rules (SAT1, SAT2, SAT3) to PyroScan data.

        This method converts PyroScan gk_output data to TGLF-compatible format
        and applies the TGLF saturation rules to calculate transport fluxes.

        Parameters
        ----------
        sat_rule : int, default 2
            TGLF saturation rule to use (1=SAT1, 2=SAT2, 3=SAT3)
        output_convention : str, default "pyrokinetics"
            Output normalization convention
        gamma_tolerance : float, default 0.001
            Tolerance for growth rate to be included in calculation
        ky_dim : str, default "ky"
            Name of ky dimension in PyroScan object
        theta0_dim : str, default "theta0"
            Name of theta0 dimension in PyroScan object
        time_avg_range : float, default 0.8
            Fraction of simulation time to use for time averaging
        units : str, default "GYRO"
            Units convention for TGLF ("GYRO" or "CGYRO")
        alpha_zf : float, default 1.0
            Zonal flow coupling coefficient
        rlnp_cutoff : float, default 18.0
            Pressure gradient cutoff parameter
        alpha_quench : float, default 0.0
            ExB quench parameter
        use_ave_ion_grid : bool, default False
            Whether to use average ion grid
        vexb_shear : float, default 0.0
            ExB shear rate
        alpha_e : float, default 1.0
            Alpha E parameter
        **tglf_params
            Additional TGLF parameters

        Returns
        -------
        gk_output : xr.Dataset
            Dataset containing transport fluxes with TGLF saturation applied
        """

        def tglf_single(data, exb_shear):
            # Extract ky spectrum
            ky_spect = data[ky_dim].values
            nky = len(ky_spect)

            # Handle theta0 dimension
            if theta0_dim in data.dims:
                theta0s = data[theta0_dim]
                ntheta0 = len(theta0s)
            else:
                theta0s = xr.DataArray([0.0], dims=[theta0_dim])
                ntheta0 = 1

            if not hasattr(gamma_tolerance, "units"):
                growth_rate_units = data["growth_rate"].data.units
                warnings.warn(
                    f"Adding units [{growth_rate_units}] to gamma_tolerance as it has not been "
                    "specified. To suppress this warning please add units"
                )
                gamma_tol = gamma_tolerance * growth_rate_units
            else:
                gamma_tol = gamma_tolerance

            # Get growth rates and apply tolerance filtering
            if "growth_rate_tolerance" in data.data_vars:
                growth_rate_tolerance = data["growth_rate_tolerance"]
                growth_rate = data["growth_rate"].where(
                    growth_rate_tolerance < gamma_tol,
                    0.0,  # Need to give gamma_tolerance units here
                )
            else:
                growth_rate = data["growth_rate"]

            # Time average growth rates if time dimension exists
            if "time" in growth_rate.dims:
                growth_rate = growth_rate.mean(dim="time")
            # Get number of modes (assume single mode if not in dimensions)
            if "mode" in data.dims:
                nmodes = len(data["mode"])
                mode_dim = "mode"
            else:
                nmodes = 1
                # Add mode dimension
                growth_rate = growth_rate.expand_dims("mode")
                mode_dim = "mode"
            # Convert to numpy array [nky, nmodes]
            if theta0_dim in growth_rate.dims:
                # For ballooning runs, use maximum over theta0
                gammas = growth_rate.max(dim=theta0_dim)
            else:
                gammas = growth_rate

            # Explicitly reorder dimensions
            growth_rate = growth_rate.transpose("ky", "mode")

            gammas = growth_rate.values  # shape (ky, modes)

            # Ensure 2D array
            if gammas.ndim == 1:
                gammas = [gammas]

            # Extract flux data and convert to quasi-linear weights
            # Get particle, heat, and momentum fluxes
            particle_flux = data["ql_particle"]
            heat_flux = data["ql_heat"]

            # Handle momentum/stress fluxes
            if "momentum" in data.data_vars:
                momentum_flux = data["ql_momentum"]
            else:
                # Create zero momentum flux if not available
                momentum_flux = xr.zeros_like(heat_flux)

            # Time average fluxes if time dimension exists
            if "time" in particle_flux.dims:
                particle_flux = particle_flux.mean(dim="time")
                heat_flux = heat_flux.mean(dim="time")
                momentum_flux = momentum_flux.mean(dim="time")

            # Apply growth rate tolerance filtering
            if "growth_rate_tolerance" in data.data_vars:
                tolerance_filter = data["growth_rate_tolerance"] < gamma_tol
                if "time" in tolerance_filter.dims:
                    tolerance_filter = tolerance_filter.mean(dim="time") > 0.5

                particle_flux = particle_flux.where(tolerance_filter, 0.0)
                heat_flux = heat_flux.where(tolerance_filter, 0.0)
                momentum_flux = momentum_flux.where(tolerance_filter, 0.0)

            # Get dimensions
            nspecies = len(data["species"])
            nfield = len(data["field"]) if "field" in data.dims else 1

            # Convert fluxes to QL weights format [nky, nmodes, nspecies, nfield]
            # For now, assume single mode and sum over fields
            if "field" in particle_flux.dims:
                particle_QL = particle_flux.sum(dim="field")
                energy_QL = heat_flux.sum(dim="field")
                toroidal_stress_QL = momentum_flux.sum(dim="field")
            else:
                particle_QL = particle_flux
                energy_QL = heat_flux
                toroidal_stress_QL = momentum_flux

            # Add mode dimension if needed
            if mode_dim not in particle_QL.dims:
                particle_QL = particle_QL.expand_dims(mode_dim)
                energy_QL = energy_QL.expand_dims(mode_dim)
                toroidal_stress_QL = toroidal_stress_QL.expand_dims(mode_dim)

            # Handle theta0 dimension by taking max
            if theta0_dim in particle_QL.dims:
                particle_QL = particle_QL.max(dim=theta0_dim)
                energy_QL = energy_QL.max(dim=theta0_dim)
                toroidal_stress_QL = toroidal_stress_QL.max(dim=theta0_dim)

            # Transpose to [ky, mode, species] and add field dimension
            particle_QL = particle_QL.transpose(ky_dim, mode_dim, "species")
            energy_QL = energy_QL.transpose(ky_dim, mode_dim, "species")
            toroidal_stress_QL = toroidal_stress_QL.transpose(
                ky_dim, mode_dim, "species"
            )

            # Add field dimension and convert to numpy
            particle_QL = particle_QL.values[:, :, :, np.newaxis]
            energy_QL = energy_QL.values[:, :, :, np.newaxis]
            toroidal_stress_QL = toroidal_stress_QL.values[:, :, :, np.newaxis]

            # Create parallel stress and exchange arrays (set to zero for now)
            parallel_stress_QL = np.zeros_like(toroidal_stress_QL)
            exchange_QL = np.zeros_like(toroidal_stress_QL)

            # Extract geometry parameters from pyro object
            geom = pyro.local_geometry
            species = pyro.local_species

            # Convert geometry to TGLF input parameters
            tglf_inputs = {
                "SAT_RULE": sat_rule,
                "UNITS": units,
                "ALPHA_ZF": alpha_zf,
                "RLNP_CUTOFF": rlnp_cutoff,
                "ALPHA_QUENCH": alpha_quench,
                "USE_AVE_ION_GRID": use_ave_ion_grid,
                "VEXB_SHEAR": exb_shear,
                "ALPHA_E": alpha_e,
                # Geometry parameters
                "RMAJ_LOC": geom.Rmaj.m,
                "RMIN_LOC": geom.rho.m,
                "Q_LOC": geom.q.m,
                "Q_PRIME_LOC": geom.shat.m,
                "P_PRIME_LOC": 0.0,  # Will be calculated from species
                "KAPPA_LOC": geom.kappa.m,
                "S_KAPPA_LOC": geom.s_kappa.m,
                "DELTA_LOC": geom.delta.m,
                "S_DELTA_LOC": geom.s_delta.m,
                "ZETA_LOC": 0.0,
                "S_ZETA_LOC": 0.0,
                # "ZETA_LOC": geom.zeta.m,
                # "S_ZETA_LOC": geom.s_zeta.m,
                "DRMAJDX_LOC": 1.0,  # Default value
                "DRMINDX_LOC": 1.0,  # Default value
                "SIGN_IT": 1.0,
                # Species parameters (use primary ion as reference)
                "NS": nspecies,
            }

            # Split species while preserving ion order
            electron_specs = []
            ion_specs = []

            for spec_name in data["species"].values:
                spec = species[spec_name]
                if spec.z.m == -1:
                    electron_specs.append(spec)
                else:
                    ion_specs.append(spec)

            if len(electron_specs) != 1:
                raise ValueError("Expected exactly one electron species")

            # Electrons always index 1
            idx = 1
            spec = electron_specs[0]
            tglf_inputs[f"ZS_{idx}"] = spec.z.m
            tglf_inputs[f"MASS_{idx}"] = spec.mass.m
            tglf_inputs[f"RLNS_{idx}"] = spec.inverse_ln.m
            tglf_inputs[f"RLTS_{idx}"] = spec.inverse_lt.m
            tglf_inputs[f"AS_{idx}"] = spec.dens.m
            tglf_inputs[f"TAUS_{idx}"] = spec.temp.m

            # Ions start at index 2, order preserved
            for idx, spec in enumerate(ion_specs, start=2):
                tglf_inputs[f"ZS_{idx}"] = spec.z.m
                tglf_inputs[f"MASS_{idx}"] = spec.mass.m
                tglf_inputs[f"RLNS_{idx}"] = spec.inverse_ln.m
                tglf_inputs[f"RLTS_{idx}"] = spec.inverse_lt.m
                tglf_inputs[f"AS_{idx}"] = spec.dens.m
                tglf_inputs[f"TAUS_{idx}"] = spec.temp.m
            """
            # Add species-specific parameters
            for i, spec_name in enumerate(data["species"].values):
                spec = species[spec_name]
                idx = i + 1  # TGLF uses 1-based indexing

                tglf_inputs[f"ZS_{idx}"] = spec.z.m
                tglf_inputs[f"MASS_{idx}"] = spec.mass.m
                tglf_inputs[f"RLNS_{idx}"] = spec.inverse_ln.m
                tglf_inputs[f"RLTS_{idx}"] = spec.inverse_lt.m
                tglf_inputs[f"AS_{idx}"] = spec.dens.m
                tglf_inputs[f"TAUS_{idx}"] = spec.temp.m

            """
            # Get reference species parameters (typically species 2 - main ion)
            if nspecies >= 2:
                tglf_inputs["MASS_2"] = tglf_inputs["MASS_2"]
                tglf_inputs["TAUS_2"] = tglf_inputs["TAUS_2"]
                tglf_inputs["ZS_2"] = tglf_inputs["ZS_2"]
            else:
                # Use first species as reference
                tglf_inputs["MASS_2"] = tglf_inputs["MASS_1"]
                tglf_inputs["TAUS_2"] = tglf_inputs["TAUS_1"]
                tglf_inputs["ZS_2"] = tglf_inputs["ZS_1"]

            # Add any additional TGLF parameters
            tglf_inputs.update(tglf_params)

            # Calculate saturation parameters
            (
                kx0_e,
                SAT_geo1_out,
                SAT_geo2_out,
                R_unit,
                Bt0_out,
                B_geo0_out,
                grad_r0_out,
                theta_out,
                Bt_out,
                grad_r_out,
                B_unit_out,
            ) = get_sat_params(
                sat_rule, ky_spect, gammas.T, **tglf_inputs
            )  # Requires Gamma in the form modes KY

            # Add calculated parameters to inputs
            tglf_inputs.update(
                {
                    "SAT_geo1_out": SAT_geo1_out,
                    "SAT_geo2_out": SAT_geo2_out,
                    "SAT_geo0_out": 1.0,  # Default value
                    "Bt0_out": Bt0_out,
                    "B_geo0_out": B_geo0_out,
                    "grad_r0_out": grad_r0_out,
                }
            )

            # Calculate transport fluxes using TGLF saturation
            # Create dummy potential array
            potential = np.ones((nky, nmodes))
            ave_p0 = np.ones(nky)  # Dummy average pressure
            R_unit_array = np.ones((nky, nmodes)) * R_unit
            # Calculate fluxes
            results = sum_ky_spectrum(
                sat_rule,
                ky_spect,
                gammas,
                ave_p0,
                R_unit_array,
                kx0_e,
                potential,
                particle_QL,
                energy_QL,
                toroidal_stress_QL,
                parallel_stress_QL,
                exchange_QL,
                **tglf_inputs,
            )

            # Convert results to xarray Dataset
            species_coords = data["species"].values

            # Add particle flux
            particle_flux_sat = xr.DataArray(
                results["particle_flux_integral"][
                    0, :, 0
                ],  # [mode=0, species, field=0]
                dims=["species"],
                coords={"species": species_coords},
                name="particle",
            )

            # Add heat flux
            heat_flux_sat = xr.DataArray(
                results["energy_flux_integral"][0, :, 0],  # [mode=0, species, field=0]
                dims=["species"],
                coords={"species": species_coords},
                name="heat",
            )

            # Add momentum flux
            momentum_flux_sat = xr.DataArray(
                results["toroidal_stresses_integral"][
                    0, :, 0
                ],  # [mode=0, species, field=0]
                dims=["species"],
                coords={"species": species_coords},
                name="momentum",
            )
            return particle_flux_sat, heat_flux_sat, momentum_flux_sat

        # Load gk_output if not already loaded
        if not hasattr(self.pyro_scan, "gk_output"):
            self.pyro_scan.load_gk_output(
                output_convention=output_convention, tolerance_time_range=time_avg_range
            )

        data_full = (
            self.pyro_scan.gk_output.data
        )  # .data is to extract xarray from GK output object
        pyro = self.pyro_scan.base_pyro

        # to deal with pyroscan we need to iterate over none necessary dimensions
        # This will be done on a per data array method. and the cordinates to drop will be universal between
        # data array
        #
        # Need to extract:
        # Growth_rates
        # Growth_rate_tolerances
        # Particles
        # heat flux
        # Momentum_flux

        keep = ("ky", "theta0", "time", "mode", "field", "species")
        stack_dims = [d for d in data_full["ql_particle"].dims if d not in keep]

        gk_output = xr.Dataset()

        if len(stack_dims) != 0:
            particle_flux_list = []
            heat_flux_list = []
            momentum_flux_list = []
            data_stacked = data_full.stack(stacked=stack_dims)
            stacked_index = data_stacked.coords["stacked"]

            for i in range(data_stacked.sizes["stacked"]):
                data = data_stacked.isel(stacked=i).drop_vars(stack_dims)
                # Extracts new ExB shearing if it is a parameter that is updated
                if "gamma_exb" in data_stacked.coords:
                    exb_shear = data_stacked.isel(stacked=i).coords["gamma_exb"].item()
                else:
                    exb_shear = vexb_shear
                # ExB is being scanned

                particle_flux_sat, heat_flux_sat, momentum_flux_sat = tglf_single(
                    data, exb_shear
                )

                particle_flux_list.append(particle_flux_sat)
                heat_flux_list.append(heat_flux_sat)
                momentum_flux_list.append(momentum_flux_sat)

            # Create output dataset
            # Concatenate results along stacked dimension
            particle_flux_single = xr.concat(particle_flux_list, dim="stacked")
            heat_flux_single = xr.concat(heat_flux_list, dim="stacked")
            momentum_flux_single = xr.concat(momentum_flux_list, dim="stacked")

            particle_flux_stacked = particle_flux_single.assign_coords(
                stacked=stacked_index
            )
            heat_flux_stacked = heat_flux_single.assign_coords(stacked=stacked_index)
            momentum_flux_stacked = momentum_flux_single.assign_coords(
                stacked=stacked_index
            )

            gk_output["particle"] = particle_flux_stacked.unstack("stacked")
            gk_output["heat"] = heat_flux_stacked.unstack("stacked")
            gk_output["momentum"] = momentum_flux_stacked.unstack("stacked")

        else:
            data = data_full
            particle_flux_sat, heat_flux_sat, momentum_flux_sat = tglf_single(
                data, vexb_shear
            )

            gk_output["particle"] = particle_flux_sat
            gk_output["heat"] = heat_flux_sat
            gk_output["momentum"] = momentum_flux_sat

        # Add metadata
        gk_output.attrs["sat_rule"] = sat_rule
        gk_output.attrs["output_convention"] = output_convention
        return gk_output
