import numpy as np
import xarray as xr
import xrft

from ..pyro import Pyro


class ConvergenceTestLinear:
    r"""
    Contains convergence info

    Need a PyroScan object to apply the rule to
    """

    def __init__(self, pyro: Pyro, tolerance_range: float = 0.8):

        self.pyro = pyro
        self.tolerance_range = tolerance_range

        if not hasattr(self.pyro, "gk_output"):
            self.pyro.load_gk_output()

        self.gk_output = self.pyro.gk_output

        self.get_eigenvalue_tolerances()
        self.get_field_end_ratios()
        self.get_grid_scale()
        self.get_kperp_fields()
        print(self.kpar_fields)
        print(self.kperp_fields)

    def get_eigenvalue_tolerances(self):

        # Check accuracy of frequency, starting at time_range*final_time
        time = self.gk_output.data["time"]

        final_time = time.isel(time=-1)

        mode_frequency = self.gk_output.data["mode_frequency"] 
        mode_frequency = mode_frequency.where(
            mode_frequency["time"] > self.tolerance_range * final_time, drop=True
        )
        final_mode_frequency = mode_frequency.isel(time=-1)

        # Calculate growth rate tolerance
        growth_rate = self.gk_output.data["growth_rate"]
        growth_rate = growth_rate.where(
            growth_rate["time"] > self.tolerance_range * final_time, drop=True
        )
        final_growth_rate = growth_rate.isel(time=-1)

        growth_difference = ((growth_rate - final_growth_rate) / final_growth_rate) ** 2
        freq_difference = ((mode_frequency - final_mode_frequency) / final_mode_frequency) ** 2

        # Average over the end of the simulation, starting at time_range*final_time        
        growth_tolerance = np.sqrt(
            growth_difference.integrate("time")
            / (growth_difference["time"].isel(time=-1) - growth_difference["time"].isel(time=0))
        )
        freq_tolerance = np.sqrt(
            freq_difference.integrate("time")
            / (freq_difference["time"].isel(time=-1) - freq_difference["time"].isel(time=0))
        )

        self.growth_rate_tolerance = growth_tolerance
        self.mode_frequency_tolerance = freq_tolerance

    def get_field_end_ratios(self):

        # Check ratio of field ends and max field to ensure proper decay
        fields = self.gk_output.data["field"].data
        field_ratios = {}
        for field_name in fields:
            field = abs(self.gk_output.data[field_name].isel(kx=0, ky=0, time=-1))
            ratio = (field.isel(theta = 0) + field.isel(theta=-1)) / (2 * max(field))
            field_ratios[field_name] = ratio.data.m

        self.field_ratios = field_ratios

    def get_grid_scale(self):

        # Check amount of grid via second derivative
        fields = self.gk_output.data["field"].data
        grid_scale_ratios = {}
        for field_name in fields:
            field = self.gk_output.data[field_name].isel(kx=0, ky=0, time=-1).real
            turning_points = np.abs(np.sign(field.differentiate("theta")).diff(dim="theta")).sum(dim="theta") / 2
            turning_ratio = turning_points / len(field["theta"])
            grid_scale_ratios[field_name] = turning_ratio.data

        self.grid_scale = grid_scale_ratios

    def get_kperp_fields(self):

        # Check accuracy of frequency, starting at time_range*final_time
        fields = self.gk_output.data["field"].data 
        theta = self.gk_output.data["theta"].data
       
        jacobian = self.gk_output.data["jacobian"]

        numerics = self.pyro.numerics
        self.pyro.load_metric_terms(ntheta = numerics.ntheta* 4)
        metric = self.pyro.metric_terms
        theta_metric = metric.regulartheta

        g_tt = metric.field_aligned_covariant_metric("theta", "theta")
        b_dot_grad = 1 / np.sqrt(g_tt)
        theta_mod = np.mod(theta, 2 * np.pi)
        b_dot_grad = np.interp(theta_mod, theta_metric, b_dot_grad, period=2 * np.pi)

        theta_long, k_perp_long = self.pyro.metric_terms.k_perp(
            ky=1.0, theta0=numerics.theta0, nperiod=numerics.nperiod
            )
        k_perp = np.interp(theta, theta_long, k_perp_long)

        kperp_fields = {}
        kpar_fields = {}
        b_dot_grad_phi = b_dot_grad * self.gk_output.data["phi"].isel(kx=0, ky=0, time=-1).differentiate("theta")

        k_par = xrft.fft(b_dot_grad * self.gk_output.data["phi"].isel(kx=0, ky=0, time=-1))
        #k_par = b_dot_grad_phi / self.gk_output.data["phi"].isel(kx=0, ky=0, time=-1)
        
        import matplotlib.pyplot as plt
        plt.plot(theta, k_par)
        plt.show()
        for field_name in fields:
            field_sq = np.abs(self.gk_output.data[field_name].isel(kx=0, ky=0, time=-1))**2 * jacobian
            kperp_fields[field_name] = np.sqrt(((k_perp**2 * field_sq).integrate("theta") / field_sq.integrate("theta")).data.m)
            k_par = np.gradient(np.unwrap(np.angle(self.gk_output.data[field_name].isel(kx=0, ky=0, time=-1))), theta) * b_dot_grad
            #k_par = np.gradient(np.unwrap(np.angle(self.gk_output.data[field_name].isel(kx=0, ky=0, time=-1))), theta)
            kpar_fields[field_name] = np.sqrt(((k_par**2 * field_sq).integrate("theta") / field_sq.integrate("theta")).data.m)

            k_nyq = np.pi / (theta[1] - theta[0]) * b_dot_grad 
            knyq_fields = np.sqrt(((k_nyq**2 * field_sq).integrate("theta") / field_sq.integrate("theta")).data.m)

            print(np.diff(theta))
            print(knyq_fields)
            plt.plot(theta, k_par)
            plt.show()
        self.kperp_fields = kperp_fields
        self.kpar_fields = kpar_fields

