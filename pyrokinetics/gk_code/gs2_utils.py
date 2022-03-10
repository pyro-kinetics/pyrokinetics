"""
Defines mappings between gs2 input files and pyrokinetics variables
"""

import f90nml

pyro_gs2_miller = {
    "rho": ["theta_grid_parameters", "rhoc"],
    "Rmaj": ["theta_grid_parameters", "rmaj"],
    "q": ["theta_grid_parameters", "qinp"],
    "kappa": ["theta_grid_parameters", "akappa"],
    "shat": ["theta_grid_eik_knobs", "s_hat_input"],
    "shift": ["theta_grid_parameters", "shift"],
    "beta_prime": ["theta_grid_eik_knobs", "beta_prime_input"],
}

pyro_gs2_species = {
    "mass": "mass",
    "z": "z",
    "dens": "dens",
    "temp": "temp",
    "nu": "vnewk",
    "a_lt": "tprim",
    "a_ln": "fprim",
    "a_lv": "uprim",
}


def gs2_is_nonlinear(nml: f90nml.Namelist):
    try:
        is_box = nml["kt_grids_knobs"]["grid_option"] == "box"
        is_nonlinear = nml["nonlinear_terms_knobs"]["nonlinear_mode"] == "on"
        return is_box and is_nonlinear
    except KeyError:
        return False
