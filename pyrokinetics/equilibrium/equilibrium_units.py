from ..units import ureg as units

# Define basic units
eq_units = {
    "len": units.meter,
    "psi": units.weber / units.rad,
    "f": units.meter * units.tesla,
    "p": units.pascal,
    "q": units.dimensionless,
    "b": units.tesla,
}

# Add derivatives
eq_units["f_prime"] = eq_units["f"] / eq_units["psi"]
eq_units["ff_prime"] = eq_units["f"] ** 2 / eq_units["psi"]
eq_units["p_prime"] = eq_units["p"] / eq_units["psi"]
eq_units["q_prime"] = eq_units["q"] / eq_units["psi"]
eq_units["len_prime"] = eq_units["len"] / eq_units["psi"]
