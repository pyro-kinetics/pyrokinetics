from ..units import ureg as units

# Define basic units, COCOS 11
eq_units = {
    "len": units.meter,
    "psi": units.weber,
    "F": units.meter * units.tesla,
    "p": units.pascal,
    "q": units.dimensionless,
    "B": units.tesla,
    "I": units.ampere,
}

# Add derivatives
eq_units["F_prime"] = eq_units["F"] / eq_units["psi"]
eq_units["FF_prime"] = eq_units["F"] ** 2 / eq_units["psi"]
eq_units["p_prime"] = eq_units["p"] / eq_units["psi"]
eq_units["q_prime"] = eq_units["q"] / eq_units["psi"]
eq_units["len_prime"] = eq_units["len"] / eq_units["psi"]
