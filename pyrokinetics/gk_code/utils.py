from ..units import ureg as units

# Define basic units
gk_output_units = {
    "ky": units.rhoref ** -1,
    "kx": units.rhoref ** -1,
    "theta": units.radians,
    "time": units.lref / units.vref,
    "energy": units.dimensionless,
    "pitch": units.dimensionless,
    "moment": units.dimensionless,
    "field": units.dimensionless,
    "species": units.dimensionless,
    "phi": units.qref / units.tref * units.lref / units.rhoref,
    "apar": units.lref / units.rhoref ** 2 / units.bref,
    "bpar": units.lref / units.rhoref / units.bref,
    "particle": units.nref * units.vref * (units.rhoref / units.lref) ** 2,
    "momentum": units.nref * units.lref * units.tref * (units.rhoref / units.lref) ** 2,
    "energy": units.nref * units.vref * units.tref * (units.rhoref / units.lref) ** 2,
    "growth_rate": units.vref / units.lref,
    "eigenvalues": units.vref / units.lref,
    "mode_frequency": units.vref / units.lref,
}

