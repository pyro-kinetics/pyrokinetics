import numpy as np
from scipy import constants

from .units import ureg as units

bk = constants.k
pi = constants.pi
mu0 = constants.mu_0
eps0 = constants.epsilon_0

electron_charge = constants.elementary_charge * units.elementary_charge

electron_mass = constants.electron_mass * units.kg
hydrogen_mass = constants.proton_mass * units.kg
deuterium_mass = constants.physical_constants["deuteron mass"][0] * units.kg
tritium_mass = constants.physical_constants["triton mass"][0] * units.kg

sqrt2 = np.sqrt(2)

three_smooth_numbers = np.array(
    [
        1,
        2,
        3,
        4,
        6,
        8,
        9,
        12,
        16,
        18,
        24,
        27,
        32,
        36,
        48,
        54,
        64,
        72,
        81,
        96,
        108,
        128,
        144,
        162,
        192,
        216,
        243,
        256,
        288,
        324,
        384,
        432,
        486,
        512,
        576,
        648,
        729,
        768,
        864,
        972,
        1024,
        1152,
        1296,
        1458,
        1536,
        1728,
        1944,
        2048,
        2187,
        2304,
        2592,
        2916,
        3072,
        3456,
        3888,
    ]
)
