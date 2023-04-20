from scipy import constants
import numpy as np
from .units import ureg as units

bk = constants.k
pi = constants.pi
mu0 = constants.mu_0
eps0 = constants.epsilon_0

electron_charge = constants.elementary_charge * units.elementary_charge

electron_mass = constants.electron_mass * units.kg
hydrogen_mass = constants.proton_mass * units.kg
deuterium_mass = 3.3435837724e-27 * units.kg

sqrt2 = np.sqrt(2)
