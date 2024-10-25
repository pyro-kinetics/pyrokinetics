import numpy as np
from scipy.optimize import least_squares


def get_miller_theta(R, Z, theta, R0, Z0, rho, kappa, delta):
    r"""Convert from regular theta to theta in the Miller scheme.

    :math:`\theta` has a different definition than expected in the Miller
    scheme, as can be seen by rearranging the equation for Z:

    .. math:

        \theta = \arcsin((Z - Z0) / (r\kappa))

    When comparing values along the flux surface, we must be careful to use the
    the native :math:`\theta` for whichever local geometry scheme and the
    corresponding Miller :math:`\theta` for the Miller reference values.
    """
    solution = np.hstack((R, Z))

    def f(x):
        R = R0 + rho * np.cos(x + np.arcsin(delta) * np.sin(x))
        Z = Z0 + rho * kappa * np.sin(x)
        return np.hstack((R, Z)) - solution

    return least_squares(f, theta).x
