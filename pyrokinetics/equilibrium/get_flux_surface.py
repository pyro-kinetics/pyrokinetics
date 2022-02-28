import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_flux_surface(R, Z, psi_RZ, psi_axis, psi_bdry, R_axis, Z_axis, psi_n=1.0):

    if psi_n > 1.0 or psi_n < 0.0:
        raise ValueError("You must have 0.0 <= psi_n <= 1.0")

    # Generate 2D mesh of normalised psi
    psi_2d = np.transpose(psi_RZ(R, Z))

    psin_2d = (psi_2d - psi_axis) / (psi_bdry - psi_axis)

    # Returns a list of list of contours for psi_n, resets backend to original value
    original_backend = mpl.get_backend()
    mpl.use("Agg")

    con = plt.contour(R, Z, psin_2d, levels=[psi_n])

    mpl.use(original_backend)

    # Check if more than one contour has been found
    if len(con.collections) > 1:
        raise ValueError("More than one contour level found in get_flux_surface")

    paths = con.collections[0].get_paths()

    if psi_n == 1.0:
        if len(paths) == 0:
            raise ValueError(
                "PsiN=1.0 for LCFS isn't well defined. Try lowering psi_n_lcfs"
            )

    # Find smallest path integral to find closed loop
    closest_path = np.argmin(
        [
            np.mean(
                np.sqrt(
                    (path.vertices[:, 0] - R_axis) ** 2
                    + (path.vertices[:, 1] - Z_axis) ** 2
                )
            )
            for path in paths
        ]
    )

    path = paths[closest_path]

    R_con, Z_con = path.vertices[:, 0], path.vertices[:, 1]

    # Start from OMP
    Z_con = np.flip(np.roll(Z_con, -np.argmax(R_con) - 1))
    R_con = np.flip(np.roll(R_con, -np.argmax(R_con) - 1))

    mpl.use(original_backend)
    return R_con, Z_con
