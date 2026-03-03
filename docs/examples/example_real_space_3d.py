from pyrokinetics import Pyro
import numpy as np
import xarray as xr
import xrft
import matplotlib.pyplot as plt
import pyvista as pv


# ============================================================
# Utilities
# ============================================================

def enforce_kx_symmetry(phi):
    """Remove duplicated kx=0 mode if present."""
    kx = phi.kx
    nkx = len(kx)
    argmin_kx = np.argmin(np.abs(kx.data))

    if (nkx % 2 == 0 and argmin_kx == nkx // 2) or \
       (nkx % 2 == 1 and argmin_kx == nkx // 2 + 1):
        phi = phi.isel(kx=slice(1, None))

    return phi


def close_theta_domain(phi, q):
    """Append periodic theta point at π."""
    theta0 = phi.theta[0].item()
    n = phi.ky / phi.ky.isel(ky=1).data

    first_slice = (
        phi.sel(theta=theta0)
        * np.exp(-2j * np.pi * q.m * n)
    )

    first_slice = first_slice.assign_coords(theta=np.pi)
    return xr.concat([phi, first_slice], dim="theta")


def ifft_to_real_space(phi):
    """IFFT in kx, ky → x, y."""
    rs_phi = xrft.ifft(
        phi,
        dim=["ky", "kx"],
        real_dim="ky",
        true_amplitude=True,
        true_phase=True,
    )

    rs_phi = rs_phi.assign_coords(
        freq_kx=rs_phi.freq_kx.data * (2 * np.pi),
        freq_ky=rs_phi.freq_ky.data * (2 * np.pi),
    ).rename({"freq_kx": "x", "freq_ky": "y"})

    return rs_phi


def close_y_domain(rs_phi):
    """Append periodic y boundary."""
    y0 = rs_phi.y[0].item()
    first_slice = rs_phi.sel(y=y0).assign_coords(y=-y0)
    return xr.concat([rs_phi, first_slice], dim="y")


def map_alpha_to_y(alpha, Ly):
    y_min = -Ly / 2
    y = alpha * Ly / (2 * np.pi)
    return ((y + y_min) % Ly) + y_min


def map_y_to_alpha(y, Ly):
    alpha = y / Ly * 2 * np.pi
    return ((alpha - np.pi) % (2 * np.pi)) - np.pi


def build_flux_surface_geometry(pyro, x, theta, rhostar):
    """Compute R(x,θ), Z(x,θ)."""
    rho0 = pyro.local_geometry.rho
    rho = rho0 + x * pyro.norms.pyrokinetics.rhoref * rhostar

    nx = len(x)
    ntheta = len(theta)

    R = np.empty((nx, ntheta)) * pyro.norms.pyrokinetics.lref
    Z = np.empty((nx, ntheta)) * pyro.norms.pyrokinetics.lref

    for i, rho_local in enumerate(rho):
        pyro.local_geometry.rho = rho_local
        R_local, Z_local = pyro.local_geometry.get_flux_surface(theta)
        R[i, :] = R_local
        Z[i, :] = Z_local

    pyro.local_geometry.rho = rho0
    return R, Z


def structured_grid_from_RZ(R, Z, zeta):
    """Create 3D cylindrical structured grid."""
    nzeta = zeta.shape[1]

    R3D = np.tile(R[:, :, None], nzeta)
    Z3D = np.tile(Z[:, :, None], nzeta)
    Phi = -zeta[None, :]

    X = (R3D * np.cos(Phi)).m
    Y = (R3D * np.sin(Phi)).m
    Z_cart = Z3D.m

    return pv.StructuredGrid(X, Y, Z_cart)


# ============================================================
# Load Simulation
# ============================================================

pyro = Pyro(gk_file="input.cgyro")
pyro.load_gk_output()
pyro.load_metric_terms(ntheta=128)

q = pyro.local_geometry.q
rhostar = 0.005 * pyro.norms.lref / pyro.norms.rhoref

phi = pyro.gk_output.data["phi"].pint.dequantify().isel(time=-1)
phi = enforce_kx_symmetry(phi)
phi = close_theta_domain(phi, q)

# ============================================================
# Fourier → Real space
# ============================================================

rs_phi = ifft_to_real_space(phi)
rs_phi = close_y_domain(rs_phi)

x = rs_phi.x.data
y = rs_phi.y.data
theta = rs_phi.theta.data

Ly = y.max() - y.min()

field = np.moveaxis(rs_phi.data, 0, 1)

# ============================================================
# Quick XY slice
# ============================================================

plt.figure()
plt.contourf(x, y, field[:, len(theta)//2, :].T, levels=50)
plt.xlabel(r"$x / \rho_s$")
plt.ylabel(r"$y / \rho_s$")
plt.title(r"$Re(\phi)(\theta=0)$")
plt.show()

# ============================================================
# Build Geometry
# ============================================================

R, Z = build_flux_surface_geometry(pyro, x, theta, rhostar)

# ============================================================
# 3D Flux Tube Plot
# ============================================================

alpha = map_y_to_alpha(y, Ly)

# Should this have q(r), alpha(r),
zeta = q * theta[:, None] - alpha[None, :]

plotter = pv.Plotter(window_size=(1000, 750))

grid = structured_grid_from_RZ(R, Z, zeta)
grid["field"] = field.flatten(order="F")

plotter.add_mesh(grid, scalars="field", cmap="plasma", show_scalar_bar=False)
plotter.add_axes()
plotter.show_bounds(location="outer")
plotter.camera.zoom(1.8)

plotter.save_graphic("flux_tube.pdf")
plotter.show()

# ============================================================
# R–Z slice at zeta = 0
# ============================================================

zeta0 = 0
Xg, THETA = np.meshgrid(x, theta, indexing="ij")
alpha = q * THETA - zeta0
Yg = map_alpha_to_y(alpha, Ly)

points = xr.Dataset(
    coords=dict(
        x=(("x", "theta"), Xg),
        y=(("x", "theta"), Yg),
        theta=(("x", "theta"), THETA),
    )
)

phi_slice = rs_phi.interp(
    x=points.x,
    y=points.y,
    theta=points.theta
)

plt.figure()
plt.contourf(R, Z, np.abs(phi_slice), levels=50)
plt.xlabel(r"$R$")
plt.ylabel(r"$Z$")
plt.title(r"$|\phi|(\zeta = 0)$")
plt.show()
