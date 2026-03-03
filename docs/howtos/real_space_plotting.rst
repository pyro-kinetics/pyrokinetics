====================================================
Real-space reconstruction and 3D flux-tube plotting
====================================================

This guide demonstrates how to reconstruct nonlinear gyrokinetic field data
from Fourier space :math:`(k_x, k_y, \theta)` into real space
:math:`(x, y, \theta)` and subsequently map this data onto a 3D toroidal
geometry :math:`(R, Z, \phi)` and thus cartesian :math:`(X, Y, Z)` using
``pyrokinetics``.

The workflow consists of four main steps:

1. Load nonlinear field data
2. Perform inverse Fourier transforms to real space
3. Construct flux-surface geometry
4. Visualise the field in 3D using PyVista

The same procedure applies to any supported gyrokinetic code.

Note this code does not yet account for radial variation of :math:`\alpha`,
:math:`q`, which requires more careful mapping between :math:`(x, y, \theta)`
to :math:`(X, Y, Z)`.

See the full example in `example_real_space_3d.py`


----------------------------------------
Loading nonlinear field data
----------------------------------------

First, import the required modules and load the nonlinear simulation:

.. code-block:: python

    >>> from pyrokinetics import Pyro, template_dir
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrft
    >>> import matplotlib.pyplot as plt
    >>> import pyvista as pv
    >>>
    >>> gk_file = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"
    >>> pyro = Pyro(gk_file=gk_file)
    >>> pyro.load_gk_output(load_fields=True)

We extract the electrostatic potential ``phi`` from the dataset:

.. code-block:: python

    >>> data = pyro.gk_output.data
    >>> phi = data["phi"].isel(time=-1).pint.dequantify()

The ``.pint.dequantify()`` call removes physical units, which is necessary
before applying Fourier transforms or trigonometric functions.

----------------------------------------
Ensuring periodicity in θ
----------------------------------------

Gyrokinetic simulations are field-aligned and periodic in the parallel
coordinate. To ensure a continuous domain in :math:`\theta`, we append the
periodic endpoint:

.. code-block:: python

    >>> q = pyro.local_geometry.q
    >>> theta0 = phi.theta[0].item()
    >>> n = phi.ky / phi.ky.isel(ky=1).data
    >>> first_slice = phi.sel(theta=theta0) * np.exp(-2j * np.pi * q.m * n)
    >>> first_slice = first_slice.assign_coords(theta=np.pi)
    >>> phi = xr.concat([phi, first_slice], dim="theta")

----------------------------------------
Inverse Fourier transform to (x, y)
----------------------------------------

Nonlinear simulations evolve fields in spectral space. To reconstruct the
real-space structure of the turbulence, we perform inverse Fourier transforms
in both perpendicular directions:

.. code-block:: python

    >>> rs_phi = xrft.ifft(
    ...     phi,
    ...     dim=["ky", "kx"],
    ...     real_dim="ky",
    ...     true_amplitude=True,
    ...     true_phase=True,
    ... )

The resulting coordinates are automatically renamed:

.. code-block:: python

    >>> rs_phi = rs_phi.assign_coords(
    ...     freq_kx=rs_phi.freq_kx.data * (2*np.pi),
    ...     freq_ky=rs_phi.freq_ky.data * (2*np.pi),
    ... ).rename({"freq_kx": "x", "freq_ky": "y"})

The dataset now has dimensions:

.. code-block:: python

    >>> print(rs_phi)
    <xarray.DataArray 'phi' (x, y, theta)>

You may now visualise slices in the perpendicular plane:

.. code-block:: python

    >>> field = rs_phi.data
    >>> plt.contourf(rs_phi.x, rs_phi.y, field[:, :, len(rs_phi.theta)//2].T)
    >>> plt.xlabel("y")
    >>> plt.ylabel("x")
    >>> plt.title("Re(phi) at theta = 0")
    >>> plt.show()

----------------------------------------
Constructing flux-surface geometry
----------------------------------------

To visualise turbulence in toroidal geometry, we map
:math:`(x, \theta)` to :math:`(R, Z)` using the local equilibrium model.

The radial coordinate is shifted using:

.. math::

   \rho = \rho_0 + x \, \rho_\mathrm{ref} \, \rho_*

We then evaluate the flux surface:

.. code-block:: python

    >>> x = rs_phi.x.data
    >>> theta = rs_phi.theta.data
    >>> rho0 = pyro.local_geometry.rho
    >>> rhostar = 0.005 * pyro.norms.lref / pyro.norms.rhoref
    >>> rho = rho0 + x * pyro.norms.pyrokinetics.rhoref * rhostar
    >>>
    >>> R = np.empty((len(x), len(theta)))
    >>> Z = np.empty((len(x), len(theta)))
    >>>
    >>> for i, rho_local in enumerate(rho):
    ...     pyro.local_geometry.rho = rho_local
    ...     R[i, :], Z[i, :] = pyro.local_geometry.get_flux_surface(theta)
    >>>
    >>> pyro.local_geometry.rho = rho0

The arrays ``R`` and ``Z`` now describe the poloidal cross-section.

Note this does not yet account for radial variation in :math:`R` and :math:`Z`

----------------------------------------
Mapping between y, α and ζ
----------------------------------------

In field-aligned gyrokinetic simulations, the perpendicular coordinate
:math:`y` is not a simple Cartesian direction. Instead, it is proportional
to the field-line label :math:`\alpha`, which is defined by

.. math::

   \alpha = q \theta - \zeta,

where:

* :math:`q` is the safety factor,
* :math:`\theta` is the poloidal angle,
* :math:`\zeta` is the toroidal angle.

Careful treatment of this mapping is essential when reconstructing real-space
fields and embedding them into toroidal geometry.

----------------------------------------
Mapping y → α
----------------------------------------

After performing the inverse Fourier transform, the field is expressed in
coordinates :math:`(x, y, \theta)`. The coordinate :math:`y` is periodic and
represents the binormal direction in the flux tube.

Because the gyrokinetic system uses periodic boundary conditions in :math:`y`,
we must map :math:`y` back to the field-line label :math:`\alpha` in a way that
preserves periodicity:

.. math::

   \alpha = \frac{2\pi}{L_y} y,

followed by wrapping into the principal domain :math:`[-\pi, \pi]`:

.. code-block:: python

    >>> def map_y_to_alpha(y, Ly):
    ...     alpha = y / Ly * 2 * np.pi
    ...     return ((alpha - np.pi) % (2*np.pi)) - np.pi

Here ``Ly`` is the total domain length in the binormal direction:

.. code-block:: python

    >>> Ly = rs_phi.y.max() - rs_phi.y.min()
    >>> alpha = map_y_to_alpha(rs_phi.y, Ly)

This step is crucial. Without enforcing periodic wrapping, the reconstructed
field would exhibit artificial discontinuities when mapped into toroidal
geometry.

----------------------------------------
Relation between α and ζ
----------------------------------------

The field-line label :math:`\alpha` connects directly to the toroidal angle
through the definition:

.. math::

   \zeta = q\theta - \alpha.

This equation describes how a field-aligned coordinate system embeds into
cylindrical toroidal coordinates.

For a fixed toroidal slice :math:`\zeta = \zeta_0`, the corresponding
binormal coordinate must satisfy:

.. math::

   \alpha = q\theta - \zeta_0.

This relation is used when interpolating onto a prescribed toroidal plane:

.. code-block:: python

    >>> zeta0 = 0.0
    >>> Xg, THETA = np.meshgrid(rs_phi.x, rs_phi.theta, indexing="ij")
    >>> alpha = pyro.local_geometry.q * THETA - zeta0
    >>> Yg = (alpha / (2*np.pi)) * Ly

We may then interpolate the real-space field:

.. code-block:: python

    >>> points = xr.Dataset(
    ...     coords=dict(
    ...         x=(("x", "theta"), Xg),
    ...         y=(("x", "theta"), Yg),
    ...         theta=(("x", "theta"), THETA),
    ...     )
    ... )
    >>> phi_slice = rs_phi.interp(
    ...     x=points.x,
    ...     y=points.y,
    ...     theta=points.theta,
    ... )


Note this code does not yet account for radial variation of :math:`\alpha`,
:math:`q`, which requires more careful mapping between :math:`(x, y, \theta)`
to :math:`(X, Y, Z)`.


----------------------------------------
Physical interpretation
----------------------------------------

The relationship

.. math::

   \alpha = q\theta - \zeta

ensures that the reconstructed field follows magnetic field lines in the
toroidal geometry. Moving along :math:`\theta` while holding :math:`\alpha`
constant corresponds to moving along a magnetic field line.

When extruding a flux surface in toroidal angle, we instead treat
:math:`\zeta` as an independent coordinate and construct

.. math::

   (X, Y, Z) =
   \left(
       R \cos \zeta,
       R \sin \zeta,
       Z
   \right).

The careful distinction between:

* :math:`y` — periodic flux-tube coordinate
* :math:`\alpha` — field-line label
* :math:`\zeta` — toroidal angle

is essential for producing geometrically consistent 3D visualisations.

Incorrect handling of this mapping will result in twisted or discontinuous
flux surfaces.


----------------------------------------
Building a 3D toroidal grid
----------------------------------------

To create a toroidal volume, we extrude the flux surface in toroidal angle using
:math:`\zeta = q\theta + \alpha`:


.. code-block:: python

    >>> zeta = np.linspace(0, 1.5*np.pi, 64)
    >>> nzeta = len(zeta)
    >>>
    >>> R3D = np.repeat(R[:, :, None], nzeta, axis=2)
    >>> Z3D = np.repeat(Z[:, :, None], nzeta, axis=2)
    >>> Phi = -zeta[None, None, :]
    >>>
    >>> X = R3D * np.cos(Phi)
    >>> Y = R3D * np.sin(Phi)
    >>> Z_cart = Z3D

We can now construct a ``pyvista.StructuredGrid``:

.. code-block:: python

    >>> grid = pv.StructuredGrid(X, Y, Z_cart)
    >>> grid["phi"] = field.flatten(order="F")

----------------------------------------
3D visualisation
----------------------------------------

Finally, visualise the turbulence structure:

.. code-block:: python

    >>> plotter = pv.Plotter()
    >>> plotter.add_mesh(grid, scalars="phi", cmap="viridis")
    >>> plotter.add_axes()
    >>> plotter.show_bounds(location="outer")
    >>> plotter.show()

This produces a 3D rendering of the turbulent flux tube in toroidal geometry.

----------------------------------------
Interpolating onto prescribed (R, Z, φ)
----------------------------------------

The real-space field may also be interpolated onto arbitrary
:math:`(x, y, \theta)` coordinates using xarray:

.. code-block:: python

    >>> Xg, THETA = np.meshgrid(x, theta, indexing="ij")
    >>> alpha = pyro.local_geometry.q * THETA
    >>> Yg = alpha / (2*np.pi) * (rs_phi.y.max() - rs_phi.y.min())
    >>>
    >>> points = xr.Dataset(
    ...     coords=dict(
    ...         x=(("x", "theta"), Xg),
    ...         y=(("x", "theta"), Yg),
    ...         theta=(("x", "theta"), THETA),
    ...     )
    ... )
    >>>
    >>> phi_slice = rs_phi.interp(
    ...     x=points.x,
    ...     y=points.y,
    ...     theta=points.theta
    ... )

This allows full flexibility in constructing arbitrary toroidal slices.

----------------------------------------
Summary
----------------------------------------

Using ``pyrokinetics`` you can:

* Load nonlinear spectral field data
* Perform inverse Fourier transforms to real space
* Map field-aligned coordinates to cylindrical geometry
* Visualise turbulence in 3D toroidal geometry
* Interpolate onto arbitrary spatial grids

This workflow enables detailed geometric visualisation of nonlinear
gyrokinetic turbulence directly from simulation output.