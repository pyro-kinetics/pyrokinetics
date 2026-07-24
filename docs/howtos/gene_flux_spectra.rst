.. _sec-gene-flux-spectra:

===============================================
 Load (kx, ky)-resolved GENE flux spectra
===============================================

GENE's ``nrg`` file records fluxes volume-integrated over the simulation
domain, so pyrokinetics normally reads them with dimensions
``(field, species, time)``. For cross-code comparison (e.g. against
CGYRO, which resolves its fluxes in ``ky``) and for turbulence analysis,
it is often useful to have the fluxes resolved in ``(kx, ky)``.
Pyrokinetics can compute these on request from the moments and fields
that GENE writes to disk.

Quick start
===========

.. code-block:: python

   from pyrokinetics import Pyro

   pyro = Pyro(gk_file="parameters_0001", gk_code="GENE")
   pyro.load_gk_output(kxky_flux_spectra=True)

   heat = pyro.gk_output["heat"]
   print(heat.dims)   # ('field', 'species', 'kx', 'ky', 'time')

What you get
============

``kxky_flux_spectra=True`` changes where the ``particle`` and ``heat``
fluxes come from: instead of being read from ``nrg``, they are computed
from the moment and field files and carry ``kx`` and ``ky`` axes. They
are otherwise ordinary fluxes, with the same units and the same
``field`` dimension, so the electrostatic and electromagnetic parts are
selected in the usual way:

.. code-block:: python

   heat.sel(field="phi")    # ExB (electrostatic) heat flux spectrum
   heat.sel(field="apar")   # flutter (electromagnetic) heat flux spectrum

Summing over ``kx`` and ``ky`` recovers the volume-integrated values in
``nrg``:

.. code-block:: python

   heat.sel(field="phi").sum(dim=["kx", "ky"])

The spectra share the standard ``time`` coordinate with the fields, and
can be time-averaged over a saturated window in the usual way:

.. code-block:: python

   heat.sel(time=slice(t0, t1)).mean(dim="time")

What the underlying formula is
==============================

For each species :math:`s`, per ``(kx, ky)``:

.. math::

   \Gamma^\phi_s = \big\langle \hat n_s^*\, v_{Ex} \big\rangle_\theta \cdot n_s

.. math::

   Q^\phi_s = \big\langle (\tfrac{1}{2}\hat T_\parallel + \hat T_\perp
   + \tfrac{3}{2}\hat n_s)^*\, v_{Ex} \big\rangle_\theta \cdot n_s T_s

.. math::

   \Gamma^{A_\parallel}_s = \big\langle \hat u_{\parallel,s}^*\, B_x
   \big\rangle_\theta \cdot n_s

.. math::

   Q^{A_\parallel}_s = \big\langle (\hat q_\parallel + \hat q_\perp)^*\,
   B_x \big\rangle_\theta \cdot n_s T_s

where :math:`v_{Ex} = -i k_y \hat\phi / B_\mathrm{ref}`,
:math:`B_x = +i k_y \hat A_\parallel / B_\mathrm{ref}`, and
:math:`\langle\cdot\rangle_\theta` is a flux-surface average using the
per-:math:`\theta` Jacobian from GENE's own geometry output file. A
factor of 2 is applied to :math:`k_y > 0` modes to account for GENE
storing only :math:`k_y \ge 0` (hermitian symmetry). This matches
GENE's built-in ``fluxspectra2D.pro`` diagnostic.

Requirements and limits
=======================

* The GENE output directory must contain ``mom_<species>`` files and a
  geometry file, in either the binary (``mom_<species>.dat`` /
  ``mom_<species>_####``) or HDF5 (``mom_<species>.dat.h5``) format.
  Runs written with ``write_h5 = .t.`` expose both.
* The moments and the fields must be written at the same times, i.e.
  ``istep_mom`` and ``istep_field`` must match in the GENE input file.
* Nonlinear runs only. On linear runs pyrokinetics raises
  ``NotImplementedError`` — a linear run holds a single mode, so its
  fluxes need no ``(kx, ky)`` spectrum.
* Only the :math:`\phi` and :math:`A_\parallel` contributions are
  computed. As in ``nrg``, GENE lumps the compressional
  (:math:`B_\parallel`) contribution in with :math:`A_\parallel`, so the
  ``bpar`` entry of the field axis is left at zero.
* GENE writes no momentum moments, so no ``momentum`` flux is produced.
  Read the output again without ``kxky_flux_spectra`` to get the
  momentum flux from ``nrg``.
* Moments are read in full into memory; for large boxes use the
  ``downsample`` argument to thin the time axis.
