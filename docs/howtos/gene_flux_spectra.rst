.. _sec-gene-flux-spectra:

===============================================
 Load (kx, ky)-resolved GENE flux spectra
===============================================

GENE's ``nrg`` file records fluxes volume-integrated over the simulation
domain — dimensions ``(species, flux, field, time)``. For cross-code
comparison (e.g. against CGYRO's ``flux`` output) and for turbulence
analysis, it is often useful to have the fluxes resolved in
``(kx, ky)``. Pyrokinetics can compute these spectra on request from the
moments and fields that GENE writes to disk.

Quick start
===========

.. code-block:: python

   from pyrokinetics import Pyro
   from pyrokinetics.gk_code import GKOutputReaderGENE

   pyro = Pyro(gk_file="parameters_0001", gk_code="GENE")
   gk = GKOutputReaderGENE().read_from_file(
       "parameters_0001",
       norm=pyro.norms,
       load_flux_spectra=True,
   )

   print(gk.data.data_vars)          # includes particle_es, heat_es, ...
   print(gk.data["heat_es"].dims)    # ('species', 'kx', 'ky', 'flux_time')

What you get
============

Four new data variables are added, all with dims
``(species, kx, ky, flux_time)``:

=================  ============================================
``particle_es``    ExB particle flux spectrum (from :math:`\phi`)
``heat_es``        ExB heat flux spectrum (from :math:`\phi`)
``particle_em``    flutter particle flux spectrum (from :math:`A_\parallel`)
``heat_em``        flutter heat flux spectrum (from :math:`A_\parallel`)
=================  ============================================

A new ``flux_time`` coordinate is added alongside the existing ``time``.
Moments are typically written less often than the ``nrg`` file, so
``flux_time`` is sparser than ``time`` — always select on the explicit
axis:

.. code-block:: python

   gk.data["heat_es"].sel(flux_time=slice(t0, t1)).mean("flux_time")

What the underlying formula is
==============================

For each species :math:`s`, per ``(kx, ky)``:

.. math::

   \Gamma^\phi_s = \big\langle \hat n_s^*\, v_{Ex} \big\rangle_\theta \cdot n_s

   Q^\phi_s = \big\langle (\tfrac{1}{2}\hat T_\parallel + \hat T_\perp
       + \tfrac{3}{2}\hat n_s)^*\, v_{Ex} \big\rangle_\theta \cdot n_s T_s

   \Gamma^{A_\parallel}_s = \big\langle \hat u_{\parallel,s}^*\, B_x \big\rangle_\theta \cdot n_s

   Q^{A_\parallel}_s = \big\langle (\hat q_\parallel + \hat q_\perp)^*\,
       B_x \big\rangle_\theta \cdot n_s T_s

where :math:`v_{Ex} = -i k_y \hat\phi / B_\mathrm{ref}`,
:math:`B_x = +i k_y \hat A_\parallel / B_\mathrm{ref}`, and
:math:`\langle\cdot\rangle_\theta` is a flux-surface average using the
per-:math:`\theta` Jacobian from GENE's own geometry output file. This
matches GENE's built-in ``fluxspectra2D.pro`` diagnostic.

Requirements and limits
=======================

* The GENE output directory must contain ``mom_<species>_####`` files
  and a geometry file (e.g. ``miller_####``).
* Nonlinear runs only. On linear runs pyrokinetics raises
  ``NotImplementedError`` — linear QL fluxes are already single-mode and
  don't need a 2D spectrum.
* Only :math:`\phi` and :math:`A_\parallel` contributions are computed
  (electrostatic and flutter). Compressional (:math:`B_\parallel`) and
  momentum fluxes are not included in this release.
* Moments are read in full into memory; for large boxes use the
  ``downsample`` argument to ``read_from_file`` to thin the time axis.
