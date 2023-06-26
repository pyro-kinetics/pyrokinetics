.. default-role:: math
.. _sec-miller:

LocalGeometryMiller
===================

:Author: B Patel

This is a subclass of the ``LocalGeometry`` class. This class is a describes the Miller representation of a local equilibrium as described in:

Miller, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
Physics of Plasmas 5.4 (1998): 973-978.

``LocalGeometryMiller`` inherits from ``LocalGeometry`` and is a CleverDict of the major Miller parameters.
It can be loaded directly from:
- The Miller parameters of an input file
- A 2D equilibrium

When loaded from a 2D equilibrium the flux surfaces are contours fits. Parameters like :math:`\kappa` and :math:`\delta` are calculated from the flux surface.
The gradients in these parameters are fitted by matching to the poloidal field


The definition of the different keys are shown in :numref:`tab-miller`.


.. _tab-miller:
.. table:: Miller Parameter definitions

   +------------------+----------------------------------------------------------------------------------------------+
   |  Dictionary key  | Description                                                                                  |
   +==================+==============================================================================================+
   | psi_n            | Normalised poloidal flux :math:`\psi_n=\psi_{surface}/\psi_{LCFS}`                           |
   +------------------+----------------------------------------------------------------------------------------------+
   | rho              | Normalised minor radius :math:`\rho=r/a`                                                     |
   +------------------+----------------------------------------------------------------------------------------------+
   | Rmaj             | Normalised major radius :math:`R_{maj}/a`                                                    |
   +------------------+----------------------------------------------------------------------------------------------+
   | a_minor          | Minor radius (m) :math:`a` (if 2D equilibrium exists)                                        |
   +------------------+----------------------------------------------------------------------------------------------+
   | B0               | Normalising field :math:`B_0 = f / R_{maj}`                                                  |
   +------------------+----------------------------------------------------------------------------------------------+
   | Bunit            | Effective field (GACODE) :math:`B_{unit} = \frac{q}{r}\frac{\partial\psi}{\partial r}`       |
   +------------------+----------------------------------------------------------------------------------------------+
   | kappa            | Elongation :math:`\kappa`                                                                    |
   +------------------+----------------------------------------------------------------------------------------------+
   | s_kappa          | Elongation shear :math:`\frac{\rho}{\kappa}\frac{\partial\kappa}{\partial\rho}`              |
   +------------------+----------------------------------------------------------------------------------------------+
   | delta            | Triangularity :math:`\delta`                                                                 |
   +------------------+----------------------------------------------------------------------------------------------+
   | s_delta          | Triangularity shear :math:`\frac{\rho}{\sqrt{1-\delta^2}}\frac{\partial\delta}{\partial\rho}`|
   +------------------+----------------------------------------------------------------------------------------------+
   | q                | Safety factor :math:`q`                                                                      |
   +------------------+----------------------------------------------------------------------------------------------+
   | shat             | Magnetic shear :math:`\hat{s} = \frac{\rho}{q}\frac{\partial q}{\partial\rho}`               |
   +------------------+----------------------------------------------------------------------------------------------+
   | shift            | Shafranov shift :math:`\Delta = \frac{\partial R_{maj}}{\partial r}`                         |
   +------------------+----------------------------------------------------------------------------------------------+
   | beta_prime       | Pressure gradient :math:`\beta'=\frac{8\pi*10^{-7}}{B_0^2}*\frac{\partial p}{\partial\rho}`  |
   +------------------+----------------------------------------------------------------------------------------------+

These will automatically be calculated when loading in GK codes/Numerical equilibria


.. automodule:: pyrokinetics.local_geometry.LocalGeometryMiller
  :members:
  :undoc-members:
  :show-inheritance:
