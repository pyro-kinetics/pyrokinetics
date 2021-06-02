.. default-role:: math
.. _miller:

Miller Class
============

:Author: B Patel

Miller class inherits from LocalGeometry Class and is a dictionary of the major Miller parameters

The definition of the different keys are shown in :numref:`_tab-miller`.


.. _tab-miller:
.. table:: Miller

   +------------------+----------------------------------------------------------------------------------------------+
   |  Dictionary key  | Description                                                                                  |
   +==================+==============================================================================================+
   | psi_n            | Normalised poloidal flux :math:`\psi_n=\psi_{surface}/\psi_{LCFS}`                           |
   +------------------+----------------------------------------------------------------------------------------------+
   | rho              | Normalised minor radius :math:`\rho=r/a`                                                     |
   +------------------+----------------------------------------------------------------------------------------------+
   | Rmaj             | Normalised major radius :math:`R_{maj}/a`                                                    |
   +------------------+----------------------------------------------------------------------------------------------+
   | a_minor          | Minor radius (m) :math:`a`                                                                   |
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
   | s_delta          | Triangularity shear :math:`\frac{\rho}{\sqrt{1-\delta}}\frac{\partial\delta}{\partial\rho}`  |
   +------------------+----------------------------------------------------------------------------------------------+
   | q                | Safety factor :math:`q`                                                                      |
   +------------------+----------------------------------------------------------------------------------------------+
   | shat             | Magnetic shear :math:`\hat{s} = \frac{\rho}{q}\frac{\partial q}{\partial\rho}`               |
   +------------------+----------------------------------------------------------------------------------------------+
   | shift            | Shafranov shift :math:`\Delta = \frac{\partial R_{maj}}{\partial r}`                         |
   +------------------+----------------------------------------------------------------------------------------------+
   | beta_prime       | Pressure gradient :math:`\beta'=\frac{8\pi*10^{-7}}{B_0^2}*\frac{\partial p}{\partial\rho}`  |
   +------------------+----------------------------------------------------------------------------------------------+
   | psi_n            | Normalised poloidal flux :math:`\psi_n`                                                      |
   +------------------+----------------------------------------------------------------------------------------------+

These will automatically be calculated when loading in GK codes/Numerical equilibria


.. automodule:: pyrokinetics.miller
  :members:
  :undoc-members:
  :show-inheritance:
