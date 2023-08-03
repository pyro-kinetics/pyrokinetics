==========================
Normalisations conventions
==========================

Different gyrokinetic codes use different default `Normalisation` conventions and pyrokinetics allows for the conversion
between different conventions. Where possible variables in pyrokinetics have assigned units via the `Pint`_ python
library. Currently these objects all have assigned units.

 - `Equilibrium`
 - `Kinetics`
 - `Species`
 - `LocalSpecies`
 - `GKOutput`


===========
Using Units
===========

When loading in the `Equilibrium` and `Kinetics` objects variables will be in SI units ``Quantity``. Objects with units must be
treated consistently, meaning that it is not possible to add two ``Quantity`` objects with different ``Units``

.. code-block:: python

    from pyrokinetics import Pyro, template_dir

    # Equilibrium file
    eq_file = template_dir / "test.geqdsk"
    # Kinetics data file
    kinetics_file = template_dir / "jetto.cdf"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        kinetics_file=kinetics_file,
    )

    # Has units of  / meter ** -3
    density = pyro.kinetics.species_data.electron.get_dens(0.5)
    print(f"Density = {density}")


Running the above results in the following where `density` is a `Quantity` object

.. code-block:: console

    2.0617094330890322e+20 / meter ** 3

Using ``.m`` on a ``Quantity`` will return the magnitude of the ``Quantity`` without units. Using ``.to`` allows you to
convert to another pint ``Unit``

.. code-block:: python

    print(f"Electron density magnitude = {density.m}")
    print(f"Electron density (/cm^3) = {density.to('cm **-3')}")

.. code-block:: console

    Electron density magnitude = 2.0617094330890322e+20
    Electron density (/cm^3)) = 206170943308903.25 / centimeter ** 3

================
Reference values
================

In gyrokinetics we often work with normalised quantities with a reference value. For example densities are usually
defined relative to the electron density. In pyrokinetics these normalised quantities are given "units" to allow
conversion between different conventions.

So any local parameters will be defined in these reference units. When creating a pyro object from a
kinetics/equilibrium object then it is possible to map these values back in to SI units using ``to_base_units``
or to back to reference values

.. code-block:: python

    pyro.load_local(psi_n=0.5, local_geometry="Miller")

    electron_density = pyro.local_species.electron.dens

    print(f"Electron density (normalised) = {electron_density}")
    print(f"Electron density (un-normalised) = {electron_density.to_base_units()}")

    deuterium_density = pyro.local_species.deuterium.dens
    print(f"Deuterium density (normalised) = {deuterium_density}")
    print(f"Deuterium density (un-normalised) = {deuterium_density.to_base_units()}")


.. code-block:: console

    Electron density (normalised) = 1.0 nref_electron_test0000
    Electron density (un-normalised) = 2.0617094330890322e+20 / meter ** 3
    Deuterium density (normalised) = 0.5057294099957877 nref_electron_test0000
    Deuterium density (un-normalised) = 1.0426670951788664e+20 / meter ** 3


Each code has a different default normalisation and it is possible to map from one code to another by "converting" the
units. For example below we see that the collisionality has different units for different codes with different magnitude

.. code-block:: python

    electron_collisionality = pyro.local_species.electron.nu
    print(f"Electron collisionality (Pyro units) {electron_collisionality}")
    print(f"Electron collisionality (CGYRO units) {electron_collisionality.to(pyro.norms.cgyro)}")
    print(f"Electron collisionality (GS2 units) {electron_collisionality.to(pyro.norms.gs2)}")
    print(f"Electron collisionality (GENE units) {electron_collisionality.to(pyro.norms.gene)}")

.. code-block:: console

    Electron collisionality (Pyro units) 0.050877383651849475 vref_nrl_test0000 / lref_minor_radius_test0000
    Electron collisionality (CGYRO units) 0.050877383651849475 vref_nrl_test0000 / lref_minor_radius_test0000
    Electron collisionality (GS2 units) 0.03597574298925234 vref_most_probable_test0000 / lref_minor_radius_test0000
    Electron collisionality (GENE units) 0.09411557703006325 vref_nrl_test0000 / lref_major_radius_test0000

When loading a ``Pyro`` object directly from a gyrokinetic input file, the physical reference values are often not
stored. In this scenario it is only possible to convert quantities between different conventions but not back to SI
units.

The following reference values are defined in pyrokinetics under ``pyro.norms``, with each code convention being stored
within that i.e. CGYRO conventions/normalisations are under ``pyro.norms.cgyro``

.. list-table:: Pyrokinetic references
   :widths: 34 33 33
   :header-rows: 1

   * - Reference value
     - Location in Norms
     - Pyrokinetics convention default
   * - :math:`m_{ref}`: Reference mass
     - ``pyro.norms.mref``
     - Deuterium mass
   * - :math:`n_{ref}`: Reference density
     - ``pyro.norms.nref``
     - Electron density
   * - :math:`T_{ref}`: Reference temperature
     - ``pyro.norms.tref``
     - Electron temperature
   * - :math:`v_{ref}`: Reference velocity
     - ``pyro.norms.mref``
     - Sound speed :math:`c_s = \sqrt{T_e/m_D}`
   * - :math:`B_{ref}`: Reference magnetic field
     - ``pyro.norms.bref``
     - :math:`B_0 = f/ R_{maj}`
   * - :math:`L_{ref}`: Reference length
     - ``pyro.norms.lref``
     - Minor radius
   * - :math:`\rho_{ref}`: Reference Larmor radius
     - ``pyro.norms.rhoref``
     - :math:`c_s / \Omega_i` where :math:`\Omega_i = eB_0/m_D`


=======
Outputs
=======

Pyrokinetics stores everything in its own normalisation. When reading/write `GKInput` and `GKOutput`, the data is read
in and then converted into pyrokinetics normalisation. The output data is stored in the format of an xarray ``Dataset``
so to convert all of the output into a different convention do the following

.. code-block:: python

    from pyrokinetics import Pyro, template_dir

    # Point to CGYRO input file
    cgyro_template = template_dir / "outputs/CGYRO_linear/input.cgyro"

    # Load in file
    pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")

    # Load in CGYRO output data
    pyro.load_gk_output()

    # Data current in pyrokinetics units
    data = pyro.gk_output

    # This converts the data to CGYRO units
    data.to(pyro.norms.cgyro)



.. _Pint: https://pint.readthedocs.io/en/stable/
