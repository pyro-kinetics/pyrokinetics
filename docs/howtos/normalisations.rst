.. _sec-normalisation-docs:

==========================
Normalisation conventions
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
     - ``pyro.norms.vref``
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



==========================
Code-specific conventions
==========================

While pyrokinetics uses a single internal normalisation, different gyrokinetic
codes adopt different definitions for their reference quantities. These
differences are encoded in *normalisation conventions*, and are used when
converting quantities to or from a given code’s units.

Each convention is defined by specifying which reference quantities differ from
the pyrokinetics defaults. Any reference not explicitly overridden uses the
pyrokinetics convention.

The available conventions are accessible via ``pyro.norms``:

.. code-block:: python

    pyro.norms.pyrokinetics
    pyro.norms.cgyro
    pyro.norms.gs2
    pyro.norms.gene
    pyro.norms.stella
    pyro.norms.gx
    pyro.norms.gkw
    pyro.norms.tglf
    pyro.norms.neo
    pyro.norms.imas

----------------------------
Pyrokinetics default choice
----------------------------

By default, pyrokinetics uses:

- Electron reference density :math:`n_{ref} = n_e`
- Electron reference temperature :math:`T_{ref} = T_e`
- Reference mass :math:`m_{ref} = m_D`
- Reference velocity :math:`v_{ref} = c_s = \sqrt{T_e / m_D}`
- Reference length :math:`L_{ref}` = minor radius
- Reference magnetic field :math:`B_{ref} = B_0`
- Reference Larmor radius :math:`\rho_{ref} = c_s / \Omega_i`

Other conventions override a subset of these choices.

-----------------------------
Summary of code conventions
-----------------------------


Some codes (e.g. GS2, STELLA, GKW, IMAS) use the *most probable thermal
velocity* as the reference velocity. In pyrokinetics this is defined as

.. math::

    v_{\mathrm{mp}} = \sqrt{\frac{2 T_{ref}}{m_{ref}}}

This differs from the sound-speed–based convention

.. math::

    c_s = \sqrt{\frac{T_{ref}}{m_{ref}}}

by a factor of :math:`\sqrt{2}`. The distinction is purely a normalisation
choice; conversion between conventions is always possible within
pyrokinetics and does not require additional physical reference values.

For all supported gyrokinetic codes, the reference Larmor radius is defined as

.. math::

    \rho_{ref} = \frac{v_{ref}}{q_{ref} B_{ref} / m_{ref}}

As a result, the defining choices for a normalisation convention are the
reference velocity :math:`v_{ref}`, reference length :math:`L_{ref}`, and
reference magnetic field :math:`B_{ref}`. Together, these determine the spatial,
velocity, and gyroradius scaling of the system.

The remaining reference quantities (such as :math:`n_{ref}` and
:math:`T_{ref}`) may vary between codes, but do not affect the fundamental
normalisation of lengths and velocities.

Some codes use a magnetic field normalisation based on the equilibrium flux
gradient rather than the on-axis field. In these cases,

.. math::

    B_{ref} = B_{\mathrm{unit}} = \frac{q}{r}\,\frac{d\psi}{dr}

The table below summarises the reference choices used by each supported code.

.. list-table:: Normalisation conventions by code
   :widths: 18 27 27 28
   :header-rows: 1

   * - Code
     - :math:`v_{ref}`
     - :math:`L_{ref}`
     - :math:`B_{ref}`
   * - Pyrokinetics
     - :math:`c_s`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_0`
   * - CGYRO
     - :math:`c_s`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_{\mathrm{unit}}`
   * - GS2
     - :math:`v_{th} = \sqrt{2 T_{ref} / m_{ref}}`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_0`
   * - STELLA
     - :math:`v_{th}`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_0`
   * - GX
     - :math:`c_s`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_0`
   * - GENE
     - :math:`c_s`
     - :math:`R_{\mathrm{major}}`
     - :math:`B_0`
   * - GKW
     - :math:`v_{th}`
     - :math:`R_{\mathrm{major}}`
     - :math:`B_0`
   * - IMAS
     - :math:`v_{th}`
     - :math:`R_{\mathrm{major}}`
     - :math:`B_0`
   * - TGLF
     - :math:`c_s`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_{\mathrm{unit}}`
   * - NEO
     - :math:`c_s`
     - :math:`a_{\mathrm{minor}}`
     - :math:`B_{\mathrm{unit}}`


-----------------------------------------
Conversion to and from SI units (contexts)
-----------------------------------------

Conversion between simulation (normalised) units and SI units is only possible
when the underlying physical reference values are known. This typically occurs
when a ``Pyro`` object is constructed from equilibrium and kinetics data, where
quantities such as :math:`n_{ref}`, :math:`T_{ref}`, :math:`B_{ref}`, and
:math:`L_{ref}` are explicitly defined.

Even when these reference values are available, conversion between SI units and
simulation units **must** be performed using a normalisation *context*. These
contexts are stored under ``pyro.norms.context`` and must be explicitly supplied
when converting quantities.

For example:

.. code-block:: python

    nu = pyro.local_species.electron.nu

    # Convert between simulation conventions (requires context)
    nu_gs2 = nu.to(pyro.norms.gs2, pyro.norms.context)

    # Convert to SI units (also requires context)
    nu_si = nu.to_base_units(pyro.norms.context)

The context provides the transformation rules linking simulation reference units
(e.g. ``vref``, ``lref``, ``rhoref``) to physical units. Without an active
context, conversions between physical and simulation units are not permitted.

If the reference values are not known (for example when loading a gyrokinetic
input file without equilibrium or kinetic profiles), conversion between
different simulation conventions remains possible, but conversion to or from
SI units will raise an error. The reference values can be specified using
``pyro.set_reference_values()``

.. warning::

   Conversion between physical (SI) units and simulation units always requires
   an explicit context. Calling ``.to()`` without supplying
   ``pyro.norms.context`` will raise an error if physical reference values are
   involved.

.. _Pint: https://pint.readthedocs.io/en/stable/
