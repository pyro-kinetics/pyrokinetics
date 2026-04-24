.. default-role:: math
.. _sec-kinetics-tutorial:


Kinetics Tutorial
=================

Pyrokinetics can be used to read and analyse plasma kinetic profile files as
produced by transport or interpretive modelling codes such as SCENE, JETTO, or
other profile generators. These files contain thermodynamic quantities for each
plasma species (such as density and temperature) defined on flux surfaces,
which may then be used to set up gyrokinetic simulations.

See :ref:`sec-reading-kinetics-files` for information about reading files and
using :py:class:`Kinetics` objects, and :ref:`sec-kinetics-plotting` for how to
visualise kinetic profiles.


.. _sec-kinetics-background:

Background
----------

In gyrokinetics simulations, plasma behaviour is determined not only by the
magnetic equilibrium but also by the kinetic state of each particle species.
For a given species `s`, the key quantities typically required are:

* Density: `n_s(\psi)`
* Temperature: `T_s(\psi)`
* Mass: `m_s`
* Charge: `Z_s(\psi)`
* Toroidal rotation: `\Omega_s(\psi)`

These quantities are commonly defined on flux surfaces labelled by the
normalised poloidal flux:

.. math::

    \psi_N = \frac{\psi - \psi_{\text{axis}}}{\psi_{\text{LCFS}} - \psi_{\text{axis}}}

where `\psi_N = 0` corresponds to the magnetic axis and `\psi_N = 1` corresponds
to the last closed flux surface (LCFS). Note charge is usually constant for a given
species but in some heavy impurity cases it can be vary due differing charge states

Gradients with respect to the normalised minor radius `\rho = r/a` are also
frequently used in gyrokinetics, for example:

.. math::

    a/L_{T_s} = -\frac{1}{T_s}\frac{\partial T_s}{\partial \rho}

These gradients are provided directly through the :py:class:`Species` interface.


.. _sec-reading-kinetics-files:

Reading Files
-------------

Kinetics files may be read using the function ``read_kinetics``. The file type
may be specified explicitly, or inferred automatically when possible.

.. code-block:: python

    >>> import pyrokinetics as pk
    >>> kin = pk.read_kinetics("my_kinetics_file")

We can inspect the contents of the ``Kinetics`` object by printing it:

.. code-block:: python

    >>> print(kin)
    <pyrokinetics.Kinetics>

A ``Kinetics`` object contains a collection of :py:class:`Species` objects,
accessible through the ``species_data`` attribute. This behaves like a
dictionary, but also allows attribute-style access:

.. code-block:: python

    >>> kin.species_data["electron"]
    >>> kin.species_data.electron

The number and names of species may be obtained using:

.. code-block:: python

    >>> kin.nspec
    >>> kin.species_names

Each species provides access to its physical quantities as functions of
`\psi_N`:

.. code-block:: python

    >>> psi_n = 0.5
    >>> ne = kin.species_data.electron.get_dens(psi_n)
    >>> Te = kin.species_data.electron.get_temp(psi_n)

Units are handled using Pint_, so returned values carry physical units.


Species Objects
---------------

Each :py:class:`Species` object contains the following information:

* Charge
* Mass
* Density profile
* Temperature profile
* Rotation profile
* Normalised minor radius `\rho = r/a`

These quantities are typically stored internally as interpolating functions
over `\psi_N`, allowing evaluation at arbitrary flux surfaces.

For example:

.. code-block:: python

    >>> electron = kin.species_data.electron
    >>> psi_n = 0.3

    >>> electron.get_charge(psi_n)
    >>> electron.get_mass()
    >>> electron.get_dens(psi_n)
    >>> electron.get_temp(psi_n)
    >>> electron.get_angular_velocity(psi_n)

Normalised gradients frequently used in gyrokinetics are also available:

.. code-block:: python

    >>> electron.get_norm_temp_gradient(psi_n)
    >>> electron.get_norm_dens_gradient(psi_n)
    >>> electron.get_norm_ang_vel_gradient(psi_n)

These correspond to:

.. math::

    -\frac{1}{f}\frac{\partial f}{\partial \rho}

for the relevant field `f`.


Writing Files
-------------

``Kinetics`` objects may be written to NetCDF files:

.. code-block:: python

    >>> kin.to_netcdf("my_kinetics.nc")

They can later be reloaded using:

.. code-block:: python

    >>> kin = pk.read_kinetics("my_kinetics.nc")


.. _sec-kinetics-plotting:

Plotting
--------

The ``Kinetics`` class provides built-in plotting utilities using Matplotlib_.
By default, density, temperature, and rotation profiles for all species are
plotted together.

.. code-block:: python

    >>> kin.plot(show=True)

This produces three panels showing:

* Density
* Temperature
* Angular frequency

Profiles may be plotted either against `\psi_N` (default) or against the
normalised minor radius `r/a`:

.. code-block:: python

    >>> kin.plot(x_grid="r/a", show=True)

As with other plotting utilities in Pyrokinetics:

* A Matplotlib ``Axes`` object may optionally be supplied.
* The function returns the axes used.
* Passing ``show=True`` immediately displays the figure.

Example with custom axes:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    kin.plot(ax=ax)
    plt.show()


See the :any:`Kinetics` and :any:`Species` API documentation for more details.


.. _Pint: https://pint.readthedocs.io/en/stable/
.. _Matplotlib: https://matplotlib.org/

