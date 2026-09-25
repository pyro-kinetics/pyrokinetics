Semi-Consistent Modification of Grad–Shafranov Equilibria Using Pyrokinetics
============================================================================

This script demonstrates how to **semi-consistently modify a Grad–Shafranov equilibrium**
within the `Pyrokinetics <https://github.com/pyrokinetics/pyrokinetics>`_ framework
while maintaining a physically meaningful balance between **bootstrap current**, **external current**, and **pressure gradients**.

The approach enables **parameter scans** (e.g. increasing β or temperature gradients)
while retaining consistency between current profiles and magnetic shear, following the relation:

.. math::

   \langle B^2 \rangle \frac{f'}{\mu_0} + f p' = \langle J_\text{tot} \cdot B \rangle = \langle J_\text{bs} \cdot B \rangle + \langle J_\text{ext} \cdot B \rangle


Overview
--------

The script modifies equilibrium parameters (e.g., density, temperature, gradients)
in a **semi-consistent** manner by recalculating the **bootstrap** and **external** currents
using the :class:`Redl2021` neoclassical diagnostic model in Pyrokinetics.

The **key goal** is to perform physically reasonable parameter scans
without regenerating the full Grad–Shafranov equilibrium from scratch.


Workflow Summary
----------------

1. Choose input data source (Kinetics/Equilibrium, or GKInput )
2. Initialize a :class:`Pyro` object from equilibrium and kinetic profiles
3. Load local geometry and numerical parameters
4. Enforce β′ consistency
5. Compute baseline bootstrap and external currents
6. Apply scaling to β and temperature/density gradients
7. Recompute currents and update magnetic geometry
8. Print diagnostic results and compare equilibrium parameters


1. Choose Input Source
~~~~~~~~~~~~~~~~~~~~~~

Depending on the chosen ``run`` type, the script loads equilibrium (``eq_file``)
and kinetic (``kinetics_file``) data using the appropriate readers. This can also
be performed with a local GK input file

.. code-block:: python

   run = "TRANSP"

Each option corresponds to a different file format and interface for equilibrium and kinetic data:

+-----------+------------------+---------------+-------------------------------------+
| Run Type  | Equilibrium Type | Kinetic Type  | Example Files                       |
+===========+==================+===============+=====================================+
| TRANSP    | TRANSP CDF       | TRANSP CDF    | ``transp.cdf``                      |
+-----------+------------------+---------------+-------------------------------------+
| SCENE     | GEQDSK           | SCENE CDF     | ``test.geqdsk``, ``scene.cdf``      |
+-----------+------------------+---------------+-------------------------------------+
| JETTO     | GEQDSK           | JETTO JSP     | ``transp_eq.geqdsk``, ``jetto.jsp`` |
+-----------+------------------+---------------+-------------------------------------+
| GKInput   | None             | None          | ``input.gene``                      |
+-----------+------------------+---------------+-------------------------------------+


2. Initialize the Pyro Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`Pyro` object encapsulates the magnetic equilibrium, kinetic profiles,
and derived quantities:

.. code-block:: python

   pyro = Pyro(eq_file=..., eq_type=..., kinetics_file=..., kinetics_type=...)

Once loaded, local parameters are initialized at a specific normalized flux surface
(``ψ_n = 0.5``) using the MXH local geometry model:

.. code-block:: python

   pyro.load_local(psi_n=0.5, local_geometry="MXH", show_fit=False)

Numerical parameters are configured for high-resolution field-aligned computations:

.. code-block:: python

   pyro.load_numerics(load_geometry_species_data=True, ntheta=512, apar=True, bpar=True)

Alternatively if loading a local GK input file then one can do

.. code-block:: python

   pyro = Pyro(gk_file=...)

3. Enforce Consistent β′
~~~~~~~~~~~~~~~~~~~~~~~~

The magnetic equilibrium should made consistent with the pressure and magnetic gradients by enforcing:

.. code-block:: python

   pyro.enforce_consistent_beta_prime()

This ensures internal consistency between the magnetic geometry and kinetic pressure gradients.


4. Compute Baseline Currents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bootstrap, external, and total currents are computed using the :class:`Redl2021` neoclassical model:

.. code-block:: python

   old_redl = Redl2021(pyro, ntheta=512)
   old_jbsdotb = old_redl.JbsdotB
   old_jextdotb = old_redl.JextdotB
   old_jdotb = old_redl.JdotB


5. Apply Parameter Modifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, the equilibrium parameters are scaled to simulate physical changes:

- ``n_factor``: density scaling
- ``T_factor``: temperature scaling
- ``alt_scale``: gradient scaling (e.g., a/L_T)

.. code-block:: python

   n_factor = 1.2
   T_factor = 1.0
   beta_scale = n_factor * T_factor
   alt_scale = 1.2

This modifies β (via pressure scaling) and normalized gradients (via inverse scale lengths):

.. code-block:: python

   pyro.numerics.beta = beta_scale * pyro.norms.beta
   for species in pyro.local_species.names:
       pyro.local_species[species].inverse_lt *= alt_scale

Enforce β′ consistency again afterward:

.. code-block:: python

   pyro.enforce_consistent_beta_prime()


6. Recompute Currents After Modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recalculate bootstrap and external currents using the modified equilibrium:

.. code-block:: python

   new_redl = Redl2021(pyro, ntheta=512)
   new_jbsdotb = new_redl.JbsdotB
   new_jextdotb = new_redl.JextdotB

The modified total current is then constructed as:

.. math::

   \langle J_\text{tot} \cdot B \rangle_\text{new} =
   \langle J_\text{bs} \cdot B \rangle_\text{new} +
   \langle J_\text{ext} \cdot B \rangle_\text{old} \frac{T_\text{factor}}{n_\text{factor}}


7. Update Magnetic Geometry (F′ and ŝ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the new total current, calculate the new :math:`F'` and magnetic shear :math:`\hat{s}`:

.. code-block:: python

   new_Fprime = new_redl.get_Fprime_from_total_current(mod_jdotb)
   new_shat = lg.get_s_hat(new_Fprime)
   lg.shat = new_shat


8. Output and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~

A diagnostic summary is printed to compare the before/after values:

.. code-block:: text

   Increase beta by 20.0%
   Increase a/LT by 0.0%
   Original F' 0.1135
   New F'      0.0438
   Original s hat 2.15
   New s hat     1.40


Example Results
---------------

Example run with a 20% β increase:

+-------------------+----------+----------+-----------------------------+
| Quantity          | Original | New      | Description                 |
+===================+==========+==========+=============================+
| :math:`\beta`     | -        | +20%     | Pressure scaling            |
+-------------------+----------+----------+-----------------------------+
| :math:`F'`        | 0.1135   | 0.0438   | Decrease in current drive   |
+-------------------+----------+----------+-----------------------------+
| :math:`\hat{s}`   | 2.15     | 1.40     | Reduction in magnetic shear |
+-------------------+----------+----------+-----------------------------+

Interpretation:
   Increasing β (pressure) leads to a **reduction in magnetic shear**, consistent with
   flatter q-profiles. However, an excessive increase in bootstrap current could
   lead to a **runaway** scenario with enhanced instability (e.g. internal ballooning).


Conceptual Summary
------------------

Key Physical Relation
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \langle B^2 \rangle \frac{f'}{\mu_0} + f p' =
   \langle J_\text{tot} \cdot B \rangle =
   \langle J_\text{bs} \cdot B \rangle + \langle J_\text{ext} \cdot B \rangle

Given that each term can be computed, the relation can be *inverted* to find
updated equilibrium quantities (notably :math:`F'` and :math:`\hat{s}`)
under modified plasma conditions.

Guiding Principles
~~~~~~~~~~~~~~~~~~

- Keep the **poloidal flux Ψ(R,Z)** fixed, preserving the MXH/Miller fit.
- Allow **pressure** and **current profiles** to evolve self-consistently.
- Optionally assume :math:`\langle J_\text{ext} \cdot B \rangle` scales as :math:`n / T^2`.
- Enforce **β′ consistency** for a physically meaningful equilibrium.


Discussion and Open Questions
-----------------------------

- Should the **q-profile** be fixed, or should it evolve with :math:`\hat{s}` for consistency?
- What constraints ensure :math:`\langle J_\text{tot} \cdot B \rangle` remains positive and physical?
- How do these semi-consistent modifications affect MHD stability?


Script Reference
----------------

**Location:**
   ``docs/example_modify_shear.py``

**Purpose:**
   Implements the above strategy for testing β and gradient scaling effects on equilibrium shear
   using the Pyrokinetics framework.

