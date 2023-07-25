====================================================
Generate an input file from equilibrium and profiles
====================================================

Here a step-by-step guide on how to generate an input file
for any gyrokinetic codes supported in ``pyrokinetics``,
starting from equilibrium and profile files.


Let's first import ``pyrokinetics`` and define our equilibrium and input files. 

.. code-block:: python

   >>> from pyrokinetics import Pyro, template_dir
   >>> eq_file = template_dir / "test.geqdsk"
   >>> kinetics_file = template_dir / "jetto.cdf"
   >>> gk_file = template_dir / "input.cgyro"

The equilibrium file ``test.geqdsk`` and the kinetics file ``jetto.cdf``
are stored in the template foder and used here as an example.
The gyrokinetic file ``input.cgyro`` is our input file template where
we set all the extra flags we need. The input parameters related to the
geometry and species will be added to this template by ``pyrokinetics``.
We now load these files into ``pyrokinetics``:

.. code-block:: python

   >>> pyro = Pyro(
        eq_file=eq_file,
        kinetics_file=kinetics_file,
	kinetics_type="JETTO",
        kinetics_kwargs={"time": 550},
	gk_file=gk_file,
    )


During initialization, `Pyro` calls `read_equilibrium` from
the class :py:class:`Equilibrium` and initializes the class :py:class:`Kinetics`.
The global equilibrium and proofiles are now stored in ``pyro``.
Let's suppose we want to generate an input file for a local gyrokinetic
simulation at :math:`\Psi_n = 0.5`. This requires loading the local geometry
at the chosen surface, which can be done by simply calling the ``load_local`` method:

.. code-block:: python

   >>> pyro.load_local(psi_n=0.5, local_geometry="Miller")

Here we have used the ``Miller`` parametrization of the local surface. Other
parametrizations are possible in ``pyrokinetics``. See the output of ``pyro.supported_local_geometries``.
Before generating an input file, we need to specify ``gk_code``:

.. code-block:: python

   >>> pyro.gk_code = "CGYRO"

Note that ``gk_code`` can be any code supported in ``pyrokinetics``, which can
be found in ``pyro.supported_gk_inputs``. 
We are now ready to write our input file:

.. code-block:: python

   >>> pyro.write_gk_file(file_name="test_jetto.cgyro")

Alternatively, ``gk_code`` can be pass as keyword argument to ``write_gk_file``,
.. code-block:: python

   >>> pyro.write_gk_file(file_name="test_jetto.cgyro", gk_code="CGYRO")

