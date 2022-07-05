Introduction
============

This project aims to standardise gyrokinetic analysis by providing a single
interface for reading and writing input and output files from different
gyrokinetic codes, normalising to a common standard, and performing standard
analysis methods.

A general ``Pyro`` object can be loaded either from simulation/experimental data or
from an existing gyrokinetics file.

Currently pyrokinetics can do the following:

-  Read data in from:

  -  Gyrokinetic input files
  -  Integrated modelling/Global Equilibrium simulation output

-  Write input files for various gyrokinetics codes
-  Generate N-D ``Pyro`` object for scans
-  Read in gyrokinetic outputs
-  Standardise analysis of gyrokinetics outputs

Future development includes:

-  Submit gyrokinetics simulations to cluster
-  Integrate with GKDB to store/catalog GK runs

At a minimum pyrokinetics needs the local geometry and species data. This can be
taken from a GK input file. At the moment the following local gyrokinetic codes
are supported:

-  `GS2 <https://gyrokinetics.gitlab.io/gs2/>`_
-  `CGYRO <http://gafusion.github.io/doc/cgyro.html>`_
-  `GENE <http://www.genecode.org>`_

It is also possible to load in global data from the following codes, from which
local parameters can be calculated.

-  `TRANSP <https://transp.pppl.gov>`_
-  JETTO
-  SCENE


Example scripts can be found in the examples folder where GK/Transport code data
is read in and GK outputs are read in. Here's how you could generate input files
for GS2, CGYRO, and GENE from one SCENE equilibrium::

    from pyrokinetics import Pyro

    # Create a Pyro object from GEQDSK and SCENE files
    # By setting 'gk_code' to "CGYRO", we implicitly load a CGYRO input file template
    # All file types are inferred automatically
    pyro = Pyro(
        gk_code="CGYRO",
        eq_file="test.geqdsk",
        kinetics_file="scene.cdf",
    )

    # Generate local Miller parameters at psi_n=0.5
    pyro.load_local(psi_n=0.5, local_geometry="Miller")

    # Write CGYRO input file using default template
    pyro.write_gk_file(file_name="test_scene.cgyro", gk_code="CGYRO")

    # Write single GS2 input file
    pyro.write_gk_file(file_name="test_scene.gs2", gk_code="GS2")

    # Write single GENE input file
    pyro.write_gk_file(file_name="test_scene.gene", gk_code="GENE")



Installation
------------


Install pyrokinetics with pip as follows:


.. code-block:: bash

  $ pip install --user pyrokinetics


Otherwise, for the latest version install directly with:


.. code-block:: bash

  $ git clone https://github.com/pyro-kinetics/pyrokinetics.git
  $ cd pyrokinetics
  $ python setup.py install --user


Structure
---------


The ``Pyro`` object is structured as follows


-  :ref:`sec-equilibrium`

  -  Accessed via ``pyro.eq``

  -  Represents full 2D equilibrium

  -  Only loaded when full equilibrium data is provided


*  :ref:`sec-local_geometry`

  *  Accessed via ``pyro.local_geometry``

  *  Represents local geometry of a flux surface

  *  Current supported LocalGeometry subclasses are:

    *  :ref:`sec-miller`


-  :ref:`sec-kinetics`

  -  Accessed via ``pyro.kinetics``

  -  Represents 1D profiles for each kinetic species

  -  Only loaded when full profile data is provided


*  :ref:`sec-local_species`

  *  Accessed via ``pyro.local_species``

  *  Contains local species parameters


*  :ref:`sec-numerics`

  *  Accessed via ``pyro.numerics``

  *  Sets up numerical grid and certain physics models


*  :ref:`sec-gk_input`

  *  Accessed via ``pyro.gk_input``

  *  Holds gyrokinetics input data and methods specific to each gyrokinetics code

  *  Can be used to directly populate LocalGeometry and LocalSpecies

  *  Used to set Numerics

  *  Current supported GKInput subclasses are:

    *  :ref:`sec-gk_input_gs2`
    *  :ref:`sec-gk_input_cgyro`
    *  :ref:`sec-gk_input_gene`

.. image:: rst_docs/figures/pyro_structure.png
  :width: 600


Analysis
-----------
Once you have a completed simulation you can read the output into a pyro object.

*  :ref:`sec-gk_output`

  *  Accessed via ``pyro.gk_output``

  *  Stores output from a GK simulation

  *  Data stored in `Xarray <https://docs.xarray.dev/en/stable/>`_ ``Datasets``


.. toctree::
   :maxdepth: 3
   :caption: Code structure
   :name: code_structure

   rst_docs/pyro.rst
   rst_docs/equilibrium.rst
   rst_docs/local_geometry.rst
   rst_docs/miller.rst
   rst_docs/kinetics.rst
   rst_docs/local_species.rst
   rst_docs/gk_input.rst
   rst_docs/gk_input_gs2.rst
   rst_docs/gk_input_gene.rst
   rst_docs/gk_input_cgyro.rst
   rst_docs/gk_output.rst
   rst_docs/numerics.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
