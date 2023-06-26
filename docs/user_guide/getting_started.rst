=================
 Getting Started
=================

Installation
------------

Install pyrokinetics with pip as follows:


.. code-block:: bash

  $ pip install --user pyrokinetics


Otherwise, for the latest version install directly with:


.. code-block:: bash

  $ git clone https://github.com/pyro-kinetics/pyrokinetics.git
  $ cd pyrokinetics
  $ pip install .


If you are developing pyrokinetics:

.. code-block:: bash

  $ pip install -e .[docs,tests]


Command Line Interface
----------------------

After installing, simple pyrokinetics operations can be performed on the command line
using either of the following methods:

.. code:: console

    $ python3 -m pyrokinetics {args...}
    $ pyro {args...}

For example, to convert a GS2 input file to CGYRO:

.. code:: console

    $ pyro convert CGYRO "my_gs2_file.in" -o "input.cgyro"

You can get help on how to use the command line interface or any of its subcommands
by providing ``-h`` or ``--help``:

.. code:: console

    $ pyro --help
    $ pyro convert --help


Library
-------

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
