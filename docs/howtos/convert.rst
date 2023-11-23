===========================================
 Convert input files from code A to code B
===========================================

One way to do this is from the command line:

.. code:: console

    $ pyro convert CGYRO "my_gs2_file.in" -o "input.cgyro"

Pyrokinetics will automatically detect the type of the input file. Note
that when converting between codes only the data stored in `LocalGeometry`,
`LocalSpecies` and `Numerics` is transferred over. Any other CGYRO input
parameters are taking from a template file which is taking from `pyrokinetics.template_dir`

This is effectively the same as doing the following in a python script

.. code:: python

    from pyrokinetics import Pyro

    # Point to GS2 input file
    gs2_template = "my_gs2_file.in"

    # Load in GS2 file
    pyro = Pyro(gk_file=gs2_template)

    # Switch to CGYRO
    pyro.gk_code = "CGYRO"

    # Write CGYRO input file
    pyro.write_gk_file(file_name="input.cgyro")


If you have your own template CGYRO input file with specific flags
turned on you can do the following 


.. code:: console

    $ pyro convert CGYRO "my_gs2_file.in" -o "input.cgyro" --template "my_cgyro_template_file.in"

Or in a python console for the last line simply do


.. code:: python

    # Write CGYRO input file using a CGYRO template file
    pyro.write_gk_file(file_name="input.cgyro", template_file="my_cgyro_template_file.in")

