=================
 Load TRANSP data
=================


This example just loads a TRANSP input file

.. literalinclude:: example_TRANSP.py

Notice that we don't have to tell `Pyro` which kind of equilibrium
file is used, as it can auto-detect this from the file
itself. You also do not need to specify the kinetics type as `JETTO` 

TRANSP profile is 2D so you have the option to select the time via
selection or indexing using the argument `kinetics_kwargs={"time": 550}`

TRANSP also stores the equilibrium file so that can be used as well.
Note below the code uses the auto-detect method built into pyro. In this
example we select the 10th time slice for the equilibrium and profiles

.. code:: python

    from pyrokinetics import Pyro, template_dir

    # Equilibrium file
    eq_file = template_dir / "transp.cdf"

    # Kinetics data file
    kinetics_file = template_dir / "transp.cdf"

    # Load up pyro object
    pyro = Pyro(
        eq_file=eq_file,
        eq_kwargs={"time_index": 10},
        kinetics_file=kinetics_file,
        kinetics_kwargs={"time_index": 10},
    )

    # Generate local Miller parameters at psi_n=0.5
    pyro.load_local(psi_n=0.5, local_geometry = "Miller")

    pyro.gk_code = "GS2"

    pyro.write_gk_file(file_name="test_transp.in")


