================
 Load JETTO data
================


This example just loads a JETTO input file

.. literalinclude:: example_JETTO.py

Notice that we don't have to tell `Pyro` which kind of equilibrium
file is used, as it can auto-detect this from the file
itself. You also do not need to specify the kinetics type as `JETTO` 

JETTO profile is 2D so you have the option to select the time via
selection or indexing using the argument `kinetics_kwargs={"time": 550}`
