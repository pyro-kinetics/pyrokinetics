================
 Simple Example
================

.. todo::

   FIXME! This is an example of an example only!


This example just loads a GS2 input file::

    import pyrokinetics as pk

    pyro = pk.Pyro(gk_file="gs2.input")

Notice that we don't have to tell `Pyro` which kind of gyrokinetic
code the input file is for, as it can auto-detect this from the file
itself.
