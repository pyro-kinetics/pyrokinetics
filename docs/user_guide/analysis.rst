==========
 Analysis
==========

Once you have a completed simulation, you can read the output into a
`Pyro` object. By default, some common analysis is either read from
the output file(s) or automatically performed for you.

- :class:`GKOutputReader`

  - Accessed via ``pyro.gk_output``
  - Stores output from a GK simulation
  - Data stored in `Xarray <https://docs.xarray.dev/en/stable/>`_ ``Datasets``
