=============================================
 Merge (multiple) species into a base species
=============================================

In the interest of computational gain, it may be useful to combine multiple species into a single species. A simple approach is to perform a density-weighted average which preserves quasi-neutrality. The attributes density (:math:`n`), charge (:math:`z`), density gradient (:math:`1/L_n`) and mass (:math:`M`) of the new species can be calculated as:

.. math::

   \begin{align*}
            n_m &= \sum_s n_s \\
            z_m &= \frac{\sum_s (z_s n_s)}{ n_m } \\
            M_m &= \frac{\sum_s (M_s n_s)} {n_m} \\.
            1/L_{n_m} &= \frac{\sum_s (z_s n_s(1/L_{n_s}))} { z_m n_m }
   \end{align*}

The summation :math:`\sum_s` is over all the species :math:`s` participating in the merge, whereas the subscript :math:`m` represents the merged-species.


This is achieved in ``pyrokinetics`` as follows:

.. code-block:: python

    >>> from pyrokinetics import template_dir, Pyro
    >>>
    >>>  # point to equilibrium and kinetics files
    >>> eq_file = template_dir / "test.geqdsk"
    >>> kinetics_file = template_dir / "jetto.jsp"
    >>>
    >>> # create pyro object which contains global properties
    >>> pyro = Pyro(
        eq_file=eq_file,
        kinetics_file=kinetics_file,
        kinetics_type="JETTO",
        kinetics_kwargs={"time": 550},
    )
    >>>
    >>> # generate local parameters at psi_n=0.5
    >>> pyro.load_local(psi_n=0.5, local_geometry="Miller")
    >>>
    >>> # merge species `impurity1` into `deuterium` (and remove impurity1 attributes)
    >>> # by calling the merge_local method
    >>> pyro.local_species.merge_species('deuterium',['impurity1'])
    >>>
    >>> # now write to your choice of GK code input (e.g. GENE)
    >>> pyro.write_gk_file(file_name="input.gene", gk_code="GENE")

A script `example_merge_species.py` is provided which does this.
