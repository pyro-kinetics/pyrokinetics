.. _sec-non-gridded-parameter-sets:

=====================================================
 Use a set of runs that is not a grid (PyroHypercube)
=====================================================

`PyroScan` describes a grid: each parameter has a list of values, and the runs
are the outer product of those lists. Plenty of parameter studies are not grids
— a Latin hypercube, points drawn from a distribution, or simply a directory of
runs somebody else made — and forcing them onto a grid either wastes runs or is
impossible.

`PyroHypercube` is the non-gridded sibling. ``parameter_dict`` holds one value
**per sample** rather than a list of values to multiply out, so sample ``i`` is
made of the ``i``-th value of every parameter, and the output ``Dataset`` has a
single ``sample`` dimension with the varied parameters attached to it as
non-dimension coordinates.

.. note::
   `PyroHypercube` writes input files and reads outputs. It never runs or
   submits anything.

Sampled points instead of a grid
--------------------------------

.. code:: python

    import numpy as np
    from pyrokinetics import Pyro, PyroHypercube, template_dir
    from pyrokinetics.units import ureg as units

    pyro = Pyro(gk_file=template_dir / "input.cgyro")

    rng = np.random.default_rng(0)
    n_samples = 20

    cube = PyroHypercube(
        pyro,
        parameter_dict={
            "ky": rng.uniform(0.1, 0.5, n_samples) / units.rhoref_pyro,
            "kappa": rng.uniform(1.0, 2.5, n_samples),
        },
        base_directory="run_directory",
    )

    cube.write()

That writes 20 input files, one per sample — not the 400 of the equivalent
`PyroScan`. Sample directories are named ``sample_0000``, ``sample_0001``, …
because scattered values do not make useful directory names (nearby samples
round onto the same one); pass ``sample_names`` to choose your own.

Reading a directory of runs
---------------------------

The runs need not have been made by pyrokinetics at all.
:meth:`~pyrokinetics.pyrohypercube.PyroHypercube.from_directory` finds the run
directories, reads each input file, and recovers the values that were varied:

.. code:: python

    cube = PyroHypercube.from_directory(
        "MTM_MODELS",
        pattern="iteration_*",
        params=["ky", "electron_temp_gradient"],
        gk_code="GS2",
    )

``pattern`` is a glob relative to the root, matching either the run directories
or the input files themselves (``"*/gs2.in"``); anything matched that is not a
run is skipped. ``params`` names the parameters to recover — either names known
to ``parameter_map`` (see
:py:meth:`pyrokinetics.pyroscan.PyroScan.load_default_parameter_keys`) or a dict
giving where each one lives in a `Pyro`, e.g. ``{"beta": ["numerics",
["beta"]]}``.

Analysing output
----------------

Outputs are loaded exactly as for `PyroScan`, but onto one ``sample``
dimension:

.. code:: python

    cube.load_gk_output()
    data = cube.gk_output.data

    data["growth_rate"]   # dims: (sample,)
    data["ky"]            # dims: (sample,) — a coordinate, not a dimension
    data["sample_name"]   # the run directory each sample came from

Because the parameters are coordinates rather than dimensions, plot against them
directly (``data.plot.scatter(x="ky", y="growth_rate")``), or use
``xarray.Dataset.set_index`` / ``groupby`` to organise the samples. The dataset
can be saved and reloaded with
:meth:`~pyrokinetics.pyrohypercube.PyroHypercube.to_netcdf` and
:meth:`~pyrokinetics.pyrohypercube.PyroHypercube.from_netcdf`.

Pointed at a `PyroScan` directory, a `PyroHypercube` reproduces it exactly; only
the shape of the output differs.

Writing the same points for another code
----------------------------------------

:meth:`~pyrokinetics.pyrohypercube.PyroHypercube.write_gk_decks` writes one
input file per sample, optionally converting to another code and applying
code-native settings on top of every deck:

.. code:: python

    cube.write_gk_decks(
        "TGLF",
        overrides={"NBASIS_MAX": 6, "WIDTH": 3.0},
        target="tglf_runs",
    )

``overrides`` takes the form that code's ``add_flags`` takes: flat for TGLF and
CGYRO, one dict per namelist for GS2 and GENE. Key case is matched to the input
file's own, so ``NBASIS_MAX`` and ``nbasis_max`` are the same setting.
