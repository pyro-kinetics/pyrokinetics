.. _sec-nd-parameter-scans:

============================================
 Perform a N-D scan in parameters A,B,C etc.
============================================

It is often common to perform parameters scan when running gyrokinetic analysis. The `PyroScan` object allows for easy
generation of input files for an N-D parameter scan along with the ability to read in the outputs for all the runs.


Simple 1-D scan
---------------

To start of you need a fully initialised `Pyro` object and a dictionary object that outlines the variable to scan
through. Here we scan through :math:`k_y\rho_s`

.. code:: python

    # Initialise a Pyro object
    from pyrokinetics import Pyro, PyroScan, template_dir
    pyro = Pyro(gk_file= template_dir / "input.cgyro")

    # Write input files
    param_dict = {"ky": [0.1, 0.2, 0.3]}

    # Create PyroScan object
    pyro_scan = PyroScan(
        pyro,
        param_dict,
        base_directory="run_directory"
    )

    # Write input files
    pyro_scan.write()

This will write 3 different input files where the parameter :py:attr:`pyrokinetics.numerics.Numerics.ky` has been
changed, with each input file in its own directory under a base directory of ``run_directory``.

The variable here is always defined in pyrokinetics default units, so when using a code that has a different normalised
be aware that the values defined here will be converted to the code own units. See `normalisations`.

This also created a ``pyroscan.json`` file that can be used to initialise a run

Note ``ky`` here is a default parameter key to scan through with others listed here
:py:meth:`pyrokinetics.pyroscan.PyroScan.load_default_parameter_keys`. To do a bespoke parameter see below


Running/Submitting simulations
------------------------------

To actually run/submit these simulations to the cluster, it is necessary to go into each input file directory and
run/submit from there. These directories are stored in ``pyro_scan.run_directories``

.. code:: python

    from shutil import copy
    from pyrokinetics.pyroscan import cd
    import os

    for run_dir in pyro_scan.run_directories:

        # Use pyroscan change directory method to go into each run directory
        with cd(run_dir):
            # Run a simulation
            os.system('cgyro -n 32 -nomp 3 ')

            # Copy (existing) batch script and submit
            copy("batch.src", run_dir)
            with cd(run_dir):

                # Submit job with run directory names
                os.system(f"sbatch -J pyroscan_{rel_path} batch.src")


Analysing output
-----------------

Once the simulations are complete the output data is stored as an xarray ``Dataset``. The dimensions
of the ``Dataset`` are the parameter values names specified in ``params_dict`` along with the original
dimensions of the data


.. code:: python

    # Load output from
    pyro_scan.load_gk_output()

    data = pyro_scan.gk_output
    growth_rate = data['growth_rate']
    mode_frequency = data['mode_frequency']

    growth_rate_tolerance = data['growth_rate_tolerance']
    growth_rate = growth_rate.where(growth_rate_tolerance < 0.1)
    mode_frequency = mode_frequency.where(growth_rate_tolerance < 0.1)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11,9))

    ax1.plot(growth_rate.ky, growth_rate.data)
    ax2.plot(mode_frequency.ky, mode_frequency.data)

    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel(r'$\gamma (c_{s}/a)$')
    ax2.set_ylabel(r'$\omega (c_{s}/a)$')
    ax2.set_xlabel(r"$k_y \rho_s$")

    fig.tight_layout()
    plt.show()

For linear runs the following data is stored but only for the final time slice

 - growth_rate
 - mode_frequency
 - eigenfunctions
 - growth_rate_tolerance
 - particle (flux)
 - heat (flux)

Nonlinear simulations are not currently supported

Higher dimensional scans
------------------------

To perform a higher dimensional scan, the only additional requirement is to extend ``param_dict`` with more key:value
pairs

.. code:: python

    # Initialise a Pyro object
    from pyrokinetics import Pyro, PyroScan, template_dir
    pyro = Pyro(gk_file= template_dir / "input.cgyro")

    # Define parameters
    param_1 = "ky"
    values_1 = [0.1, 0.2, 0.3]

    param_2 = "kappa"
    values_2 = [1.0, 1.5, 2.0]

    param_dict = {
                  param_1: values_1,
                  param_2: values_2,
                 }

    # Create PyroScan object
    pyro_scan = PyroScan(
        pyro,
        param_dict,
        value_fmt=".3f",
        value_separator="_",
        parameter_separator="_",
        base_directory="run_directory"
    )

After which the process is the same. Note that an outer product is formed of all the specified parameters so the number
of runs can become large very quickly.

Here we have specified:

 - How to separate each value from its parameter: ``value_separator``
 - How to separate each different value: ``parameter_separator``
 - How each value is formatted: ``value_fmt``

If we wanted to have each run in its own directory then we could set ``parameter_separator="/"``

Outputs are loaded in the same with, but now with the elongation (:attr:`pyrokinetics.local_geometry.miller.LocalGeometryMiller.kappa`) as a dimension too


Bespoke parameters/functions
----------------------------

Given that the `PyroScan` class can't contain all parameters it is possible to modify any parameter
defined in a `Pyro` object via the :meth:`pyrokinetics.pyroscan.PyroScan.add_parameter_key` method


.. code:: python

    # Use existing parameter
    param_1 = "q"
    values_1 = np.arange(1.0, 1.5, 0.1)

    # Add new parameter to scan through
    param_2 = "my_electron_density_gradient"
    values_2 = np.arange(0.0, 1.5, 0.5)

    # Dictionary of param and values
    param_dict = {param_1: values_1, param_2: values_2}

    # Create PyroScan object
    pyro_scan = PyroScan(
        pyro,
        param_dict,
        value_fmt=".3f",
        value_separator="_",
        parameter_separator="_",
        base_directory="run_directory",
    )

    # Add in path to each defined parameter to scan through
    pyro_scan.add_parameter_key(param_1, "local_geometry", ["q"])
    pyro_scan.add_parameter_key(param_2, "local_species", ["electron", "inverse_ln"])


The scan as is would violate quasi-neutrality as the density gradient is changing for only one species. So it is
possible to apply a function to the `Pyro` object after each variable is set.

.. code:: python

    def maintain_quasineutrality(pyro):
        for species in pyro.local_species.names:
            if species != "electron":
                pyro.local_species[species].inverse_ln = pyro.local_species.electron.inverse_ln

    # If there are kwargs to function then define here
    param_2_kwargs = {}

    # Add function to pyro
    pyro_scan.add_parameter_func(param_2, maintain_quasineutrality, param_2_kwargs)

This allows for multiple parameters to be changed in tandem and can be defined for each input parameter.
