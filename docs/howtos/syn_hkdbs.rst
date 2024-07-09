.. _sec-syn-hk-dbs:

============================================
        Using synthetic diagnostic
============================================

The following discusses how one uses the synthetic diagnostic for ``pyrokinetics``. The steps in the synthetic diagnostic are the following:

1. Inputs are diagnostic specific (diagnostic, filter, k, location, resolution, local rhos). See example_syn_hk_dbs.py
2. Load equilibrium, kinetics files. Find scattering location theta. See __init__ in class SyntheticHighkDBS
3. Map (kn, kb) to (kx, ky) for all k's / channels specified in 1. See function mapk
4. Load GK output data (fluctuation moment file). See function get_syn_fspec 
5. For each input condition (eg. for each k in highk/DBS), filter simulation data. See class Filter, functions apply_filter, get_syn_fspec
6. Generate synthetic spectra and make plots. See functions get_syn_fspec, plot_syn


1. We first point is to provide input data from the specific diagnostic (high-k, DBS for now). See example_syn_hk_dbs.py as reference. 
Here we have defined the necessary inputs for the synthetic diagnostic. 

.. code:: python

    from pyrokinetics.diagnostics import SyntheticHighkDBS
    import numpy as np
    import matplotlib.pyplot as plt
    from pyrokinetics.units import ureg
        
    # inputs
    diag = "highk"                      # 'highk', 'dbs', 'rcdr', 'bes'
    filter_type = ("gauss")             # 'bt_slab', 'bt_scotty', 'gauss' 
    Rloc = 2.89678 * ureg.meter         # 
    Zloc = 0.578291 * ureg.meter        # [m]       
    Kn0_exp = np.asarray([0.5, 1]) / ureg.centimeter      # [cm-1]
    Kb0_exp = np.asarray([0.1, 0.2]) / ureg.centimeter    # [cm-1]
    wR = 0.1 * ureg.meter       #
    wZ = 0.05 * ureg.meter      # 
    eq_file = "path_to_equilibrium_file"
    kinetics_file = eq_file
    simdir = "path_to_simulation_directory"
    savedir = "path_to_save_figures"
    if_save = 0
    fsize = 22

## 2. 
Next, call SyntheticHighkDBS, which defines the syn_diag object. 

.. code-block:: python 
    syn_diag = SyntheticHighkDBS(
        diag=diag,
        filter_type=filter_type,
        Rloc=Rloc,
        Zloc=Zloc,
        Kn0_exp=Kn0_exp,
        Kb0_exp=Kb0_exp,
        wR=wR,
        wZ=wZ,
        eq_file=eq_file,
        kinetics_file=kinetics_file,
        simdir=simdir,
        savedir=simdir,
        fsize=fsize,
    )

In the __init__ of SyntheticHighkDBS, we also calculate the scattering location in theta from a given (Rloc,Zloc) location. 
For that, we first find the radial location (poloidal flux) corresponding to (Rloc, Zloc). 
Then, we fit the flux surface using the local geometry specification (eq. "Miller", "MXH"), similar to what should be used in the GK simulation output. 

.. code:: python
    # calcualte thetaloc
    pyro = Pyro(
        eq_file=eq_file,
        kinetics_file=kinetics_file,
        gk_file=simdir + "/input.cgyro",
    )
    self.pyro = pyro
    self.eq = pk.read_equilibrium(eq_file)
    self.psin = self.eq._psi_RZ_spline(
        Rloc * pyro.norms.units.meter, Zloc * pyro.norms.units.meter
    ) / (self.eq.psi_lcfs - self.eq.psi_axis)
    pyro.load_local(psi_n=self.psin, local_geometry="Miller")
    self.geometry = pyro.local_geometry
    pyro.load_metric_terms()

Once we have a radial location (poloidal flux) and flux surface parametrization, we find the theta grid point that corresponds to an (R,Z) that is closest to (Rloc, Zloc). 
For theta locations above the magnetic axis Z location, we can calculate it as:

.. code:: python         
    # find thetaloc:
    thetatmp = self.geometry.theta[self.geometry.Z > self.geometry.Z0]
    Rtmp = self.geometry.R[self.geometry.Z > self.geometry.Z0] * self.a_minor  # [m]
    Ztmp = self.geometry.Z[self.geometry.Z > self.geometry.Z0] * self.a_minor  # [m]
    tmp_ind = np.argmin(np.abs(Rtmp - Rloc))
    self.thetaloc = thetatmp[tmp_ind]  # np.interp(Zloc, Ztmp, thetatmp)
    self.Rtmp = Rtmp[tmp_ind]
    self.Ztmp = Ztmp[tmp_ind]

This gives the following plot: 
.. image:: figures/jet_example_scatloc.png       
   :width: 600

## 3. 
Next, call the function mapk. Given a pair (kn, kb), we calculate the corresponding (kx, ky) in the simulation grid. Here, we need to first define a right handed coordinate system. 
We use the basis of unit vectors :math:`(\hat{\mathbf{b}}, \hat{\mathbf{e}}_n, \hat{\mathbf{e}}_b)`. Here :math:`\hat{\mathbf{b}}` is along the background magnetic field. 
The normal unit vector :math:`\hat{\mathbf{e}}_n = \nabla \psi/|\nabla \psi|` is normal to the flux surface. 
The binormal unit vector :math:`\hat{\mathbf{e}}_b = \hat{\mathbf{b}} \times \hat{\mathbf{e}}_n` is in the binormal direction, that is, in the flux surface and perpendicular to :math:`\hat{\mathbf{b}}`.
Additionally, in an axisymmetric device, we can write the magnetic field as :math:`\mathbf{B} = \nabla \alpha \times \nabla \psi`. 
With this, the normal and binormal components of the perpendicular wave vector :math:`\mathbf{k}_\perp = k_n \hat{\mathbf{e}}_n + k_b \hat{\mathbf{e}}_b` are 

.. math::
    \begin{equation}
        \begin{alignedat}{2}
        & k_n = \mathbf{k}_\perp \cdot \hat{\mathbf{e}}_n = - n \frac{\nabla \alpha \cdot \nabla r}{| \nabla r |} + k_x |\nabla r|, \\
        & k_b = \mathbf{k}_\perp \cdot \hat{\mathbf{e}}_b = - n \left( \hat{\mathbf{b}} \times \frac{\nabla r}{r} \right) \cdot \nabla \alpha
        \end{alignedat}
        \label{knkb_map}
    \end{equation}

where :math:`k_x = 2 \pi p / L_x` is the radial wave number definition in pyro, :math:`n` is the toroidal mode number, and :math:`L_x` is the radial extent of the numerical simulation. 
The mapping in equation \ref{knkb_map} is performed within the function mapk.py, and executed as follows: 

.. code:: python
    # map k
    syn_diag.mapk()

## 6. 
Next, apply the synthetic diagnostic. Use get_syn_fspec and plot_syn

.. code:: python         
    # apply synthetic diagnostic:
    [pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann] = syn_diag.get_syn_fspec( 0.7, 1, savedir, if_save )

    syn_diag.plot_syn()

(continue here)

For linear simulations, one tends to only have a single ``ky`` and ``kx``, and thus
data variables such as ``growth_rate`` and ``mode_frequency`` are essentially 1D
functions of time. These can be plotted using ``plot`` (see xarray's `Plotting`_ for further details):

For data variables with higher dimensions, indexing can be performed using the standard
xarray dataset methods, such as ``.sel`` and ``.isel``. For example, to plot the ``phi``
eigenfunction at the final time point as a function of ``theta``:

And analogously for the field data, for example looking at
the magnitude of the ``phi`` fluctuations at :math:`\theta = 0.0`:

Details regarding normalisations and units can be found in `sec-normalisation-docs`.

.. _Plotting: https://docs.xarray.dev/en/stable/user-guide/plotting.html
.. _xarray Dataset: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
