====================
Bicoherence analysis
====================

Bicoherence is an analysis method that determines the level of triadic mode
coupling by examining whether three modes remain phase locked throughout the
statistically equivalent periods of a simulation. Given that nonlinear
simulations have two wavenumbers in two direction we determine the 2D
bicoherence. The 2D bicoherence square for a complex field :math:`X` is given by

.. math::
    b^2(X) = \frac{|B(k_{x1}, k_{y1},k_{x2}, k_{y2})|^{2}}{\langle
    |X(k_{x1}, k_{y1})X(k_{x2}, k_{y2})|\rangle^2 \langle|{X(k_{x3}, k_{y3})
    \rangle|^2}}


where :math:`B`, the bispectrum is given by

.. math::
    B(k_{x1}, k_{y1},k_{x2}, k_{y2}) = \langle X(k_{x1}, k_{y1})X(k_{x2}, k_{y2})
    \overline{X(k_{x3}, k_{y3})}\rangle


is the bispectrum, with :math:`k_{x3} = k_{x1} + k_{x2}`, :math:`k_{y3} = k_{y1} + k_{y2}`,
:math:`\overline{X}` denoted a complex conjugate and :math:`\langle\rangle` denoting
an average over many realisations. If the modes at :math:`(k_{x1},k_{y1})`,
:math:`(k_{x2},k_{y2})` and :math:`(k_{x3},k_{y3})` are not coupled then that
complex triple product will have random phases so will average to zero over many
realisations, but if there is strong coupling then this average will have a
non-zero value. This can then be normalised to the total amplitude of the fluctuations
such that :math:`b^2` runs between 0 (no phase coupling) and 1 (entirely phase
coupled). It should be noted that bicoherence cannot determine the direction of energy
transfer but only if modes are coupled.


The Bicoherence of :math:`\phi` can be calculated from a nonlinear simulation output
with the following


.. code:: python

    from pyrokinetics import Pyro, template_dir
    from pyrokinetics.diagnostics import Diagnostics

    # Set up simulation
    data_dir = template_dir / "CGYRO_nonlinear"
    file_name = f"{data_dir}/input.cgyro"

    # Load phi data
    pyro = Pyro(gk_file=file_name)
    pyro.load_gk_output()
    data = pyro.gk_output.data

    # Need dimensions (kx, ky, time)
    phi = (
        data["phi"]
        .sel(theta=0.0, method="nearest")
        .pint.dequantify()
    )

    # Set up diagnostic and calculate bicoherence
    diagnostics  = Diagnostics(pyro)
    data = diagnostics.bicoherence(phi)
    
    # Read in bicoherence and phase data
    bicoherence = data["bicoherence"]
    phase = data["phase"]
    
    # Filter data where bicoherence is strong
    threshold= 0.5
    phase = phase.where(bicoherence > threshold, np.nan)
    bicoherence = bicoherence.where(bicoherence > threshold, np.nan)


One can also calculate the cross-bicoherence between 3 different fields
like for :math:`\delta \phi`, :math:`\delta n_e`, :math:`\delta v_e` by doing


.. code:: python

    from pyrokinetics import Pyro, template_dir
    from pyrokinetics.diagnostics import Diagnostics

    # Set up simulation
    data_dir = template_dir / "CGYRO_nonlinear"
    file_name = f"{data_dir}/input.cgyro"

    # Load data from 3 different fields
    pyro = Pyro(gk_file=file_name)
    pyro.load_gk_output()

    # Ensure dimensions are (kx, ky, time)
    phi = (
        data["phi"]
        .sel(theta=0.0, method="nearest")
        .pint.dequantify()
    )
    density = (
        data["density"]
        .sel(theta=0.0, method="nearest")
	.sel(species="electron")
        .pint.dequantify()
    )
    velocity = (
        data["velcoity"]
        .sel(theta=0.0, method="nearest")
	.sel(species="electron")
        .pint.dequantify()
    )
    
    # Set up diagnostic and calculate cross-bicoherence
    diagnostics  = Diagnostics(pyro)
    data = diagnostics.cross_bicoherence(phi, density, velocity)



If a signal is not stationary (like during the linear phase of
a nonlinear simulation), it is possible to examine the bicoherence
of the phase of the fluctuations :math:`\hat{X} = X / |X|`  by setting
`stationary=False` as a kwarg.
