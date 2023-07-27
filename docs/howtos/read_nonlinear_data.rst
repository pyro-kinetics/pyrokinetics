====================================================
Reading and plotting nonlinear outputs
====================================================

This is a step-by-step guide on how to use ``pyrokinetics`` to read the ouput of a nonlinear gyrokinetic simulation and create some simple plots. In this guide, we will use the example nonlinear output from CGYRO. The same ideas apply for  any gyrokinetic codes supported in ``pyrokinetics``,

Let's first import ``pyrokinetics`` and define our nonlinear input file.  

.. code-block:: python

   >>> from pyrokinetics import Pyro, template_dir
   >>> gk_file = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"

The gyrokinetic file ``input.cgyro`` is our input file template where we set all the extra flags that have been used in the simulation.  

We can then read the nonlinear simulation into a dataset using ``pyrokinetics``.

.. code-block:: python 

   >>> pyro.load_gk_output(load_moments=True, load_fluxes=True, load_fields=True)
   >>> data = pyro.gk_output.data

Here, ``pyro`` initialises the utility base dataclass ``GKOutputArgs`` which is used to pass quantities to ``GKOutput``. Derived classes include ``Coords``, ``Fields``, ``Fluxes``, etc. This clas contains features such as automatic unit conversion and a dict-like interface to quantities. Derived classes should define an ``InitVar[Tuple[str, ...]]`` called ``dims``, which sets the dimensionality of each quantity, e.g. ``("kx", "ky", "time")``.

It is helpful to view the  different data stored in the ``pyro`` object along with the dimensions of each quantity. 

.. code-block:: python 

   >>> print(pyro.gk_output)

Let's suppose we want to plot the time evolution of the total heat and particle fluxes from our nonlinear simulation. First, we need to extract the relevant quantities from the dataset. After examining the data set, we see that the heat and particle fluxes are given as functions of ``("field", "species", "ky", "time")``. In order to plot the total fluxes as a function of time we sum over the other dimensions. 

.. code-block:: python

   >>> hflux = data['heat'].pint.dequantify().sum(dim='ky').sum(dim='field').sum(dim='species') 
   >>> pflux = data['particle'].pint.dequantify().sum(dim='ky').sum(dim='field').sum(dim='species')  

We can then plot the data

.. code-block:: python 
   
   >>> hflux.plot()
   >>> pflux.plot()
   >>> plt.show()


