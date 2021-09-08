Introduction
============

This project aims to standardise gyrokinetic analysis. 

A general pyro object can be loaded either from simulation/experimental data or from an existing gyrokinetics file. 

Currently pyrokinetics can do the following

-  Read data in from

  -  Gyrokinetic input files

  -  Integrated modelling/Global Equilibrium simulation output

-  Write input files for various GK codes

-  Generate N-D pyro object for scans

-  Read in gyrokinetic outputs

-  Standardise analysis of gk outputs


Future development includes

-  Submit GK simulations to cluster

-  Integrate with GKDB to store/catalog GK runs


At a minimum pyrokinetics needs the local geometry and species data. This can be taken from a GK input file. At the moment the following local gyrokinetic codes are supported


-  GS2
-  CGYRO
-  GENE


It is also possible to load in global data from the following codes, from which local parameters can be calculated.


-  TRANSP
-  JETTO
-  SCENE


Example scripts can be found in the examples folder where GK/Transport code data is read in and GK outputs are read in.


Installation
------------


Install pyrokinetics with pip as follows:


.. code-block:: bash
		
  $ pip install --user pyrokinetics


Otherwise, for the latest version install directly with:


.. code-block:: bash
   
  $ git clone https://github.com/pyro-kinetics/pyrokinetics.git
  $ cd pyrokinetics
  $ python setup.py install --user


Structure
---------


The Pyro object is structured as follows


-  :ref:`sec-equilibrium`

  -  Accessed via ``pyro.eq``

  -  Represents full 2D equilibrium

  -  Only loaded when full equilibrium data is provided


*  :ref:`sec-local_geometry`

  *  Accessed via ``pyro.local_geometry``

  *  Represents local geometry of a flux surface

  *  Current supported LocalGeometry subclasses are:

    *  :ref:`sec-miller`


-  :ref:`sec-kinetics`

  -  Accessed via ``pyro.kinetics``

  -  Represents 1D profiles for each kinetic species

  -  Only loaded when full profile data is provided


*  :ref:`sec-local_species`

  *  Accessed via ``pyro.local_species``

  *  Contains local species parameters


*  :ref:`sec-numerics`

  *  Accessed via ``pyro.numerics``

  *  Sets up numerical grid and certain physics models


*  :ref:`sec-gk_code`

  *  Accessed via ``pyro.gk_code``

  *  Holds gyrokinetics input data and methods specific to each GK code

  *  Can be used to directly populate LocalGeometry and LocalSpecies

  *  Used to set Numerics

  *  Current supported GKCode subclasses are:

    *  :ref:`sec-gs2`
    *  :ref:`sec-cgyro`
    *  :ref:`sec-gene`


.. image:: figures/pyro_structure.png
  :width: 600


Analysis
-----------
Once you have a completed simulation you can read the output into a pyro object.

*  :ref:`sec-gk_output`

  *  Accessed via ``pyro.gk_output``

  *  Stores output from a GK simulation

