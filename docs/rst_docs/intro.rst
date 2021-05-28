Introduction
============

This project aims to standardise gyrokinetic analysis. 

A general pyro object can be loaded either from simulation/experimental data or from an existing gyrokinetics file. 

In general pyrokinetics can do the following

-  Read data in from
  -  Gyrokinetic input files
  -  Simulations outputs
-  Write input files for various GK codes#.  WIP: Generate N-D pyro object for scans
-  WIP: Read in gyrokinetic outputs
-  WIP: Standardise analysis of gk outputs

At a minimum pyrokinetics needs the local geometry and species data. Example scripts can be found in the examples folder

Installation
------------

Install pyrokinetics with pip as follows:

.. code-block:: bash
		
  $ pip install --user pyrokinetics

Otherwise, for the latest version install directly with:

.. code-block:: bash
   
  $ git clone https://github.com/bpatel2107/pyrokinetics.git
  $ cd pyrokinetics
  $ python setup.py install --user


