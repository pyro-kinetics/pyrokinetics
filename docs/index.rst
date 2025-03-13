Introduction
============

This project aims to standardise gyrokinetic analysis by providing a single
interface for reading and writing input and output files from different
gyrokinetic codes, normalising to a common standard, and performing standard
analysis methods.

A general ``Pyro`` object can be loaded either from simulation/experimental data or
from an existing gyrokinetics file.

Currently pyrokinetics can do the following:

- Read data in from:

  - Gyrokinetic input files
  - Integrated modelling/Global Equilibrium simulation output

- Write input files for various gyrokinetics codes
- Generate N-D ``Pyro`` object for scans
- Read in gyrokinetic outputs
- Standardise analysis of gyrokinetics outputs

Future development includes:

- Submit gyrokinetics simulations to cluster
- Integrate with GKDB to store/catalog GK runs

At a minimum pyrokinetics needs the local geometry and species data. This can be
taken from a GK input file. At the moment the following local gyrokinetic codes
are supported:

- `GS2 <https://gyrokinetics.gitlab.io/gs2/>`_
- `CGYRO <http://gafusion.github.io/doc/cgyro.html>`_
- `GENE <http://www.genecode.org>`_
- `STELLA <https://stellagk.github.io/stella/index.html>`_
- `TGLF <http://gafusion.github.io/doc/tglf.html>`_
- `GKW <http://https://bitbucket.org/gkw/gkw/src>`_

It is also possible to load in global data from the following codes, from which
local parameters can be calculated.

- `TRANSP <https://transp.pppl.gov>`_
- JETTO
- SCENE
- `IMAS <https://conferences.iaea.org/event/251/contributions/20713/attachments/11191/16492/IMAS%20Tutorial%20-%20Pinches.pdf>`_


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Getting Started <user_guide/getting_started>
   User Guide <user_guide/index>
   How-to Guides <howtos/index>
   Examples <examples/index>
   Tutorials <tutorials/index>

.. toctree::
   :maxdepth: 2
   :caption: Reference

   API Reference <api>

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   Contributing Guide <developer_guide/contributing_guide>
   Writing Documentation <developer_guide/writing_docs>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
