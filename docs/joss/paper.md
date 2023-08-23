---
title: 'Pyrokinetics - A python library to standardise gyrokinetic analysis'
tags:
  - Python
  - gyrokinetics
  - turbulence
  - plasma
  - tokamak
  - fusion
authors:
  - name: Bhavin S. Patel
    orcid: 0000-0003-0121-1187
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Maurizio Giacomin
    orcid: 0000-0003-2821-2008
    affiliation: 2
  - name: David Dickinson
    orcid: 0000-0002-0868-211X
    affiliation: 2
  - name: Peter Hill
    orcid: 0000-0003-3092-1858
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Culham Centre for Fusion Energy, Abingdon OX14 3DB, UK
   index: 1
 - name: University of York, Heslington, Y010 5DD, UK 
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 03 August 2023
bibliography: paper.bib

---

# Summary

Fusion energy offers the potential for a near limitless source of low carbon energy and is often 
regarded as a solution for the worlds long term energy needs. To realise such a scenario requires
the design of high performance fusion reactors capable of containing the extreme conditions
necessary to enable fusion. Turbulence is typically the dominant source of transport in fusion
plasmas, accounting for the majority of the particle and heat loss. Gyrokinetic modelling aims to 
quantify the level of turbulent transport in fusion reactors and can be used to understand the 
major drivers of turbulence. The realisation of fusion critically depends on understanding how to
mitigate turbulent transport and thus requires high levels of confidence in the tools being
used. Many different gyrokinetic modelling codes are available for use and pyrokinetics aims to 
standardise the analysis of such simulations.

# Statement of need

[Pyrokinetics](https://github.com/pyro-kinetics/pyrokinetics) is a Python project (package: 
[`pyrokinetics`](https://pypi.org/project/pyrokinetics/))
that aims to simplify and standardise gyrokinetic analysis. A wide 
variety of different gyrokinetic solvers exist that utilise different input file formats and
use different normalisations for plasma parameters such as densities, temperatures, velocities,
and magnetic fields. To improve confidence in the predictions from gyrokinetic solvers it is often 
desirable to benchmark the results of one code against another. Pyrokinetics aims to make this
easier for researchers by acting as an interface between the different codes, automatically 
handling the conversion of physical input parameters between different normalisations
and file formats. Furthermore, gyrokinetics inputs can come from a
wide variety of modelling tools outside of gyrokinetics. Pyrokinetics interfaces with
these allowing for the easy generation of both linear and nonlinear gyrokinetic input files and 
has been designed to be extensible and easy to add new sources of data. 

The output of gyrokinetic codes is often multidimensional and each code stores this data in a
different format with different normalisations, and potentially across multiple files. Pyrokinetics will seamlessly read in all this data and
store it in a single object using an [`xarray`](https://pypi.org/project/xarray/) Dataset, automatically converting the outputs to a 
standard normalisation (using [`pint`](https://pypi.org/project/Pint/)), allowing for direct comparisons between codes. Furthermore, additional derived
outputs such as the linear growth rate of a turbulent instability, can be calculated using the exact
same method such that the modeller is confident that the output is consistent across codes.

Pyrokinetics was designed to be used by gyrokinetics modellers and has already been used in a 
number of scientific publications 
[@giacomin:2023a; @giacomin:2023b; @kennedy:2023]. Furthermore, the 
python interface opens up gyrokinetic analysis to the wide variety of python packages available, 
allowing for a range of analyses from simple parameter scans to using 
thousands of linear gyrokinetic runs to develop Gaussian process regression models of the
linear properties of electromagnetic turbulence [@hornsby:2023]. Pyrokinetics also maintains 
compatibility with IMAS, a community wide standard for data format [@imbeaux:2015], allowing for 
further interfacing with the whole fusion community and the potential to develop a global gyrokinetic
database.

We hope that pyrokinetics makes gyrokinetics modelling more accessible and will help to increase the
communities confidence in tools available.


# Acknowledgements

We acknowledge contributions from [PlasmaFAIR](https://plasmafair.github.io), EPSRC Grant EP/V051822/1

# References
