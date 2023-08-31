---
title: 'Pyrokinetics - A Python library to standardise gyrokinetic analysis'
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
  - name: Peter Hill
    orcid: 0000-0003-3092-1858
    affiliation: 2
  - name: Liam Pattinson
    orcid: 0000-0001-8604-6904
    affiliation: 2
  - name: Maurizio Giacomin
    orcid: 0000-0003-2821-2008
    affiliation: 2
  - name: Arkaprava Bokshi
    orcid: 0000-0001-7095-7172
    affiliation: 2
  - name: Daniel Kennedy
    orcid: 0000-0001-7666-782X
    affiliation: 1
  - name: Harry G. Dudding
    orcid: 0000-0002-3413-0861
    affiliation: 1
  - name: Jason. F. Parisi
    orcid: 0000-0002-8763-3016
    affiliation: 3
  - name: Tom F. Neiser
    orcid: 0000-0002-8763-3016
    affiliation: 4
  - name: Ajay C. Jayalekshmi
    orcid: 0000-0002-6447-581X
    affiliation: 5
  - name: David Dickinson
    orcid: 0000-0002-0868-211X
    affiliation: 2
affiliations:
 - name: Culham Centre for Fusion Energy, Abingdon OX14 3DB, UK
   index: 1
 - name: University of York, Heslington, Y010 5DD, UK 
   index: 2
 - name: Princeton Plasma Physics Laboratory, Princeton, NJ 08536, USA
   index: 3
 - name: General Atomics, San Diego, CA 92121, USA
   index: 4
 - name: University of Warwick, Warwick, CV4 7AL, UK
   index: 5
date: 31 August 2023
bibliography: paper.bib

---

# Summary

Fusion energy offers the potential for a near limitless source of low-carbon energy and is often 
regarded as a solution for the world's long-term energy needs. To realise such a scenario requires
the design of high-performance fusion reactors capable of maintaining the extreme conditions
necessary to enable fusion. Turbulence is typically the dominant source of transport in magnetically-confined fusion
plasmas, accounting for the majority of the particle and heat losses. Gyrokinetic modelling aims to 
quantify the level of turbulent transport encountered in fusion reactors and can be used to understand the 
major drivers of turbulence. The realisation of fusion critically depends on understanding how to
mitigate turbulent transport, and thus requires high levels of confidence in the predictive tools being
employed. Many different gyrokinetic modelling codes are available and Pyrokinetics aims to
standardise the analysis of such simulations.

# Statement of need

[Pyrokinetics](https://github.com/pyro-kinetics/pyrokinetics) is a Python project (package: 
[`pyrokinetics`](https://pypi.org/project/pyrokinetics/))
that aims to simplify and standardise gyrokinetic analysis. A wide 
variety of gyrokinetic solvers are used in practice, each utilising different input file formats and
normalisations for plasma parameters such as densities, temperatures, velocities,
and magnetic fields. To improve confidence in the predictions from gyrokinetic solvers it is often 
desirable to benchmark the results of one code against another. Pyrokinetics aims to make this
easier for researchers by acting as an interface between each code, automatically
handling the conversion of physical input parameters between different normalisations
and file formats. Furthermore, gyrokinetics inputs can come from a
wide variety of modelling tools outside gyrokinetics, such as TRANSP [@pankin:2004] and JETTO [@cenacchi:1988]. 
Pyrokinetics interfaces with these tools, allowing for the easy generation of both linear and nonlinear gyrokinetic 
input files, and has been designed to be extensible and simple to incorporate new sources of data. 

The output of gyrokinetic codes is often multidimensional, and each code stores this data in a
different format with different normalisations, potentially across multiple files. Pyrokinetics will seamlessly read in
all this data and store it in a single object using an [`xarray`](https://pypi.org/project/xarray/) Dataset,
automatically converting the outputs to a standard normalisation (using [`pint`](https://pypi.org/project/Pint/)),
permitting direct comparisons between codes. Furthermore, additional derived
outputs, such as the linear growth rate of a turbulent instability, can be calculated using the exact
same method, such that the modeller can be confident that the output is consistent across codes.

Pyrokinetics is designed to be used by gyrokinetics modellers and has already been used in several
scientific publications [@giacomin:2023a; @giacomin:2023b; @kennedy:2023]. Furthermore, the 
Python interface opens up gyrokinetic analysis to the wide variety of Python packages available,
allowing for a range of analyses from simple parameter scans to the use of
thousands of linear gyrokinetic runs to develop Gaussian process regression models of the
linear properties of electromagnetic turbulence [@hornsby:2023]. Pyrokinetics also maintains 
compatibility with IMAS, a standard data schema for magnetic confinement fusion [@imbeaux:2015], enabling
greater interoperability with the wider fusion community and the potential development of a global gyrokinetic
database.

With pyrokinetics we strive to make gyrokinetic modelling more accessible and to increase the
community's confidence in the tools available.


# Acknowledgements

We acknowledge contributions from [PlasmaFAIR](https://plasmafair.github.io), EPSRC Grant EP/V051822/1, U.S. DOE under grant DE-SC0018990, and used HPC resources funded by DOE-AC02-05CH11231 (NERSC).

# References
