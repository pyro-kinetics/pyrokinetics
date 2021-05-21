
# Pyrokinetics

This project aims to standardise gyrokinetic analysis. 

A general pyro object can be loaded either from simulation/experimental data or from an existing gyrokinetics file. 

In general pyrokinetics can do the following

* Read data in from:
    * Gyrokinetic input files
    * Simulations outputs
* Write input files for various GK codes
* WIP: Generate N-D pyro object for scans
* WIP: Read in gyrokinetic outputs
* WIP: Standardise analysis of gk outputs

At a minimum pyrokinetics needs equilibrium and local species data 


## Supported sources of kinetic data

pyrokinetics currently supports
* SCENE
* JETTO
* TRANSP

Future codes to be add 
* SimDB
* OMFIT

## Supports sources of Equilibrium data
pyrokinetics currently supports
* GEQDSK

Future formats to be added are
* CHEASE



## Supported GK code

Currently the following gk codes are supported in pyrokinetics

* CGYRO
* GS2

Codes to be added in the future
* GENE
* TGLF


## Code structure 

Pyro object comprised of 

* Equilibrium
* Kinetic profiles
* Numerical set-up
## Installation 

Install pyrokinetics with 

```bash 
$ git clone https://github.com/bpatel2107/pyrokinetics.git
$ cd pyrokinetics
$ python setup.py install --user
```

## Documentation

[Documentation](https://linktodocumentation)

  
## Used By

This project is used by the following institutions

- CCFE
- University of York

  
