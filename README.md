[![Documentation Status](https://readthedocs.org/projects/pyrokinetics/badge/?version=latest)](https://pyrokinetics.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/pyro-kinetics/pyrokinetics/workflows/tests/badge.svg?branch=unstable)](https://github.com/pyro-kinetics/pyrokinetics/actions?query=workflow%3Atests)
[![Available on pypi](https://img.shields.io/pypi/v/pyrokinetics.svg)](https://pypi.org/project/pyrokinetics/)
[![Formatted with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Code coverage](https://codecov.io/gh/pyro-kinetics/pyrokinetics/branch/unstable/graph/badge.svg)](https://codecov.io/gh/pyro-kinetics/pyrokinetics)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05866/status.svg)](https://doi.org/10.21105/joss.05866)


# Pyrokinetics

This project aims to standardise gyrokinetic analysis. 

A general pyro object can be loaded either from simulation/experimental data or from an existing gyrokinetics file. 

In general pyrokinetics can do the following

* Read data in from:
    * Gyrokinetic input files
    * Simulations outputs
* Write input files for various GK codes
* Generate N-D pyro object for scans
* Read in gyrokinetic outputs
* Standardise analysis of gk outputs

At a minimum pyrokinetics needs the local geometry and species data. Example scripts can be found in the examples folder

## Documentation

Documentation can be found at [readthedocs](https://pyrokinetics.readthedocs.io/en/latest/).

## Installation 

Pyrokinetics requires a minimum Python version of 3.9. It may be necessary to upgrade
`pip` to install:

```bash
$ pip install --upgrade pip
```

To install the latest release:

```bash
$ pip install pyrokinetics
```

Otherwise, to install from source:

```bash 
$ git clone https://github.com/pyro-kinetics/pyrokinetics.git
$ cd pyrokinetics
$ pip install .
```

If you are planning on developing pyrokinetics use the following instead to install:

```bash 
$ pip install -e .[docs,tests]
```

Note that currently the installation of pyrokinetics requires an available Fortran compiler

## Testing

To run the tests:

```bash
$ pip install -e .[tests]
$ pytest --cov .
```

## Basic Usage

The simplest action in Pyrokinetics is to convert a gyrokinetics input file for code
'X' into an equivalent input file for code 'Y'. The easiest way to achieve this is to
use a `Pyro` object, which manages the various other classes in the API. For example,
to convert a GS2 input file to a CGYRO input file:

```python
>>> from pyrokinetics import Pyro
>>> pyro = Pyro(gk_file="my_gs2_file.in") # file type is automatically inferred
>>> pyro.write_gk_file("input.cgyro", gk_code="CGYRO")
```

There are many other features in Pyrokinetics, such as methods for building gyrokinetics
input files using global plasma equilibria and/or kinetics profiles. There are also
methods for analysing and comparing the results from gyrokinetics code runs. Please
[read the docs](https://pyrokinetics.readthedocs.io/en/latest/#) for more information.

## Command Line Interface

After installing, simple pyrokinetics operations can be performed on the command line
using either of the following methods:

```bash
$ python3 -m pyrokinetics {args...}
$ pyro {args...}
```

For example, to convert a GS2 input file to CGYRO:

```bash
$ pyro convert CGYRO "my_gs2_file.in" -o "input.cgyro"
```

You can get help on how to use the command line interface or any of its subcommands
by providing `-h` or `--help`:

```bash
$ pyro --help
$ pyro convert --help
```

## Docker

Pyrokinetics provides a `Dockerfile` from which you can build and run Docker containers.
To do so, you must have [Docker](https://docs.docker.com/engine/install/) installed on
your system. To build a local container:

```bash
$ docker build . -t pyrokinetics
```

It can then be run using:

```bash
$ docker run -it --rm -v ./path/to/local:/mymount pyrokinetics
```

where:

- `-it` runs an interactive shell.
- `--rm` deletes the Docker instance after use.
- `-v ./path/to/local:/mymount` mounts the local directory `./path/to/local` to the
  directory `/mymount` within the Docker container.

The container runs an IPython interpreter, with Pyrokinetics already installed. Note
that you will need to `import` Pyrokinetics before it can be used.

## Code structure 

Pyro object comprised of 

* Equilibrium
   * LocalGeometry
      * Miller
      * Fourier (to be added)
* Kinetics
   * LocalSpecies 
* Numerics
* GKCodes
* GKOutput
   * For nonlinear simulations
      * Fields (field, kx, ky, theta, time)
      * Fluxes (field, species, moment, ky, theta, time)
   * For linear simulations
      * Fields (field, kx, ky, theta, time)
      * Fluxes (field, species, moment, ky, theta, time)
      * Eigenfunctions (field, ky, theta, time)
      * Eigenvalues - growth rate and mode freq (ky, time)

There also exists the PyroScan object which allows you to make a N-D parameter scan of Pyro objects


## Supports sources of Equilibrium data
pyrokinetics currently supports
* [GEQDSK](https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf)
* [TRANSP](https://w3.pppl.gov/~pshare/help/body_transp_hlp.html#outfile56.html)
* [GACODE](https://gafusion.github.io/doc/input_gacode.html)
* [IMAS](https://conferences.iaea.org/event/251/contributions/20713/attachments/11191/16492/IMAS%20Tutorial%20-%20Pinches.pdf)

Future formats to be added are
* CHEASE


## Supported sources of Kinetic data

Sources of kinetic profile data currently supported are
* SCENE
* JETTO
* TRANSP
* GACODE
* PFILE
* IMAS

Future codes to be add 
* SimDB
* OMFIT

## Supported GK codes

The following gk codes are supported in pyrokinetics

* CGYRO
* GS2
* GENE
* TGLF
* GKW
* STELLA

Codes to be added in the future
* GX

## Note on units

The pyro object uses standardised reference values to normalise the results. It will automatically handle converting to a GK codes standard units.

Note any scans/parameter changes made will be in standard pyro units so please account for this.

Reference values
- $T_{ref} = T_e$ Electron temperature at flux surface
- $n_{ref} = n_e$ Electron density at flux surface
- $m_{ref} = m_D$ Deuterium mass
- $v_{ref} = c_s = \sqrt{T_e/m_D}$ Ion sound speed at flux surface
- $B_{ref} = B_0$ Toroidal field at centre of the flux surface
- $L_{ref} = a$ Minor radius of the last closed flux surface
- $t_{ref} = a/c_s$ Ion sound time at flux surface
- $\rho_{ref} = \frac{c_s}{eB_0/m_D}$ Ion Larmor radius at flux surface

It is possible to change the reference units but proceed with caution
  
## Used By

This project is used by the following institutions

- UKAEA
- University of York


Copyright owned by UKAEA. Pyrokinetics is licensed under LGPL-3.0, and is free to use, modify, and distribute.
  
