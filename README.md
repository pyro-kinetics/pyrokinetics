
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
* WIP: Standardise analysis of gk outputs

At a minimum pyrokinetics needs the local geometry and species data. Example scripts can be found in the examples folder

## Documentation

[Documentation](https://pyrokinetics.readthedocs.io/en/latest/)

## Installation 

Install pyrokinetics with pip as follows

```bash
pip install --user pyrokinetics
```

Otherwise, for the latest version install directly with 

```bash 
$ git clone https://github.com/pyro-kinetics/pyrokinetics.git
$ cd pyrokinetics
$ python setup.py install --user
```

If you are planning on developing pyrokinetics use the following instead to install 
```bash 
$ python setup.py develop --user
```
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

Future formats to be added are
* CHEASE


## Supported sources of Kinetic data

Sources of kinetic profile data currently supported are
* SCENE
* JETTO
* TRANSP

Future codes to be add 
* SimDB
* OMFIT

## Supported GK codes

The following gk codes are supported in pyrokinetics

* CGYRO
* GS2
* GENE
* TGLF

Codes to be added in the future
* Stella
* GX

## Note on units

The pyro object uses standardised reference values to normalise the results. It will automatically handle converting to a GK codes standard units.

Note any scans/parameter changes made will be in standard pyro units so please account for this.

Reference values
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20T_%7Bref%7D%20%3D%20T_e" /> 
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20n_%7Bref%7D%20%3D%20n_e" />
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20m_%7Bref%7D%20%3D%20m_D" />
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20v_%7Bref%7D%20%3D%20c_s%20%3D%20%5Csqrt%7BT_e/m_D%7D" />
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20B_%7Bref%7D%20%3D%20B_0" />
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20L_%7Bref%7D%20%3D%20a_%7Bmin%7D" />
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20t_%7Bref%7D%20%3D%20a_%7Bmin%7D/c_s" />
- <img src="https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5CLARGE%20%5Crho_%7Bref%7D%20%3D%20%5Cfrac%7Bc_s%7D%7BeB_0/m_D%7D" />

It is possible to change the reference units but proceed with caution
  
## Used By

This project is used by the following institutions

- CCFE
- University of York

  
