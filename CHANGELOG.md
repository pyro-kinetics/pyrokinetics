
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added

### Changed
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
  - `PyroScan.load_gk_output`: `phi`, `apar`, `bpar` and `eigenfunctions` are no
    longer reduced to `ky[0]` and the smallest `|kx|`. They follow `sum_ky`
    (default True, as for fluxes) and the new `sum_kx` (default False), and are
    otherwise kept on their `kx`/`ky` grid.
  - Nonlinear fields are no longer time-averaged as complex values (the mean of
    a Fourier coefficient with moving phase cancels). By default they keep their
    `time` dimension (`nonlinear_fields="time_resolved"`);
    `nonlinear_fields="amplitude_squared"` gives `|field|**2` averaged in time.

### Fixed

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

