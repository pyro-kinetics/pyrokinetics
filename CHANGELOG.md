
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
  - Nonlinear fields default to `|field|**2` averaged in time
    (`nonlinear_fields="amplitude_squared"`) instead of the time average of the
    complex field; `nonlinear_fields="time_resolved"` keeps the complex field in
    time.

### Fixed

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

