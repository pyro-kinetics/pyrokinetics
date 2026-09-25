
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added

### Changed
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
  - `PyroScan.load_gk_output`: `phi`, `apar`, `bpar` and `eigenfunctions` keep
    their `kx` and `ky` dimensions instead of being reduced to `ky[0]` and the
    smallest `|kx|`; select one with the new `field_kx` / `field_ky`.
    Nonlinear fields keep their `time` dimension instead of being averaged.
  - `PyroScan.load_gk_output` raises if runs give a quantity different shapes,
    and drops a coordinate that differs between runs instead of using the last
    run's values.
 
### Fixed

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

