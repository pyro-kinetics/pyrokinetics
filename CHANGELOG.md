
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added

### Changed
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
 
### Fixed
  - `PyroScan.load_gk_output` stacks runs whose grids differ (e.g. TGLF runs at
    different `NMODES`, different `theta` resolution, different `kx` values)
    on the sorted union of their coordinates, NaN where a run has no value,
    instead of failing or labelling every run with the last run's
    coordinates. `integrate_over_valid_range` integrates over such an axis,
    each run exactly over its own range. `SaturationRules` uses it, and no
    longer holds the Jacobian, k_perp and |B| constant beyond the base Pyro's
    theta range: integrals are restricted to that range, with a warning.

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

