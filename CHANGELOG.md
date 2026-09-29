
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added
  - `PyroHypercube`: non-gridded parameter sets (one value per sample), with a
    single `sample` output dimension, and `from_directory` to read existing run
    trees. Shares output loading with `PyroScan` through
    `PyroScan.output_layout`.

### Changed
  - `add_flags` has one implementation per input format: `GKInput.add_flags` for grouped inputs and the new `GKInputFlat.add_flags` for flat `KEY = value` inputs. Per-code overrides are removed and `GKInput.add_flags` is no longer abstract
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
  - `PyroScan.load_gk_output` uses one time setting for linear and nonlinear
    runs: `time_mode` (`"average"` default, `"last"`, `"trace"`), with
    `tolerance_time_range` (default 0.8) setting the averaging window. With
    `"average"`, fields and eigenfunctions are `|field|**2` averaged in time,
    stored as `phi_squared`, `apar_squared`, `bpar_squared` and
    `eigenfunctions_squared`, since a time mean of a complex field is
    meaningless. Linear growth rates and fluxes are now averaged rather than
    taken at the last time.
  - `phi`, `apar`, `bpar` and `eigenfunctions` are no longer reduced to `ky[0]`
    and the smallest `|kx|`; they keep `kx` and `ky` unless summed with
    `sum_kx` / `sum_ky`. `sum_ky` now defaults to False, for fluxes too.

### Fixed
  - `add_flags` for TGLF, CGYRO and NEO matches keys case-insensitively, so e.g. `NBASIS_MAX` overwrites the existing TGLF `nbasis_max` instead of writing it twice. New keys take the code's stored case (`flag_key_case`: lowercase for TGLF, uppercase for CGYRO/NEO)
  - `add_flags` raises `TypeError` when flags do not match the input structure (a dict of groups for namelist/TOML codes, a flat dict for TGLF/CGYRO/NEO) instead of crashing or writing an invalid file
  - `PyroScan.convert_gk_code` now sets `file_name` to the new code's default and
    keeps each run's `gk_file` in its run directory, so a following `write`
    writes the new code's input files. It also takes `template_file`.
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

