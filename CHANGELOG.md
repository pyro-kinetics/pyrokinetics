
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added

### Changed
  - `add_flags` has one implementation per input format: `GKInput.add_flags` for grouped inputs and the new `GKInputFlat.add_flags` for flat `KEY = value` inputs. Per-code overrides are removed and `GKInput.add_flags` is no longer abstract
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
 
### Fixed
  - `add_flags` for TGLF, CGYRO and NEO matches keys case-insensitively, so e.g. `NBASIS_MAX` overwrites the existing TGLF `nbasis_max` instead of writing it twice. New keys take the code's stored case (`flag_key_case`: lowercase for TGLF, uppercase for CGYRO/NEO)
  - `add_flags` raises `TypeError` when flags do not match the input structure (a dict of groups for namelist/TOML codes, a flat dict for TGLF/CGYRO/NEO) instead of crashing or writing an invalid file

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

