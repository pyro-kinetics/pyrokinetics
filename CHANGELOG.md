
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added

### Changed
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
 
### Fixed
  - `add_flags` for TGLF, GFTM, CGYRO and NEO matches keys case-insensitively, so e.g. `NBASIS_MAX` overwrites the existing `nbasis_max` instead of writing it twice. New keys use the code's stored case (lowercase for TGLF/GFTM, uppercase for CGYRO/NEO)

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

