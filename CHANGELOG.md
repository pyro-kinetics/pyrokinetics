
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
  
## [Unreleased] - 2021-01-26
 
### Added

### Changed
  - Changed Pyro kwarg from `gk_type` to  `gk_code`
 
### Fixed
  - `PyroScan.convert_gk_code` now sets `file_name` to the new code's default and
    keeps each run's `gk_file` in its run directory, so a following `write`
    writes the new code's input files. It also takes `template_file`.

## [0.0.1] - 2021-01-26  
 
### Added
 
### Changed
 
### Fixed

