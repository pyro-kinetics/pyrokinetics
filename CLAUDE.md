# CLAUDE.md

Guidance for AI assistants working in this repository. For a description of the
code structure, read [`docs/architecture.md`](docs/architecture.md) first.

## Project

Pyrokinetics is a Python library (`src/pyrokinetics`) that reads, writes, converts
and analyses input/output files of local gyrokinetic (GK) codes (GS2, CGYRO, GENE,
STELLA, TGLF, GKW, GX, NEO), and derives local GK parameters from global
equilibrium and kinetic-profile files (GEQDSK, TRANSP, JETTO, SCENE, GACODE,
IMAS, ELITEINP, ITERDB, pFile). All quantities carry `pint` units tied to a
per-simulation normalisation system.

## Commands

```bash
pip install -e .[tests,docs,linting]    # dev install (Python >= 3.10)

pytest tests                             # full suite
pytest tests/gk_code/test_gk_input_gs2.py -k <name>   # targeted run
pytest --cov=pyrokinetics tests          # with coverage (as CI)

black src tests                          # formatting (CI auto-commits black output)
isort src tests docs/examples            # import order, profile=black
flake8 src tests                         # max-line-length 160, see .flake8

make -C docs html                        # Sphinx docs, needs .[docs]
pyro convert <code> <file> -o <out>      # CLI; see `pyro --help`
```

CI (`.github/workflows/`) runs pytest on Python 3.10–3.12, black/isort and flake8.

## Layout

```
src/pyrokinetics/
  pyro.py            Pyro: top-level object, owns everything below
  pyroscan.py        PyroScan: N-D parameter scans over a Pyro
  gk_code/           GKInput/GKOutput base classes + one module per GK code
  equilibrium/       Equilibrium + readers (GEQDSK, TRANSP, IMAS, ...)
  kinetics/          Kinetics + readers (JETTO, SCENE, TRANSP, pFile, ...)
  local_geometry/    Miller, MillerTurnbull, MXH, FourierGENE, FourierCGYRO
  local_species.py   LocalSpecies (per-species local parameters)
  numerics.py        Numerics dataclass (grids, fields, time step)
  normalisation.py   SimulationNormalisation / ConventionNormalisation
  units.py           Custom pint registry `ureg`, PyroQuantity, unit splines
  file_utils.py      FileReader / ReadableFromFile registration machinery
  factory.py         Generic Factory
  plugins.py         Entry-point plugin registration
  diagnostics/       Post-processing (ballooning, saturation rules, bootstrap, ...)
  databases/         IMAS IDS and YAML export
  cli/               `pyro` command (convert, generate)
  templates/         Default input files per code + test data (outputs/ excluded from wheel)
tests/               Mirrors src layout; golden answers in tests/gk_code/golden_answers
docs/                Sphinx (rst + myst markdown)
```

## Conventions

- **Registration by subclassing.** New file readers subclass `FileReader` with class
  keywords, e.g. `class GKInputFOO(GKInput, FileReader, file_type="FOO", reads=GKInput)`.
  They must be imported in the package `__init__.py` to register. File type inference
  calls `verify_file_type` on every registered reader, so it must be fast and raise
  on mismatch.
- **Units everywhere.** Use `from pyrokinetics.units import ureg`. Quantities in
  `LocalGeometry`, `LocalSpecies`, `Numerics` and `GKOutput` carry simulation units
  (`lref_minor_radius`, `vref_nrl`, ...). Convert with `.to(norms.<code>)`, not by
  hand-written factors. `bref` is not a base dimension: convert B fields with
  `.to(norms.<code>.bref)`. Tests turn `pint.UnitStrippedWarning` into an error.
- **Code-specific logic stays in `gk_code/<code>.py`.** `Pyro`, `LocalGeometry`,
  `LocalSpecies` and `Numerics` are code-agnostic. A new GK code implements
  `read_from_file`/`read_str`/`read_dict`, `verify_file_type`, `write`,
  `get_local_geometry`, `get_local_species`, `get_numerics`, `set`,
  `_detect_normalisation`, and a `GKOutputReader<CODE>` if outputs are supported.
  Add a template to `templates/` and to `templates.py`.
- **Pyro contexts.** `Pyro` keeps one record per `gk_code`; setting `pyro.gk_code`
  switches context. Use `convert_gk_code`, `write_gk_file(gk_code=...)` and
  `update_gk_code` rather than mutating `gk_input.data` directly.
- **Style.** black + isort (profile black), NumPy-style docstrings, type hints on
  public functions, British spelling in identifiers (`normalisation`).
- **Tests.** Every reader/writer change needs a test in the matching `tests/`
  subdirectory. Round-trip conversions are covered by `tests/test_roundtrip.py`.
  Output readers compare against netCDF golden answers; regenerate these only
  when a change in output is intended, and state why in the PR.
- **Dependencies** are pinned tightly in `pyproject.toml` (numpy, xarray, pint,
  pint-xarray, idspy). Do not loosen pins without running the full suite.
- Record user-facing changes in `CHANGELOG.md`.
