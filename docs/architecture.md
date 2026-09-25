# Architecture

This document describes how Pyrokinetics is structured and how its main pieces
interact. It is aimed at contributors; user-facing material lives in
`docs/user_guide`.

## Purpose and data flow

Pyrokinetics maps between three representations of a local gyrokinetic (GK)
problem:

1. **Global data** — a Grad–Shafranov equilibrium (`Equilibrium`) and radial
   kinetic profiles (`Kinetics`) read from integrated-modelling or reconstruction
   outputs.
2. **Code-agnostic local data** — `LocalGeometry`, `LocalSpecies` and `Numerics`,
   expressed in physical or normalised units.
3. **Code-specific data** — a `GKInput` (the input file of one GK code) and a
   `GKOutput` (its results, as an `xarray.Dataset`).

```
 GEQDSK/TRANSP/...           JETTO/SCENE/TRANSP/...
        │                              │
  Equilibrium ──(psi_n)──┐    ┌──(psi_n)── Kinetics
        │                ▼    ▼
        │         LocalGeometry  LocalSpecies  Numerics
        │                ▲    ▲         ▲
        │                │    │         │   get_*  (read)
        │                └── GKInput ───┘   set    (write)
        │                      │
        │            GS2/CGYRO/GENE/... input file
        │
        └──> SimulationNormalisation (reference values: lref, bref, nref, tref, ...)

 GK output files ──> GKOutputReader<CODE> ──> GKOutput (xarray, units) ──> diagnostics
```

Conversion from code A to code B is: `GKInputA.get_local_geometry/species/numerics()`
→ code-agnostic objects → `GKInputB.set(...)` on B's template → `GKInputB.write()`.
No code-to-code mapping is written directly.

## Top-level objects

### `Pyro` (`src/pyrokinetics/pyro.py`)

`Pyro` is the user entry point. It owns:

- `eq` (`Equilibrium`) and `kinetics` (`Kinetics`), loaded by
  `load_global_eq` / `load_global_kinetics` or constructor arguments.
- `norms`, a `SimulationNormalisation` unique to the instance (named after the
  input file via `_unique_name`).
- A set of **GK contexts**, one per `gk_code`. Each context stores
  `gk_input`, `gk_file`, `local_geometry`, `local_species`, `numerics`,
  `gk_output` and `gk_output_file` in `_<name>_record` dictionaries keyed by
  code name. The public attributes are properties that index these records with
  the current `gk_code`.

Context rules (`_switch_gk_context`):

- Setting `pyro.gk_code = "CGYRO"` enters the CGYRO context. If it does not yet
  exist, the CGYRO template is read and the current local geometry, species and
  numerics are copied into it.
- `convert_gk_code(code)` forces an overwrite of that context from the current one.
- `write_gk_file(path, gk_code=...)` syncs the target context via
  `update_gk_code` and writes it, without switching the current context.
- `update_gk_code()` pushes edits made to `local_geometry`/`local_species`/
  `numerics` back into `gk_input` through `GKInput.set`.

Local parameters are built from global data with `load_local_geometry(psi_n)`,
`load_local_species(psi_n)` or `load_local(psi_n)`. `load_local_geometry` calls
`LocalGeometry.from_global_eq`, which fits the chosen shaping parametrisation to
the flux surface extracted from the equilibrium. `switch_local_geometry` refits
an existing geometry to a different parametrisation.

Consistency helpers: `enforce_consistent_beta_prime` and
`enforce_consistent_pvg` recompute derived quantities from species gradients.

### `PyroScan` (`src/pyrokinetics/pyroscan.py`)

Builds an N-dimensional scan from a template `Pyro` and a
`{parameter: values}` dictionary. `parameter_map` maps short names
(`ky`, `kappa`, `electron_temp_gradient`, ...) to attribute paths inside
`Pyro`; `parameter_func` allows derived updates (e.g. keeping `beta_prime`
consistent). It writes one input file per point into a directory tree, records
itself in a JSON file (`pyroscan.json`) and can collect outputs into a
`PyroScanGKOutput` dataset with scan coordinates.

#### Loading a scan's output

`load_gk_output` loads every run and stacks each quantity along the scan
dimensions. Rules to keep when changing it:

- **Time reduction only converges a quantity onto one value.** Linear growth
  rate and frequency follow `linear_time_mode` (`"average"` over the last part
  of the run set by `linear_time_range`, default; `"last"`; or `"trace"`, the
  full time series). Linear fields and eigenfunctions take the last time;
  nonlinear fluxes are averaged over `tolerance_time_range`. `reduce_time`'s
  modes are listed in `VALID_TIME_MODES`; reuse `"trace"` for "keep the time
  dimension" rather than adding a new mode.
- **Never time-average a complex field.** The phase of each Fourier coefficient
  keeps moving in a nonlinear run, so its mean cancels. Nonlinear fields are
  kept complex with their `time` dimension (`nonlinear_fields="trace"`,
  the default), or reduced to `|field|**2` averaged in time
  (`"amplitude_squared"`). `|field|**2` is taken before any `kx`/`ky` sum.
- **Fields and eigenfunctions are reduced identically.** `phi`, `apar`, `bpar`
  and `eigenfunctions` go through the same `select_kx_ky_time` call: `ky` is
  summed when `sum_ky` (default, shared with fluxes), `kx` when `sum_kx`
  (default off); otherwise the dimension is kept. Never select a single
  `ky[0]` or smallest `|kx|` implicitly. Consumers that need one `kx` select it
  themselves, as `SaturationRules` does. A scanned parameter that is also a
  dimension of each run (e.g. `ky`) is squeezed out, since its value is already
  the scan coordinate.
- Not every linear output is electromagnetic. Test `apar`/`bpar` handling on
  the `STELLA_linear` and `GX_linear` outputs in `templates/outputs/` (GX's has
  more than one `ky`).

## File reading infrastructure

### `Factory` (`factory.py`)

A mapping from string keys to classes, constrained to a base class. Used for
`local_geometry_factory` and, via `FileReaderFactory`, for all file readers.

### `FileReader` / `ReadableFromFile` (`file_utils.py`)

- `ReadableFromFile` is mixed into data classes (`GKInput`, `GKOutput`,
  `Equilibrium`, `Kinetics`). `__init_subclass__` gives each such class its own
  `_factory`, and the class gains `from_file(path, file_type=None)` and
  `supported_file_types()`.
- `FileReader` subclasses register themselves at class-creation time:

  ```python
  class EquilibriumReaderGEQDSK(FileReader, file_type="GEQDSK", reads=Equilibrium):
      def read_from_file(self, filename, **kwargs) -> Equilibrium: ...
      def verify_file_type(self, filename) -> None: ...
  ```

- When `file_type` is omitted, `FileReaderFactory._infer_file_type` instantiates
  every registered reader and calls `verify_file_type`; the first that does not
  raise wins. If all fail, a `FileInferenceException` lists every error.
  `verify_file_type` should therefore be cheap and strict.
- Registration is triggered by importing the reader module, which each
  subpackage `__init__.py` does.

### Plugins (`plugins.py`)

`register_file_reader_plugins(group, Readable)` loads entry points from groups
`pyrokinetics.gk_input`, `pyrokinetics.gk_output`, `pyrokinetics.equilibrium`
and `pyrokinetics.kinetics`. Loading a plugin class registers it through the
same `FileReader` mechanism. The entry-point name must match the `file_type`.
`pyrokinetics-plugin-examples` is a test dependency exercising this path.

## Subpackages

### `gk_code/`

| Module | Contents |
| --- | --- |
| `gk_input.py` | `GKInput` base: holds `data` (usually an `f90nml.Namelist` or dict), `read_*`, `write`, `set`, `get_local_geometry`, `get_local_species`, `get_numerics`, normalisation detection. |
| `gk_output.py` | `GKOutput` (`DatasetWrapper`) and argument containers `Coords`, `Fields`, `Fluxes`, `Moments`, `Eigenvalues`, `Eigenfunctions`, each defining dimensions and units. Computes eigenvalues/eigenfunctions from fields when a code does not provide them. |
| `gs2.py`, `cgyro.py`, `gene.py`, `stella.py`, `tglf.py`, `gkw.py`, `gx.py`, `neo.py` | `GKInput<CODE>` and `GKOutputReader<CODE>` for each code. |
| `ids.py` | `GKOutputReaderIDS`, reads IMAS gyrokinetics IDS. |

A `GKInput<CODE>` implements:

- `read_from_file` / `read_str` / `read_dict` and `verify_file_type`.
- `get_local_geometry`, `get_local_species`, `get_numerics`: translate
  code parameters into code-agnostic objects with that code's normalisation.
- `set(local_geometry, local_species, numerics, local_norm, template_file, code_normalisation)`:
  the inverse mapping, writing into `self.data`.
- `_detect_normalisation`: determines which reference quantities the input file
  uses (e.g. GS2 `tref`/`nref` species, GENE `minor_r`/`major_R`), and
  `_set_up_normalisation` registers the resulting convention.
- `is_nonlinear`, `add_flags`, `get_reference_values` where applicable.

A `GKOutputReader<CODE>` implements `read_from_file` returning a `GKOutput`,
`verify_file_type`, and `infer_path_from_input_file` so outputs can be found
from the input path. Private helpers `_get_raw_data`, `_get_coords`,
`_get_fields`, `_get_fluxes`, `_get_moments`, `_get_eigenvalues` follow a common
pattern across codes.

### `local_geometry/`

`LocalGeometry` is the base; subclasses implement a flux-surface shape
parametrisation:

| Key in `local_geometry_factory` | Class | Parametrisation |
| --- | --- | --- |
| `Miller` | `LocalGeometryMiller` | κ, δ, s_κ, s_δ, Shafranov shift |
| `MillerTurnbull` | `LocalGeometryMillerTurnbull` | Miller plus squareness ζ |
| `MXH` | `LocalGeometryMXH` | Miller extended harmonic moments |
| `FourierGENE` | `LocalGeometryFourierGENE` | Fourier series in (R, Z) about the centroid, GENE convention |
| `FourierCGYRO` | `LocalGeometryFourierCGYRO` | Fourier series, CGYRO convention |

Subclasses provide `_set_shape_coefficients` (least-squares fit to equilibrium
R, Z and B_pol), `get_flux_surface`, and analytic `get_RZ_derivatives` /
second derivatives. The base class uses these to compute B_pol, `bunit_over_b0`,
`f_psi`, `s_hat`, surface areas and volumes. `from_local_geometry` converts
between parametrisations by fitting one to the surface generated by the other.
`metric.py` (`MetricTerms`) computes field-aligned metric coefficients for
diagnostics.

### `equilibrium/`

`Equilibrium` (`DatasetWrapper`) holds R–Z grids of ψ and 1D profiles in ψ
(F, p, q, ...), converted to COCOS 11 on construction. `FluxSurface` extracts a
single contour with `contourpy` and provides the R, Z, B_pol data used for local
fits. Readers: `GEQDSK`, `TRANSP`, `GACODE`, `IMAS`, `ELITEINP`, and
`Pyrokinetics` (netCDF round trip).

### `kinetics/`

`Kinetics` holds a `CleverDict` of `Species` (`species.py`), each storing
density, temperature, rotation and their radial interpolants as unit-aware
splines (`units.UnitSpline`). Readers: `JETTO`, `SCENE`, `TRANSP`, `pFile`,
`GACODE`, `IMAS`, `ELITEINP`, `ITERDB`.

### `local_species.py`, `numerics.py`

- `LocalSpecies` (`CleverDict`) stores per-species `z`, `mass`, `dens`, `temp`,
  `nu`, gradient scale lengths, rotation and derived totals (`pressure`,
  `inverse_lp`, `zeff`). `from_kinetics` evaluates `Species` at `psi_n`.
  Quasineutrality is enforced on write unless disabled.
- `Numerics` is a dataclass of grid sizes, time step, field switches
  (`phi`, `apar`, `bpar`), `nonlinear`, `beta`, `gamma_exb`, etc.

### Units and normalisation (`units.py`, `normalisation.py`)

- `units.ureg` is a custom `pint` registry (`PyroUnitRegistry`) that defines
  simulation reference units as extra dimensions (`[lref]`, `[vref]`, `[bref]`,
  `[beta_ref]`, ...) with concrete variants such as `lref_minor_radius`,
  `lref_major_radius`, `vref_nrl`, `vref_most_probable`, `bref_B0`,
  `bref_Bunit`.
- `PyroQuantity.to(convention)` converts any quantity to a target convention by
  substituting each reference unit.
- `SimulationNormalisation` (one per `Pyro`) holds a `ConventionNormalisation`
  per code (`norms.gs2`, `norms.cgyro`, `norms.gene`, ..., `norms.pyrokinetics`,
  `norms.imas`). Calling `set_bref`, `set_lref`, `set_kinetic_references`,
  `set_all_references` converts simulation units into physical units specific to
  that run (e.g. `lref_minor_radius_<run>`), enabling conversion to SI.
- Unit-aware interpolators (`UnitSpline`, `UnitSpline2D`,
  `UnitCloughTocher2DInterpolator`) wrap SciPy.

`DatasetWrapper` (`dataset_wrapper.py`) wraps an `xarray.Dataset` with
`pint-xarray` quantification, forwarding attribute access and providing
`to_netcdf`/`from_netcdf` that preserve units.

### `diagnostics/`

Post-processing that operates on a `Pyro` or `GKOutput`:

- `Diagnostics` (`__init__.py`): GS2-style geometry terms, ideal-ballooning
  solver (`ideal_ballooning_solver`, `gamma_ball_full`), bicoherence and
  cross-bicoherence.
- `field_line.py`: `FieldLine` — field-line following, Poincaré maps, radial
  diffusion coefficient, parallel correlation length, linear tearing parameter.
- `saturation_rules.py`: quasilinear saturation models applied to a `PyroScan`.
- `convergence.py`: linear convergence checks.
- `neoclassical.py`: bootstrap current models (`Sauter1999`, `Redl2021`).
- `synthetic_highk_dbs.py`: synthetic high-k / Doppler back-scattering diagnostic.

### `databases/`

- `imas.py`: `pyro_to_ids` / `ids_to_pyro`, mapping a `Pyro` and its output to
  the IMAS gyrokinetics IDS via `idspy`.
- `yaml.py`: `SimDBYaml`, a YAML summary for simulation databases.

### `cli/`

`pyro` console script (`cli.entrypoint`), built with `argparse` subparsers.
Each subcommand module exposes `description`, `add_arguments(parser)` and
`main(args)`:

- `convert`: convert a GK input file to another code, optionally rebuilding
  geometry/species from an equilibrium and kinetics file at `--psi`.
- `generate`: create a GK input from global data.

### `templates/`

Default input files per code (`templates.gk_templates`), sample equilibrium and
kinetics files (`eq_templates`, `kinetics_templates`) and reference outputs in
`templates/outputs/` used by tests. `outputs/` is excluded from the wheel.

## Tests

`tests/` mirrors the package layout. Notable patterns:

- `conftest.py` provides shared fixtures (`generate_miller`, `array_similar`)
  and monkeypatches equilibrium/kinetics reading with `functools.cache` to avoid
  re-reading sample files.
- `tests/gk_code/test_gk_input_<code>.py` checks reading, writing and
  `get_*`/`set` round trips per code.
- `tests/gk_code/test_gk_output_reader_<code>.py` compares readers against
  netCDF golden answers in `tests/gk_code/golden_answers/`.
- `tests/test_roundtrip.py` converts between every pair of codes and checks
  that local parameters survive.
- `pint.UnitStrippedWarning` is promoted to an error in `pyproject.toml`, so any
  code path that silently drops units fails the suite.

## Extending

| Task | Where |
| --- | --- |
| New GK code | `gk_code/<code>.py` with `GKInput<CODE>` and optionally `GKOutputReader<CODE>`; import in `gk_code/__init__.py`; template in `templates/` and `templates.py`; normalisation convention in `normalisation.py`; tests in `tests/gk_code/`. |
| New equilibrium or kinetics format | Reader in `equilibrium/` or `kinetics/`, imported in the subpackage `__init__.py`, test file and sample in `templates/`. External packages can use the entry-point groups instead. |
| New geometry parametrisation | Subclass `LocalGeometry`, implement fit and derivatives, register in `local_geometry/__init__.py`, add support in each `GKInput.set`/`get_local_geometry` that can use it. |
| New scan parameter | Add to `PyroScan.parameter_map` (and `parameter_func` if derived quantities must be updated). |
| New diagnostic | Module in `diagnostics/` operating on `Pyro`/`GKOutput` with units preserved. |
