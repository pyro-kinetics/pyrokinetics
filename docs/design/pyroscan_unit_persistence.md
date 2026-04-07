# PyroScan unit persistence: problem and solution options

## The problem

When a PyroScan is created from a GENE file (which stores physical
normalisations), converted to TGLF, written to disk, and reloaded,
the normalisation data is lost because TGLF input files cannot store it.

Additionally, pint Quantities in the `parameter_dict` carry
**instance-specific** unit names like `vref_nrl_gene0042`.  After a
write/reload cycle, the new Pyro instance has a different name
(e.g. `pyroscan_base0000`), so the old unit names are unrecognised.

### Minimal reproduction

```python
from pyrokinetics import Pyro, PyroScan
import numpy as np

# 1. Load a GENE file that has physical reference values
pyro = Pyro(gk_file="input_wunits.gene")

# 2. Convert to TGLF (normalisations live in pyro.norms, not in the file)
pyro.gk_code = "TGLF"

# 3. Create a gamma_exb scan -- units are instance-specific
scan = PyroScan(pyro, {"gamma_exb": np.linspace(0, 0.5, 5)})
#   parameter_dict values have units like  vref_nrl_input_wunits0000 / lref_minor_radius_input_wunits0000

# 4. Write to disk
scan.write(file_name="input.tglf", base_directory="my_scan")
#   Writes: pyroscan.json, pyroscan_base.input (TGLF), per-run dirs

# 5. Reload
scan2 = PyroScan(pyroscan_json="my_scan/pyroscan.json", load_base_pyro=True)
#   New pyro has norms.name = "pyroscan_base0000"
#   JSON contains units referencing "input_wunits0000" -- mismatch
#   Also: TGLF file has no normalisations, so physical reference values are gone
```

## The two sub-problems

### A. Normalisation reference values are lost

**Already solved** (non-controversial): write a `pyroscan_norms.json`
alongside the scan containing `{Bref, Tref, nref, Lref, mref}`.
On reload, call `pyro.read_reference_values()` to restore them.

### B. Instance-specific unit names don't survive the round-trip

The JSON stores units like `vref_nrl_input_wunits0000`.  After reload
the Pyro's norms are named `pyroscan_base0000`, so pint cannot resolve
the old unit string.

The unit name needs to change from:

    vref_nrl_input_wunits0000 / lref_minor_radius_input_wunits0000

to:

    vref_nrl_pyroscan_base0000 / lref_minor_radius_pyroscan_base0000

This document focuses on sub-problem B.

---

## Complicating factor: mixed conventions

PyroScan auto-assigns units to bare-number parameters.  The assigned
units can come from **different conventions** within the same scan:

| parameter | auto-assigned unit | convention |
|-----------|--------------------|------------|
| `ky`      | `1 / rhoref_unit`  | cgyro      |
| `beta`    | `beta_ref_ee_B0`   | pyrokinetics |
| `gamma_exb` | `vref_nrl / lref_minor_radius` | pyrokinetics |

This means there is no single `ConventionNormalisation` object that can
convert all parameters without changing some magnitudes.

---

## Solution options

### Option 1: Suffix string manipulation (current PR implementation)

**Write:** strip the instance suffix from unit strings in the JSON encoder.

    vref_nrl_input_wunits0000  -->  vref_nrl

**Read:** tokenise the unit string and append the new instance suffix.

    vref_nrl  -->  vref_nrl_pyroscan_base0000

```python
# Write side (in NumpyEncoder)
unit_str = unit_str.replace(f"_{norm_name}", "")

# Read side
tokens = re.split(r"(\s+|[*/^()]|\*\*)", unit_str)
for token in tokens:
    candidate = token + suffix
    if candidate in ureg:
        use candidate
```

**Pros:**
- Magnitude-preserving for all unit types regardless of convention
- Works with mixed conventions in the same scan
- Backward-compatible with old JSON files (no suffix to strip/add = no-op)

**Cons:**
- String manipulation on unit names is fragile and not idiomatic pint usage
- The tokeniser must handle all pint formatting edge cases (e.g. `**`, parentheses, numeric exponents)
- Not reusable -- solves only this one serialisation problem rather than addressing the underlying design

### Option 2: Convention round-trip via `.to(convention, context)`

**Write:** convert all quantities to a fixed convention (e.g. pyrokinetics),
then strip the instance suffix.

**Read:** convert from generic pyrokinetics units to the target convention.

```python
# Write side
quantity.to(norms.pyrokinetics, norms.context)  # normalise to pyrokinetics
# then strip suffix in encoder

# Read side
generic = magnitude * ureg(unit_str)
generic.to(target_convention, norms.context)    # restore instance units
```

**Pros:**
- Uses pint's native `.to()` mechanism -- idiomatic and reusable
- No string tokenisation needed on the read side

**Cons:**
- Fails with mixed conventions.  `.to(convention)` converts *all* unit
  components to the convention's preferred bases, which changes magnitudes
  when the original unit was from a different convention.
  Example: `0.1 / rhoref_unit` (cgyro) converted to pyrokinetics becomes
  `0.35 / rhoref_pyro` -- the magnitude changes because `rhoref_unit != rhoref_pyro`.
- Requires storing which convention was used, plus a backward-compatibility
  path for old JSON files.
- **Cannot work unless all parameters use the same convention.**

### Option 3: Normalisation system registers suffix-only context rules

Add 1:1 context transformations in `SimulationNormalisation` that map each
generic base unit to its instance-specific counterpart without changing the
base name:

    rhoref_unit  <-->  rhoref_unit_run0042     (factor = 1)
    beta_ref_ee_B0  <-->  beta_ref_ee_B0_run0042  (factor = 1)

Then `value.to(unit_with_suffix, context)` is magnitude-preserving
regardless of convention, and the read side becomes:

```python
generic.to(target_convention, norms.context)
# context rules handle the suffix mapping with factor=1
```

**Pros:**
- Fully pint-native: no string manipulation anywhere
- Convention-agnostic: works with mixed conventions because the context
  rules are per-unit, not per-convention
- Reusable for any code that needs to convert between generic and instance units

**Cons:**
- Requires changes to `SimulationNormalisation` / `ConventionNormalisation`
  in `normalisation.py`
- Increases the number of context rules (one per base unit per convention)
- Needs careful design to avoid conflicts with existing context transformations

### Option 4: Eliminate mixed conventions at source

Change the auto-unit-assignment in PyroScan to always use the
`gk_input.norm_convention`, so all parameters are in the same convention.
Then Option 2 works cleanly.

**Pros:**
- Fixes the root cause: parameters should arguably be in the gk_input's convention
- Makes Option 2 viable with no special cases

**Cons:**
- Could be a breaking change for users who expect specific unit conventions
  on scan parameters
- Doesn't help with user-supplied quantities that are explicitly in a
  different convention
- Still requires a backward-compatibility path for existing JSON files

### Option 5: Store generic dimensionless keys, convert on use

Store parameter values in the JSON as plain numbers with a separate
`"units"` field that records the full unit string (with instance name).
On reload, ignore the stored unit name entirely and re-derive units from
the parameter key's default mapping + the new Pyro's norms.

**Pros:**
- JSON contains only numbers -- no unit strings to break
- Re-derivation from parameter keys is already implemented

**Cons:**
- Loses information if the user supplied explicit units that differ from defaults
- May silently change the meaning of stored values if defaults change

---

## Recommendation

**Option 3** (context rules in the normalisation system) is the cleanest
long-term fix.  It keeps all unit logic inside pint and makes the
generic-to-instance mapping a first-class operation.

If that is too large a change for this PR, **Option 1** (suffix manipulation)
is the pragmatic short-term fix that works correctly today, with the
understanding that it should be replaced once the normalisation system
supports suffix-only context rules.
