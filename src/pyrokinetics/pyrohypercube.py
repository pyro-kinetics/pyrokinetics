r"""
Non-gridded parameter sets: :class:`PyroHypercube`.

:class:`~pyrokinetics.pyroscan.PyroScan` describes a *grid*: every parameter has
a list of values, and the runs are the outer product of those lists.
Many parameter studies are not grids — a Latin hypercube, a set of runs sampled
from a distribution, or simply a directory of runs somebody else made — and
squeezing them into a grid either wastes runs or is impossible.

:class:`PyroHypercube` is the non-gridded sibling. It keeps everything
``PyroScan`` does with a base :class:`~pyrokinetics.pyro.Pyro`, run directories
and output loading, and changes exactly two things:

* the runs are an explicit list of sample points, not an outer product, so
  ``parameter_dict`` holds one value **per sample** for each varied parameter;
* the output ``Dataset`` has a single ``sample`` dimension, with each varied
  parameter attached as a non-dimension coordinate along it — the natural xarray
  form for scattered points.

It also reads run trees that pyrokinetics never wrote, via
:meth:`PyroHypercube.from_directory`, recovering the varied values from each
run's input file.

Notes
-----
``PyroHypercube`` writes input files and reads outputs. It never runs or submits
anything.
"""

from __future__ import annotations

import copy
import pathlib
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pint import Quantity

from .gk_code import GKInput
from .pyro import Pyro
from .pyroscan import PyroScan, ScanLayout, get_from_dict, magnitude_and_units
from .typing import PathLike

__all__ = ["PyroHypercube"]


def _natural_key(path: pathlib.Path) -> Tuple:
    """
    Sort key that orders embedded numbers numerically.

    Run trees are usually named with a counter (``iteration_2``,
    ``iteration_10``), which lexicographic sorting gets wrong.
    """
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    )


def _is_gk_input(path: pathlib.Path, gk_code: Optional[str]) -> bool:
    """Can ``path`` be read as a gyrokinetics input file?"""
    if not path.is_file():
        return False
    try:
        if gk_code is None:
            GKInput._factory.type(str(path))
        else:
            GKInput._factory[gk_code]().verify_file_type(path)
    except Exception:
        return False
    return True


def _find_input_file(
    directory: pathlib.Path,
    file_name: Optional[str],
    gk_code: Optional[str],
) -> Optional[pathlib.Path]:
    """
    Find the gyrokinetics input file inside a run directory.

    Returns ``None`` if the directory holds no readable input file, so that
    directories which are not runs at all (``plots/``, ``logs/``, …) can be
    skipped rather than raising.
    """
    if file_name is not None:
        candidate = directory / file_name
        return candidate if candidate.is_file() else None

    if gk_code is not None:
        default = getattr(GKInput._factory[gk_code], "default_file_name", None)
        if default is not None and (directory / default).is_file():
            return directory / default

    for candidate in sorted(directory.iterdir(), key=_natural_key):
        if _is_gk_input(candidate, gk_code):
            return candidate

    return None


def _quantity_array(values: List[Any]) -> Union[Quantity, np.ndarray]:
    """
    Stack per-sample parameter values into a single array or ``Quantity``.
    """
    if any(isinstance(value, Quantity) for value in values):
        units = next(value.units for value in values if isinstance(value, Quantity))
        magnitudes = [
            value.to(units).magnitude if isinstance(value, Quantity) else value
            for value in values
        ]
        return np.asarray(magnitudes) * units
    return np.asarray(values)


class PyroHypercube(PyroScan):
    """
    A set of runs at explicitly listed parameter values, rather than on a grid.

    ``parameter_dict`` maps each varied parameter to **one value per sample**;
    every entry must therefore have the same length, which is the number of
    samples. Sample ``i`` is the point made of the ``i``-th value of every
    parameter.

    Parameters
    ----------
    pyro: Pyro, default None
        Base Pyro object, as for ``PyroScan``.
    parameter_dict: dict, default None
        ``{parameter: [value per sample]}``. All entries must be the same
        length.
    sample_names: list of str, default None
        Run-directory name of each sample, relative to ``base_directory``. If
        unset, samples are named ``sample_0000``, ``sample_0001``, … Unlike a
        gridded scan, sample values do not make useful directory names — nearby
        samples round onto the same name — so they are not used by default.
        :meth:`from_directory` passes the directory names it found.
    run_pyros: list of Pyro, default None
        Pyro object for each sample, in sample order, used instead of copying
        the base Pyro. :meth:`from_directory` passes the Pyros it read from
        disk, so that each sample keeps its own geometry and normalisation.
    **kwargs
        Passed to :class:`~pyrokinetics.pyroscan.PyroScan`.

    Examples
    --------
    Read a directory of GS2 runs that pyrokinetics did not write::

        cube = PyroHypercube.from_directory(
            "MTM_MODELS",
            pattern="iteration_*",
            params=["ky", "electron_temp_gradient"],
            gk_code="GS2",
        )
        cube.load_gk_output()
        cube.gk_output.data["growth_rate"]  # dims: (sample, ...)

    and write the same points out as TGLF decks::

        cube.write_gk_decks("TGLF", overrides={"NBASIS_MAX": 6}, target="tglf")
    """

    JSON_ATTRS = PyroScan.JSON_ATTRS + ["sample_names"]

    def __init__(
        self,
        pyro=None,
        parameter_dict: Optional[Dict[str, Any]] = None,
        sample_names: Optional[Iterable[str]] = None,
        run_pyros: Optional[Iterable[Pyro]] = None,
        **kwargs,
    ):
        self._sample_names = None
        self._run_pyros = list(run_pyros) if run_pyros is not None else None

        if parameter_dict:
            self._validate_parameter_dict(parameter_dict)

        self.sample_names = sample_names

        super().__init__(pyro=pyro, parameter_dict=parameter_dict, **kwargs)

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_parameter_dict(parameter_dict: Dict[str, Any]) -> int:
        """Check every parameter has one value per sample. Returns that count."""
        sizes = {name: len(values) for name, values in parameter_dict.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(
                "PyroHypercube parameters are sampled together, so every "
                "parameter must have one value per sample. Got differing "
                f"lengths: {sizes}. (For an outer product over per-parameter "
                "value lists, use PyroScan.)"
            )
        return next(iter(sizes.values()), 0)

    @property
    def n_samples(self) -> int:
        """Number of sample points."""
        if self.parameter_dict:
            return self._validate_parameter_dict(self.parameter_dict)
        if self._sample_names is not None:
            return len(self._sample_names)
        if self._run_pyros is not None:
            return len(self._run_pyros)
        return 0

    @property
    def sample_names(self) -> Optional[List[str]]:
        """Run-directory name of each sample, in sample order."""
        return self._sample_names

    @sample_names.setter
    def sample_names(self, value: Optional[Iterable[str]]) -> None:
        if value is None:
            self._sample_names = None
        else:
            names = [str(name) for name in value]
            duplicates = {name for name in names if names.count(name) > 1}
            if duplicates:
                raise ValueError(
                    "PyroHypercube sample names must be unique, so that every "
                    f"sample gets its own run directory. Repeated: {sorted(duplicates)}"
                )
            self._sample_names = names

        if hasattr(self, "pyroscan_json"):
            self.pyroscan_json["sample_names"] = self._sample_names

    def default_sample_names(self) -> List[str]:
        """Names used when the caller does not supply any: ``sample_0000``, …"""
        width = max(4, len(str(max(self.n_samples - 1, 0))))
        return [f"sample_{i:0{width}d}" for i in range(self.n_samples)]

    def sample_points(self):
        """
        Generator of ``{parameter: value}`` for each sample, in sample order.

        This is the non-gridded replacement for ``PyroScan.outer_product``.
        """
        names = list(self.parameter_dict.keys())
        if not names:
            return ({} for _ in range(self.n_samples))
        return (
            dict(zip(names, values))
            for values in zip(*(self.parameter_dict[name] for name in names))
        )

    def outer_product(self):
        """
        The runs of this parameter set: its samples, *not* an outer product.

        Kept under ``PyroScan``'s name so that everything built on it (writing
        input files, updating the run Pyros, loading outputs) is inherited
        unchanged. Prefer :meth:`sample_points` in new code.
        """
        return self.sample_points()

    def build_pyro_dict(self):
        """
        Create one Pyro per sample, named by ``sample_names``.
        """
        if not self.sample_names:
            self.sample_names = self.default_sample_names()

        if self._run_pyros is not None:
            if len(self._run_pyros) != len(self.sample_names):
                raise ValueError(
                    f"Got {len(self._run_pyros)} run Pyros for "
                    f"{len(self.sample_names)} samples."
                )
            self.pyro_dict = dict(zip(self.sample_names, self._run_pyros))
        else:
            self.pyro_dict = dict(
                self.create_single_run(parameters, name=name)
                for name, parameters in zip(self.sample_names, self.sample_points())
            )

        self.run_directories = [pyro.run_directory for pyro in self.pyro_dict.values()]

    def set_parameter_dict(self, parameter_dict: Dict[str, Any]) -> None:
        """
        Replace the sampled parameter values, keeping the existing samples.

        Used by :meth:`from_directory`, which has to create the Pyros before it
        can read their parameter values back out of them.
        """
        n_samples = self._validate_parameter_dict(parameter_dict)
        if self.sample_names is not None and n_samples != len(self.sample_names):
            raise ValueError(
                f"Got {n_samples} values per parameter for "
                f"{len(self.sample_names)} samples."
            )

        convention = self.base_pyro.norms.pyrokinetics
        self.parameter_dict = {
            name: (
                values.convert_physical_units(convention)
                if hasattr(values, "convert_physical_units")
                else values
            )
            for name, values in parameter_dict.items()
        }
        self.pyroscan_json["parameter_dict"] = self.parameter_dict
        self.value_size = [len(values) for values in self.parameter_dict.values()]

    # ------------------------------------------------------------------
    # Output layout
    # ------------------------------------------------------------------
    def output_layout(self) -> ScanLayout:
        """
        Map the samples onto the dimensions of the output Dataset.

        Scattered points have no grid to lie on, so there is a single ``sample``
        dimension and every varied parameter becomes a non-dimension coordinate
        along it, alongside ``sample_name`` (the run directory each sample came
        from).
        """
        coords: Dict[str, Any] = {"sample": np.arange(self.n_samples)}
        coord_units = {}

        for name, values in self.parameter_dict.items():
            magnitude, unit = magnitude_and_units(values)
            coords[name] = (("sample",), magnitude, {"units": unit})
            coord_units[name] = unit

        if self.sample_names:
            coords["sample_name"] = ("sample", np.asarray(self.sample_names, dtype=str))

        return ScanLayout(
            coords=coords,
            coord_units=coord_units,
            shape=(self.n_samples,),
            dims=("sample",),
            squeeze_dims=tuple(self.parameter_dict.keys()),
        )

    # ------------------------------------------------------------------
    # Reading an existing run tree
    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        root: PathLike,
        pattern: str = "*",
        params: Union[Sequence[str], Mapping[str, Sequence], None] = None,
        gk_code: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> "PyroHypercube":
        """
        Build a hypercube from a directory of runs, whoever made it.

        Each matching run directory is found, its input file read, and the
        values of ``params`` recovered from it. Nothing is assumed about the
        directory names, so this works on trees pyrokinetics never wrote.

        Parameters
        ----------
        root: PathLike
            Directory the runs live under.
        pattern: str, default "*"
            Glob, relative to ``root``, matching either the run directories
            (``"iteration_*"``, ``"*/iteration_*"``) or the input files
            themselves (``"*/gs2.in"``). Matches that are not runs are skipped.
        params: sequence of str, or dict
            Parameters to recover from each input file. Either names known to
            ``parameter_map`` (see ``PyroScan.load_default_parameter_keys``), or
            a dict mapping a name to ``[pyro_attribute, [keys, to, value]]``,
            e.g. ``{"beta": ["numerics", ["beta"]]}``.
        gk_code: str, default None
            Code the inputs are written for, e.g. ``"GS2"``. If unset, the type
            of each input file is inferred.
        file_name: str, default None
            Name of the input file within each run directory. If unset, the
            code's default name is used if present, otherwise the first
            readable input file in the directory.
        **kwargs
            Passed to :class:`PyroHypercube`.

        Returns
        -------
        PyroHypercube
            One sample per run found, in natural-sorted directory order.
        """
        root = pathlib.Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"PyroHypercube.from_directory: no directory {root}"
            )

        if params is None:
            raise ValueError(
                "PyroHypercube.from_directory: 'params' must name the varied "
                "parameters to recover from each input file — there is no way "
                "to tell which of the many inputs were deliberately varied."
            )

        if isinstance(params, Mapping):
            param_names = list(params.keys())
            extra_map = {name: list(value) for name, value in params.items()}
        else:
            param_names = [str(name) for name in params]
            extra_map = {}

        names, input_files = cls._discover_runs(root, pattern, file_name, gk_code)

        pyros = [Pyro(gk_file=path, gk_code=gk_code) for path in input_files]
        base_pyro = copy.deepcopy(pyros[0])

        hypercube = cls(
            pyro=base_pyro,
            parameter_dict={},
            sample_names=names,
            run_pyros=pyros,
            base_directory=root,
            file_name=input_files[0].name,
            **kwargs,
        )

        for name, (attr, location) in extra_map.items():
            hypercube.add_parameter_key(name, attr, location)

        hypercube.set_parameter_dict(hypercube._read_parameters(param_names, pyros))

        return hypercube

    @staticmethod
    def _discover_runs(
        root: pathlib.Path,
        pattern: str,
        file_name: Optional[str],
        gk_code: Optional[str],
    ) -> Tuple[List[str], List[pathlib.Path]]:
        """Find run directories under ``root``, and the input file in each."""
        names: List[str] = []
        input_files: List[pathlib.Path] = []
        skipped: List[pathlib.Path] = []

        for match in sorted(root.glob(pattern), key=_natural_key):
            if match.is_file():
                if file_name is not None and match.name != file_name:
                    continue
                if not _is_gk_input(match, gk_code):
                    skipped.append(match)
                    continue
                run_directory, input_file = match.parent, match
            elif match.is_dir():
                input_file = _find_input_file(match, file_name, gk_code)
                if input_file is None:
                    skipped.append(match)
                    continue
                run_directory = match
            else:
                continue

            names.append(str(run_directory.relative_to(root)))
            input_files.append(input_file)

        if not input_files:
            raise FileNotFoundError(
                f"PyroHypercube.from_directory: no gyrokinetics input files found "
                f"under {root} matching '{pattern}'"
                + (f" named '{file_name}'" if file_name else "")
                + (f" for gk_code '{gk_code}'" if gk_code else "")
            )

        if skipped:
            warnings.warn(
                f"PyroHypercube.from_directory: skipped {len(skipped)} match(es) "
                f"under {root} with no readable input file, e.g. {skipped[0]}",
                stacklevel=3,
            )

        if len({path.name for path in input_files}) > 1:
            warnings.warn(
                "PyroHypercube.from_directory: runs use different input file "
                f"names, e.g. {input_files[0].name} and "
                f"{next(p.name for p in input_files if p.name != input_files[0].name)}. "
                f"Using '{input_files[0].name}' as file_name.",
                stacklevel=3,
            )

        return names, input_files

    def _read_parameters(
        self, param_names: Sequence[str], pyros: Sequence[Pyro]
    ) -> Dict[str, Any]:
        """Recover each parameter's value from every run's Pyro."""
        convention = self.base_pyro.norms.pyrokinetics
        parameter_dict = {}

        for name in param_names:
            if name not in self.parameter_map:
                raise ValueError(
                    f"PyroHypercube.from_directory: don't know where to find "
                    f"'{name}' in a Pyro object. Either use one of "
                    f"{sorted(self.parameter_map)}, or pass params as a dict, "
                    f"e.g. {{'{name}': ['numerics', ['{name}']]}}."
                )
            attr, location = self.parameter_map[name]

            values = []
            for pyro in pyros:
                value = get_from_dict(getattr(pyro, attr), location)
                if isinstance(value, Quantity):
                    value = value.to(convention, convention.context)
                values.append(value)

            parameter_dict[name] = _quantity_array(values)

        return parameter_dict

    # ------------------------------------------------------------------
    # Writing input files for another code
    # ------------------------------------------------------------------
    def write_gk_decks(
        self,
        gk_code: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        target: PathLike = None,
        file_name: Optional[str] = None,
        template_file: Optional[PathLike] = None,
        code_normalisation: Optional[str] = None,
    ) -> List[pathlib.Path]:
        """
        Write one input file per sample, optionally for a different code.

        The samples keep their own geometry, species and numerics; ``overrides``
        are code-native settings applied on top of every deck, in the form that
        code's ``add_flags`` takes — flat for TGLF and CGYRO
        (``{"NBASIS_MAX": 6}``), one dict per namelist for GS2 and GENE
        (``{"knobs": {"delt": 0.01}}``). Key case is matched to the input file's
        own, so ``NBASIS_MAX`` and ``nbasis_max`` are the same setting.

        Files are written and nothing else: no job is run or submitted.

        Parameters
        ----------
        gk_code: str, default None
            Code to write for, e.g. ``"TGLF"``. Defaults to the samples' own
            code.
        overrides: dict, default None
            Code-native settings applied to every deck.
        target: PathLike
            Directory to write into, with one sub-directory per sample.
        file_name: str, default None
            Name of the input file within each sample directory. Defaults to
            the code's own default name.
        template_file: PathLike, default None
            Template used when converting to a code the samples were not read
            from.
        code_normalisation: str, default None
            Normalisation convention to write in. Defaults to the code's own.

        Returns
        -------
        list of pathlib.Path
            The files written, in sample order.
        """
        if target is None:
            raise ValueError(
                "PyroHypercube.write_gk_decks: 'target' directory is required, "
                "so that decks never overwrite the runs they came from."
            )
        target = pathlib.Path(target)

        self.update_self_parameters()

        written = []
        for name, pyro in self.pyro_dict.items():
            run = self._copy_without_output(pyro)

            if gk_code is not None and gk_code != run.gk_code:
                run.convert_gk_code(gk_code, template_file=template_file)

            if overrides:
                run.add_flags(self._match_override_case(run.gk_input, overrides))

            deck_name = file_name or GKInput._factory[run.gk_code].default_file_name
            path = target / name / deck_name
            run.write_gk_file(file_name=path, code_normalisation=code_normalisation)
            written.append(path)

        return written

    @staticmethod
    def _copy_without_output(pyro: Pyro) -> Pyro:
        """Deep-copy a Pyro, leaving behind any loaded output (which cannot be copied)."""
        output = getattr(pyro, "gk_output", None)
        if output is None:
            return copy.deepcopy(pyro)
        pyro.gk_output = None
        try:
            return copy.deepcopy(pyro)
        finally:
            pyro.gk_output = output

    @staticmethod
    def _match_override_case(
        gk_input: GKInput, overrides: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Re-key ``overrides`` to the case the input file itself uses.

        Codes are written in their conventional case (TGLF's ``NBASIS_MAX``)
        but held internally in another (``nbasis_max``); adding the wrong case
        would write the setting twice.
        """
        data = getattr(gk_input, "data", None) or {}
        lookup = {str(key).lower(): key for key in data}

        matched = {}
        for key, value in overrides.items():
            name = lookup.get(str(key).lower(), key)
            existing = data.get(name)
            if isinstance(value, Mapping) and isinstance(existing, Mapping):
                inner = {str(k).lower(): k for k in existing}
                value = {inner.get(str(k).lower(), k): v for k, v in value.items()}
            matched[name] = value

        return matched

    # ------------------------------------------------------------------
    # netCDF
    # ------------------------------------------------------------------
    def to_netcdf(self, path: PathLike, *args, **kwargs) -> None:
        """
        Write the loaded output to netCDF.

        Thin wrapper over ``self.gk_output.to_netcdf``, which converts physical
        units to generic simulation units first so the file is readable by a
        pyro that does not share this one's normalisation.
        """
        if getattr(self, "gk_output", None) is None:
            raise RuntimeError(
                "PyroHypercube.to_netcdf: no output loaded. Call "
                "load_gk_output() first."
            )
        self.gk_output.to_netcdf(path, *args, **kwargs)

    def from_netcdf(
        self, path: PathLike, output_convention: str = "pyrokinetics"
    ) -> None:
        """
        Load output previously written by :meth:`to_netcdf` into ``gk_output``.

        Equivalent to ``load_gk_output(netcdf_file=path)``; the file's generic
        simulation units are converted to ``output_convention`` using this
        object's base Pyro.
        """
        self.load_gk_output(netcdf_file=path, output_convention=output_convention)
