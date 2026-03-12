from pathlib import Path

from io import StringIO
import numpy as np
from ruamel.yaml import YAML

from pyrokinetics import __commit__

from ..diagnostics.convergence import ConvergenceTestLinear
from ..local_geometry.local_geometry import default_inputs as lg_default_inputs


def _magnitude(quantity):
    if hasattr(quantity, "magnitude"):
        value = quantity.magnitude
        if value.size == 1:
            value = float(value)
        else:
            value = list(value)
    else:
        value = quantity

    return value


class SimDBYaml:
    def __init__(self, filepath=None, pyro=None):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(sequence=4, offset=2)
        self._add_numpy_representations()
        self.data = None
        self.filepath = None
        self.pyro = pyro
        if filepath:
            self.read(filepath)

        input_file = str(self.pyro._gk_file_record[self.pyro.gk_code].resolve())
        if self.data["inputs"][0]["uri"] == "file://path/to/inputs":
            self.data["inputs"][0]["uri"] = f"file:{input_file}"
        else:
            self.add_input(input_file)

        output_file = str(self.pyro._gk_output_file_record[self.pyro.gk_code].resolve())
        if self.data["outputs"][0]["uri"] == "file://path/to/outputs":
            self.data["outputs"][0]["uri"] = f"file:{output_file}"
        else:
            self.add_output(output_file)

        pyro_workflow = {
            "name": "pyrokinetics",
            "commit": __commit__,
            "repo": "https://github.com/pyro-kinetics/pyrokinetics",
        }
        self.set_workflow(**pyro_workflow)

        # Save Pyro info
        self.add_local_geometry()
        self.add_local_species()
        self.add_numerics()
        self.add_units()

        if self.pyro.gk_output:
            self.add_gk_output()
            self.add_convergence()

    def _add_numpy_representations(self):
        self.yaml.representer.add_representer(
            np.float64, lambda dumper, data: dumper.represent_float(float(data))
        )

        self.yaml.representer.add_representer(
            np.int64, lambda dumper, data: dumper.represent_int(int(data))
        )

        self.yaml.representer.add_representer(
            np.str_, lambda dumper, data: dumper.represent_str(str(data))
        )

    def read(self, filepath):
        """Read a YAML manifest file."""
        self.filepath = Path(filepath)
        with self.filepath.open("r") as f:
            self.data = self.yaml.load(f)

    def write(self, filepath=None):
        """Write the YAML manifest file back to disk."""
        target = Path(filepath) if filepath else self.filepath
        if not target:
            raise ValueError("No filepath specified for writing.")
        with target.open("w") as f:
            self.yaml.dump(self.data, f)

    def get_metadata(self):
        """Return the metadata dictionary (first entry in metadata list)."""
        if "metadata" in self.data and len(self.data["metadata"]) > 0:
            return self.data["metadata"][0]["values"]
        return {}

    def get_workflow(self):
        """Return the main workflow dictionary (may contain codes)."""
        meta = self.get_metadata()
        return meta.get("workflow", {})

    def get_codes(self):
        """Return the list of codes inside the workflow (empty if missing)."""
        wf = self.get_workflow()
        return wf.get("codes", [])

    def add_code(self, name, commit, repo):
        """Add a new code to the workflow.codes list."""
        wf = self.get_workflow()
        if "codes" not in wf or wf["codes"] is None:
            wf["codes"] = []
        wf["codes"].append({"name": name, "commit": commit, "repo": repo})

    def set_alias(self, alias):
        self.data["alias"] = alias

    def set_workflow(self, name, commit, repo):
        """Set or overwrite the main workflow information."""
        meta = self.get_metadata()
        meta["workflow"] = {"name": name, "commit": commit, "repo": repo}

    def set_metadata_value(self, key, value):
        """Set a metadata field (creates it if missing)."""
        if "metadata" not in self.data or len(self.data["metadata"]) == 0:
            self.data["metadata"] = [{"values": {}}]
        self.data["metadata"][0]["values"][key] = value

    def add_input(self, uri):
        """Add a new input file URI."""
        if "inputs" not in self.data:
            self.data["inputs"] = []
        self.data["inputs"].append({"uri": f"file:{uri}"})

    def add_output(self, uri):
        """Add a new output file URI."""
        if "outputs" not in self.data:
            self.data["outputs"] = []
        self.data["outputs"].append({"uri": f"file:{uri}"})

    def add_description(self, text):
        """Add a new output file URI."""
        meta = self.get_metadata()
        meta["description"] = text

    def add_units(self):
        meta = self.get_metadata()
        convention = self.pyro.norms.default_convention.references
        convention = {k: str(v) for k, v in convention.items()}
        meta["convention"] = convention

    def add_local_geometry(self):
        meta = self.get_metadata()
        local_geometry = self.pyro.local_geometry
        lg_dict = {"type": local_geometry.local_geometry}
        for key in lg_default_inputs().keys():
            lg_dict[key] = _magnitude(getattr(local_geometry, key))
        for key in local_geometry._shape_coefficient_names():
            lg_dict[key] = _magnitude(getattr(local_geometry, key))
        lg_dict["bunit_over_b0"] = _magnitude(local_geometry.bunit_over_b0)

        meta["local_geometry"] = lg_dict

    def add_local_species(self):
        meta = self.get_metadata()
        local_species = self.pyro.local_species
        keys = [
            "name",
            "mass",
            "z",
            "dens",
            "temp",
            "omega0",
            "nu",
            "inverse_ln",
            "inverse_lt",
            "domega_drho",
        ]

        ls_dict = {
            "nspecies": local_species.nspec,
            "names": local_species.names,
            "zeff": _magnitude(local_species.zeff),
        }
        for name in local_species.names:
            species_dict = {}
            species = local_species[name]
            for key in keys:
                species_dict[key] = _magnitude(getattr(species, key))
                ls_dict[name] = species_dict

        meta["local_species"] = ls_dict

    def add_numerics(self):
        meta = self.get_metadata()
        numerics = self.pyro.numerics
        numerics_dict = {}
        for key, value in numerics.items():
            if key in ["_metadata", "_already_warned"]:
                continue
            numerics_dict[key] = _magnitude(value)
        meta["numerics"] = numerics_dict

    def add_convergence(self):
        meta = self.get_metadata()
        if not self.pyro.numerics.nonlinear:
            convergence = ConvergenceTestLinear(self.pyro)

        meta["convergence"] = convergence.to_dict()

    def add_gk_output(self):
        meta = self.get_metadata()
        if not self.pyro.numerics.nonlinear:
            gk_output = {}

        meta["gk_output"] = gk_output

    def __repr__(self):
        stream = StringIO()
        self.yaml.dump(self.data, stream)
        return stream.getvalue()
