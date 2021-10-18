from pyrokinetics import Pyro
import os

home = os.environ["HOME"]
base = home + "/pyrokinetics/pyrokinetics/templates/"

gs2input = base + "input.gs2"
eqfile = base + "test.geqdsk"
pyro = Pyro(eq_file=eqfile, eq_type="GEQDSK")


geo = pyro.geo

print(geo["shat"])
geo["shat"] = 2.3

print(geo["shat"])

flags = {
    "gs2_diagnostics_knobs": {
        "write_fields": True,
        "write_kpar": True,
    },
}

pyro.add_flags(flags)
pyro.write_single(filename="test_pyro.in")
