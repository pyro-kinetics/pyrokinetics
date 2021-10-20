from pyrokinetics import Pyro, template_dir

gs2input = template_dir / "input.gs2"
eqfile = template_dir / "test.geqdsk"
pyro = Pyro(eq_file=eqfile, eq_type="GEQDSK", local_geometry="Miller", gk_type="GS2")

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
pyro.write_gk_file(file_name="test_pyro.in")
