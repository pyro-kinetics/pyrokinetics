from pyrokinetics import Pyro, template_dir

gs2input = template_dir / "input.gs2"
eqfile = template_dir / "test.geqdsk"
pyro = Pyro(gk_code="GS2", gk_file=gs2input)

geo = pyro.local_geometry

geo["shat"] = 2.3


flags = {
    "gs2_diagnostics_knobs": {
        "write_fields": True,
        "write_kpar": True,
    },
}

pyro.add_flags(flags)
pyro.write_gk_file(file_name="test_pyro.in")
