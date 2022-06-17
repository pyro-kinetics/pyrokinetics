import pyrokinetics

gs2_template = pyrokinetics.template_dir / "input.gs2"

pyro = pyrokinetics.Pyro(gk_file=gs2_template)

flags = {
    "gs2_diagnostics_knobs": {
        "write_fields": True,
        "write_kpar": True,
    },
}

pyro.add_flags(flags)

pyro.write_gk_file(file_name="test_gs2.gs2")
pyro.write_gk_file(file_name="test_gs2.cgyro", gk_code="CGYRO")
pyro.write_gk_file(file_name="test_gs2.gene", gk_code="GENE")
