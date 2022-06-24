from pyrokinetics import Pyro, template_dir

cgyro_template = template_dir / "input.cgyro"

pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")

flags = {"THETA_PLOT": 32}

pyro.add_flags(flags)
pyro.write_gk_file(file_name="test_cgyro.cgyro")

pyro.gk_code = "GS2"
pyro.write_gk_file(file_name="test_cgyro.gs2")

pyro.gk_code = "GENE"
pyro.write_gk_file(file_name="test_cgyro.gene")
