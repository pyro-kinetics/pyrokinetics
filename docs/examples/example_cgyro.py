from pyrokinetics import Pyro, template_dir

cgyro_template = template_dir / "input.cgyro"

pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")

flags = {"N_THETA": 64}

pyro.add_flags(flags)
pyro.write_gk_file(file_name="test_cgyro.cgyro")
