from pyrokinetics import Pyro, template_dir

cgyro_template = template_dir / "input.cgyro"

pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")

pyro.add_flags({"THETA_PLOT" : 32})

#FIXME This doesn't work! write_gk_file calls GKInputCGYRO.set, which overwrites
#      THETA_PLOT with ntheta. 
pyro.write_gk_file(file_name="test_cgyro.cgyro") # Should write modified CGYRO
pyro.write_gk_file(file_name="test_cgyro.gs2", gk_code="GS2")
pyro.write_gk_file(file_name="test_cgyro.gene", gk_code="GENE")
