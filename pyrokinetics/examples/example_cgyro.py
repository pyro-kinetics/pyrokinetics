from pyrokinetics import Pyro
import os

templates = os.path.join('..', 'pyrokinetics', 'pyrokinetics', 'templates')

cgyro_template = os.path.join(templates, 'input.cgyro')

pyro = Pyro(gk_file=cgyro_template, gk_type='CGYRO')

miller = pyro.miller

flags =  {'THETA_PLOT' : 32 }

pyro.add_flags(flags)
pyro.write_gk_file(file_name='test_cgyro.cgyro')

pyro.set_output_code('GS2')
pyro.write_gk_file(file_name='test_cgyro.gs2')

