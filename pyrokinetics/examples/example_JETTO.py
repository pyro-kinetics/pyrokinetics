from pyrokinetics import Pyro
import os
import numpy as np

# Point to input files
templates = os.path.join('..', 'pyrokinetics', 'pyrokinetics', 'templates')

# Equilibrium file
eq_file = os.path.join(templates, 'test.geqdsk')

# Kinetics data file
kinetics_file = os.path.join(templates, 'jetto.cdf')

# Load up pyro object
pyro = Pyro(eq_file=eq_file, eq_type='GEQDSK', kinetics_file=kinetics_file, kinetics_type='JETTO')

# Generate local parameters at psi_n=0.5
pyro.load_local(psi_n=0.5, geometry_type='Miller')

# Select code as CGYRO
pyro.gk_code = 'CGYRO'

# Write CGYRO input file using default template
pyro.write_gk_file(file_name='test_jetto.cgyro')

# Change GK code to GS2
pyro.gk_code = 'GS2'


# Write single input file using my own template
pyro.write_gk_file(file_name='test_jetto.gs2', template_file='step.in')
