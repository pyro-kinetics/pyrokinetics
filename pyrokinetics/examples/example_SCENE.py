from pyrokinetics import Pyro
import os

# Point to input files
templates = os.path.join('..', 'pyrokinetics', 'pyrokinetics', 'templates')

# Equilibrium file 
eq_file = os.path.join(templates, 'test.geqdsk')

# Kinetics data file 
kinetics_file = os.path.join(templates, 'scene.cdf')

pyro = Pyro(eq_file=eq_file, eq_type='GEQDSK', kinetics_file=kinetics_file, kinetics_type='SCENE')

# Generate local Miller parameters at psi_n=0.5
pyro.load_local(psi_n=0.5, local_geometry='Miller')

# Select code as CGYRO
pyro.gk_code = 'CGYRO'

# Write CGYRO input file using default template
pyro.write_gk_file(file_name='test_scene.cgyro')

# Write single GS2 input file, specifying the code type
# in the call.
pyro.write_gk_file(file_name='test_scene.gs2', gk_code = 'GS2')
