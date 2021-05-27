from pyrokinetics import Pyro
import os
import numpy as np

home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

# Equilibrium file 
eq_file = base + 'test.geqdsk'

# Kinetics data file 
kinetics_file = base + 'scene.cdf'

# Load up pyro object
pyro = Pyro(eq_file=eq_file, eq_type='GEQDSK', kinetics_file=kinetics_file, kinetics_type='SCENE')

# Generate local Miller parameters at psi_n=0.5
pyro.load_local_geometry(psi_n=0.5)

pyro.load_local_species(psi_n=0.5)

pyro.gk_code = 'GS2'

pyro.write_gk_file(file_name='test_gs2.in')
