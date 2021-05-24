from pyrokinetics import Pyro
import os
import numpy as np

# Point to input files
home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

# Equilibrium file 
eq_file = base + 'test.geqdsk'

# Kinetics data file 
kinetics_file = base + 'jetto.cdf'

# Template file
gk_file = 'myTemplate.gs2'

# Load up pyro object
pyro = Pyro(eq_file=eq_file, eq_type='GEQDSK', kinetics_file=kinetics_file, kinetics_type='JETTO', gk_file=gk_file,
            gk_type='GS2', geometry_type='Miller')

# Generate local parameters at psi_n=0.5
pyro.load_local(psi_n=0.5)

# Change GK code to GS2
pyro.gk_code = 'GS2'

# Dictionary for extra flags
# Nested for GS2 namelist
flags =  {'gs2_diagnostics_knobs' :
          { 'write_fields' : True,
            'write_kpar' : True,
          },
}

pyro.add_flags(flags)

# Write single input file using my own template
pyro.write_gk_file(file_name='test_jetto.gs2')
