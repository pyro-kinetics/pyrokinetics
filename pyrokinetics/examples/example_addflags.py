from pyrokinetics import Pyro
import os
import numpy as np

# Point to input files
home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

# Equilibrium file 
eqFile = base+'test.geqdsk'

# Kinetics data file 
kinFile = base+'jetto.cdf'

# Template file
gkFile = 'myTemplate.gs2'

# Load up pyro object
pyro = Pyro(eqFile=eqFile, eqType='GEQDSK',
            kinFile=kinFile, kinType='JETTO',
            gkFile=gkFile, gkType='GS2',
            geoType='Miller')

# Generate local parameters at psiN=0.5
pyro.loadLocal(psiN=0.5)

# Change GK code to GS2
pyro.gkCode = 'GS2'

# Dictionary for extra flags
# Nested for GS2 namelist
flags =  {'gs2_diagnostics_knobs' :
          { 'write_fields' : True,
            'write_kpar' : True,
          },
}

pyro.addFlags(flags)

# Write single input file using my own template
pyro.writeSingle(filename='test_jetto.gs2')
