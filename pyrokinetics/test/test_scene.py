from pyrokinetics import Pyro
import os
import numpy as np

# Point to input files
home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

# Equilibrium file 
eqFile = base+'test.geqdsk'

# Kinetics data file 
kinFile = base+'scene.cdf'

# Load up pyro object
pyro = Pyro(eqFile=eqFile, eqType='GEQDSK', kinFile=kinFile, kinType='SCENE')

# Generate local Miller parameters at psiN=0.5
pyro.loadLocal(psiN=0.5, geoType='Miller')

# Select code as CGYRO
pyro.gkCode = 'CGYRO'

# Write CGYRO input file using default template
pyro.writeSingle(filename='test_scene.cgyro')

# Change GK code to GS2
pyro.gkCode = 'GS2'

# Write single GS2 input file
pyro.writeSingle(filename='test_scene.gs2')
