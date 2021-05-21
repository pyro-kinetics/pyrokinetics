from pyrokinetics import Pyro
import os
import numpy as np

home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

# Equilibrium file 
eqFile = base+'test.geqdsk'

# Kinetics data file 
kinFile = base+'scene.cdf'

# Load up pyro object
pyro = Pyro(eqFile=eqFile, eqType='GEQDSK', kinFile=kinFile, kinType='SCENE')



# Eq object
eq = pyro.eq

# Kinetics Object
kin = pyro.kin


# Generate local Miller parameters at psiN=0.5
pyro.loadMiller(psiN=0.5)


pyro.loadSpecies(psiN=0.5)

pyro.gkCode = 'GS2'

pyro.writeSingle(filename='test_gs2.in')


"""

pyro = Pyro()
# Set up equilibrium
pyro.load_geqdsk('/path/to/geqdsk')

pyro.load_miller(psiN=0.5)
"""
