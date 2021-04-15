from pyrokinetics import Pyro
import os

home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

pyro = Pyro(gkFile=base+'input.cgyro', gkType='CGYRO')

mil = pyro.mil

print(mil['shat'])
mil['shat'] = 2.3

print(mil['shat'])

flags =  {'THETA_PLOT' : 32 }

pyro.addFlags(flags)
pyro.writeSingle(filename='test_cgyro.in')
