from pyrokinetics import Pyro
import os

home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

pyro = Pyro(gkFile=base+'test_gs2.cgyro', gkType='CGYRO')

mil = pyro.mil

flags =  {'THETA_PLOT' : 32 }

pyro.addFlags(flags)
pyro.writeSingle(filename='test_cgyro.cgyro')

pyro.setOutputCode('GS2')
pyro.writeSingle(filename='test_cgyro.gs2')

