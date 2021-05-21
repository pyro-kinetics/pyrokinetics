from pyrokinetics import Pyro
import os

home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

pyro = Pyro(gkFile=base+'input.gs2', gkType='GS2')

flags =  {'gs2_diagnostics_knobs' :
          { 'write_fields' : True,
            'write_kpar' : True,
          },
}

pyro.addFlags(flags)
pyro.writeSingle(filename='test_gs2.gs2')

pyro.setOutputCode('CGYRO')
pyro.writeSingle(filename='test_gs2.cgyro')
