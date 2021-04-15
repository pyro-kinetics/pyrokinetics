from pyrokinetics import Pyro
import os

home = os.environ['HOME']
base = home+'/pyrokinetics/pyrokinetics/templates/'

pyro = Pyro(gkFile=base+'input.gs2', gkType='GS2')

mil = pyro.mil

print(mil['shat'])
mil['shat'] = 2.3

print(mil['shat'])

flags =  {'gs2_diagnostics_knobs' :
          { 'write_fields' : True,
            'write_kpar' : True,
          },
}

pyro.addFlags(flags)
pyro.writeSingle(filename='test_gs2.in')
