import pyrokinetics
import os

templates = os.path.join('..', 'pyrokinetics', 'pyrokinetics', 'templates')

gs2_template = os.path.join(templates, 'step.in')

pyro = pyrokinetics.Pyro(gk_file=gs2_template, gk_type='GS2')

flags =  {'gs2_diagnostics_knobs' :
          { 'write_fields' : True,
            'write_kpar' : True,
          },
}

pyro.add_flags(flags)
pyro.write_gk_file(file_name='step_gs2.gs2')

pyro.gk_code = 'GENE'
pyro.write_gk_file(file_name='step.gene')
