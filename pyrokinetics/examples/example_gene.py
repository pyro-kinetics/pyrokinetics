import pyrokinetics
import os
import sys


templates = os.path.join("..", "pyrokinetics", "pyrokinetics", "templates")

gene_template = os.path.join(templates, "input.gene")

pyro = pyrokinetics.Pyro(gk_file=gene_template, gk_type="GENE")

pyro.write_gk_file(file_name="test_gene.gene")

pyro.gk_code = "GS2"
pyro.write_gk_file(file_name="test_gene.gs2")

# pyro.gk_code = 'CGYRO'
# pyro.write_gk_file(file_name='test_gene.cgyro')
