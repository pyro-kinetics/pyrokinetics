import pyrokinetics

gene_template = pyrokinetics.template_dir / "input.gene"

pyro = pyrokinetics.Pyro(gk_file=gene_template, gk_code="GENE")

pyro.write_gk_file(file_name="test_gene.gene")

pyro.gk_code = "GS2"
pyro.write_gk_file(file_name="test_gene.gs2")

# pyro.gk_code = 'CGYRO'
# pyro.write_gk_file(file_name='test_gene.cgyro')
