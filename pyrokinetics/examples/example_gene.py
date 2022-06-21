import pyrokinetics

gene_template = pyrokinetics.template_dir / "input.gene"

pyro = pyrokinetics.Pyro(gk_file=gene_template)

pyro.write_gk_file(file_name="test_gene.gene")
pyro.write_gk_file(file_name="test_gene.gs2", gk_code="GS2")
pyro.write_gk_file(file_name='test_gene.cgyro', gk_code="CGYRO")
