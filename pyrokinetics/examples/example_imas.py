from pyrokinetics import Pyro, template_dir
from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro


ref_dict = {
    "tref": 1000.0,
    "nref": 1e19,
    "lref": 1.5,
    "bref": 2.0,
}


#
# Point to CGYRO input file
cgyro_template = (
    "C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_linear\input.cgyro"
)
# Load in file
pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output
# ids = pyro_to_ids(pyro, comment="Testing IMAS CGYRO", name='testing', ref_dict=ref_dict, format="hdf5", file_name="cgyro_hdf5")
ids = pyro_to_ids(pyro, comment="Testing IMAS CGYRO", name="testing", ref_dict=ref_dict)

# Point to CGYRO input file
gene_template = (
    "C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\GENE_linear\parameters"
)

# Load in file
pyro = Pyro(gk_file=gene_template)
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

ids = pyro_to_ids(pyro, comment="Testing IMAS GENE", name="testing", ref_dict=ref_dict)


# Point to CGYRO input file
gs2_template = (
    "C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\GS2_linear\gs2.in"
)

# Load in file
pyro = Pyro(gk_file=gs2_template)
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

ids = pyro_to_ids(pyro, comment="Testing IMAS GS2", name="testing", ref_dict=ref_dict)

# Point to CGYRO input file
cgyro_nl_template = (
    "C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_run\input.cgyro"
)
# Load in file
pyro = Pyro(gk_file=cgyro_nl_template, gk_code="CGYRO")
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output
ids = pyro_to_ids(
    pyro, comment="Testing IMAS CGYRO NL", name="testing", ref_dict=ref_dict
)
