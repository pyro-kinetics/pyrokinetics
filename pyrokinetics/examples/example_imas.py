from pyrokinetics import Pyro
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
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_linear\input.cgyro"
)
# Load in file
pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

# Write IDS file
ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS CGYRO",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="cgyro_test.hdf5",
)

# Read IDS file back in
new_ids = ids_to_pyro("cgyro_test.hdf5")


# Point to GENE input file
gene_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\GENE_linear\parameters"
)

# Load in GENE file and output
pyro = Pyro(gk_file=gene_template)
pyro.load_gk_output()
data = pyro.gk_output

ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS GENE",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="gene_test.hdf5",
)

new_ids = ids_to_pyro("gene_test.hdf5")


# Point to GS2 input file
gs2_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\GS2_linear\gs2.in"
)

# Load in file and load data
pyro = Pyro(gk_file=gs2_template)
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS GS2",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="gs2_test.hdf5",
)

new_ids = ids_to_pyro("gs2_test.hdf5")


# Point to CGYRO input file
cgyro_nl_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_run\input.cgyro"
)
# Load in file
pyro = Pyro(gk_file=cgyro_nl_template, gk_code="CGYRO")
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output
ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS CGYRO NL",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="cgyro_nl_test.hdf5",
)

new_ids = ids_to_pyro("cgyro_nl_test.hdf5")
