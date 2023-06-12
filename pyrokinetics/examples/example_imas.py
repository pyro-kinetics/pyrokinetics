from pyrokinetics import Pyro, template_dir
from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro
from idspy_toolkit import ids_to_hdf5

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


ref_dict = {
    "tref": 1000.0,
    "nref": 1e19,
    "lref": 1.5,
    "bref": 2.0,
}


ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS",
    name="testing",
    ref_dict=ref_dict,
    format="json",
    file_name="imas.json",
)


ids_to_hdf5(ids, "test_pyro.hdf5")


# new_ids = ids_to_pyro(pyro, comment="Testing IMAS", name='testing', ref_dict=ref_dict, format="json", file_name="imas.json")
