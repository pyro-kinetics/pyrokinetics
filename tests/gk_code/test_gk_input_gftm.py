from pyrokinetics import template_dir
from pyrokinetics.gk_code import GKInputGFTM


def test_set_nxgrid_above_tglf_limit():
    gftm = GKInputGFTM(template_dir / "input.gftm")
    local_geometry = gftm.get_local_geometry()
    local_species = gftm.get_local_species()
    numerics = gftm.get_numerics()
    numerics.ntheta = 64

    gftm.set(local_geometry, local_species, numerics)

    assert gftm.data["nxgrid"] == 64
