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


def test_nxgrid_is_written_as_an_integer():
    """A float-valued ntheta must not write a float nxgrid into the deck.

    PyroScan axes are commonly built with ``np.array(..., dtype=float)``, which
    made ``set()`` write ``NXGRID = 65.0``. That deck is then unreadable: the
    output reader does ``MetricTerms(ntheta=data["nxgrid"] * 4)``, and
    ``np.linspace`` rejects a float ``num``.
    """
    gftm = GKInputGFTM(template_dir / "input.gftm")
    local_geometry = gftm.get_local_geometry()
    local_species = gftm.get_local_species()
    numerics = gftm.get_numerics()
    numerics.ntheta = 65.0

    gftm.set(local_geometry, local_species, numerics)

    assert gftm.data["nxgrid"] == 65
    assert isinstance(gftm.data["nxgrid"], int)


def test_ntheta_read_back_as_integer_from_float_deck():
    """A deck carrying a float nxgrid must still give an integer ntheta.

    Covers decks already written by the version that emitted "NXGRID = 65.0".
    """
    gftm = GKInputGFTM(template_dir / "input.gftm")
    gftm.data["nxgrid"] = 65.0

    numerics = gftm.get_numerics()

    assert numerics.ntheta == 65
    assert isinstance(numerics.ntheta, int)
