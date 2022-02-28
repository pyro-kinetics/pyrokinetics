from pyrokinetics.gk_code import GKCodeGENE
from pyrokinetics import template_dir
import pytest


class TestGKCodeGENE:
    @pytest.fixture
    def gene(self):
        return GKCodeGENE()

    def test_verify(self, gene):
        """Ensure that 'verify' does not raise exception on GENE file"""
        gene.verify(template_dir.joinpath("input.gene"))

    @pytest.mark.parametrize("filename", ["input.cgyro", "input.gs2", "transp.cdf"])
    def test_verify_bad_inputs(self, gene, filename):
        """Ensure that 'verify' raises exception on non-GENE file"""
        with pytest.raises(Exception):
            gene.verify(template_dir.joinpath(filename))
