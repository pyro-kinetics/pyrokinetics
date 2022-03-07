from pyrokinetics.gk_code import GKCodeCGYRO
from pyrokinetics import template_dir
import pytest


class TestGKCodeCGYRO:
    @pytest.fixture
    def cgyro(self):
        return GKCodeCGYRO()

    def test_verify(self, cgyro):
        """Ensure that 'verify' does not raise exception on CGYRO file"""
        cgyro.verify(template_dir.joinpath("input.cgyro"))

    @pytest.mark.parametrize("filename", ["input.gs2", "input.gene", "transp.cdf"])
    def test_verify_bad_inputs(self, cgyro, filename):
        """Ensure that 'verify' raises exception on non-CGYRO file"""
        with pytest.raises(Exception):
            cgyro.verify(template_dir.joinpath(filename))
