from pyrokinetics.gk_code import GKCodeGS2
from pyrokinetics import template_dir
import pytest


class TestGKCodeGS2:
    @pytest.fixture
    def gs2(self):
        return GKCodeGS2()

    def test_verify(self, gs2):
        """Ensure that 'verify' does not raise exception on GS2 file"""
        gs2.verify(template_dir.joinpath("input.gs2"))

    @pytest.mark.parametrize("filename", ["input.cgyro", "input.gene", "transp.cdf"])
    def test_verify_bad_inputs(self, gs2, filename):
        """Ensure that 'verify' raises exception on non-GS2 file"""
        with pytest.raises(Exception):
            gs2.verify(template_dir.joinpath(filename))
