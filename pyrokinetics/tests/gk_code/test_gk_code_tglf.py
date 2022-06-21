from pyrokinetics.gk_code import GKCodeTGLF
from pyrokinetics import template_dir
import pytest


class TestGKCodeTGLF:
    @pytest.fixture
    def tglf(self):
        return GKCodeTGLF()

    def test_verify(self, tglf):
        """Ensure that 'verify' does not raise exception on TGLF file"""
        tglf.verify(template_dir.joinpath("input.tglf"))

    @pytest.mark.parametrize("filename", ["input.gs2", "input.gene", "transp.cdf"])
    def test_verify_bad_inputs(self, tglf, filename):
        """Ensure that 'verify' raises exception on non-TGLF file"""
        with pytest.raises(Exception):
            tglf.verify(template_dir.joinpath(filename))
