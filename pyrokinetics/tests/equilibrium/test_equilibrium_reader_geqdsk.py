from pyrokinetics.equilibrium import EquilibriumReaderGEQDSK
from pyrokinetics import template_dir
import pytest
import numpy as np


class TestEquilibriumReaderGEQDSK:
    @pytest.fixture
    def geqdsk_reader(self):
        return EquilibriumReaderGEQDSK()

    @pytest.fixture
    def example_file(self):
        return template_dir.joinpath("transp_eq.geqdsk")

    def test_read(self, geqdsk_reader, example_file):
        """
        Ensure it can read the example GEQDSK file, and that it produces a valid dict
        """
        result = geqdsk_reader(example_file)
        assert isinstance(result, dict)
        # Check that a subset of variables are present
        assert np.all(
            np.isin(
                ["psi_axis", "psi_bdry", "a_minor", "pressure"], list(result.keys())
            )
        )

    def test_verify(self, geqdsk_reader, example_file):
        """Ensure verify completes without throwing an error"""
        geqdsk_reader.verify(example_file)

    def test_read_file_does_not_exist(self, geqdsk_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir.joinpath("helloworld")
        with pytest.raises((FileNotFoundError, ValueError)):
            geqdsk_reader(filename)

    @pytest.mark.parametrize("filename", ["input.gs2", "transp_eq.cdf"])
    def test_read_file_is_not_geqdsk(self, geqdsk_reader, filename):
        """Ensure failure when given a non-geqdsk file"""
        filename = template_dir.joinpath(filename)
        with pytest.raises(Exception):
            geqdsk_reader(filename)

    def test_verify_file_does_not_exist(self, geqdsk_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir.joinpath("helloworld")
        with pytest.raises((FileNotFoundError, ValueError)):
            geqdsk_reader.verify(filename)

    @pytest.mark.parametrize("filename", ["input.gs2", "transp_eq.cdf"])
    def test_verify_file_is_not_geqdsk(self, geqdsk_reader, filename):
        """Ensure failure when given a non-geqdsk file"""
        filename = template_dir.joinpath(filename)
        with pytest.raises(ValueError):
            geqdsk_reader.verify(filename)
