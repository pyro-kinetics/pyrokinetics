from pyrokinetics.equilibrium import EquilibriumReaderTRANSP
from pyrokinetics import template_dir
import pytest
import numpy as np


class TestEquilibriumReaderTRANSP:
    @pytest.fixture
    def transp_reader(self):
        return EquilibriumReaderTRANSP()

    @pytest.fixture
    def example_file(self):
        return template_dir.joinpath("transp_eq.cdf")

    def test_read(self, transp_reader, example_file):
        """
        Ensure it can read the example TRANSP file, and that it produces a valid dict
        """
        result = transp_reader(example_file)
        assert isinstance(result, dict)
        # Check that a subset of variables are present
        assert np.all(
            np.isin(["psi_axis", "psi_bdry", "a_minor", "pressure"], list(result))
        )

    def test_verify(self, transp_reader, example_file):
        """Ensure verify completes without throwing an error"""
        transp_reader.verify(example_file)

    def test_read_file_does_not_exist(self, transp_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir.joinpath("helloworld")
        with pytest.raises((FileNotFoundError, ValueError)):
            transp_reader(filename)

    def test_read_file_is_not_netcdf(self, transp_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir.joinpath("input.gs2")
        with pytest.raises(OSError):
            transp_reader(filename)

    @pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
    def test_read_file_is_not_transp(self, transp_reader, filename):
        """Ensure failure when given a non-transp netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir.joinpath(filename)
        with pytest.raises(Exception):
            transp_reader(filename)

    def test_verify_file_does_not_exist(self, transp_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir.joinpath("helloworld")
        with pytest.raises((FileNotFoundError, ValueError)):
            transp_reader.verify(filename)

    def test_verify_file_is_not_netcdf(self, transp_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir.joinpath("input.gs2")
        with pytest.raises(ValueError):
            transp_reader.verify(filename)

    @pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
    def test_verify_file_is_not_transp(self, transp_reader, filename):
        """Ensure failure when given a non-transp netcdf file"""
        filename = template_dir.joinpath(filename)
        with pytest.raises(ValueError):
            transp_reader.verify(filename)
