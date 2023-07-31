from pyrokinetics.kinetics import Kinetics, KineticsReaderTRANSP
from pyrokinetics.species import Species
from pyrokinetics import template_dir
import pytest


class TestKineticsReaderTRANSP:
    @pytest.fixture
    def transp_reader(self):
        return KineticsReaderTRANSP()

    @pytest.fixture
    def example_file(self):
        return template_dir / "transp.cdf"

    def test_read(self, transp_reader, example_file):
        """
        Ensure it can read the example TRANSP file, and that it produces a Species dict.
        """
        result = transp_reader(example_file)
        assert isinstance(result, Kinetics)
        for _, value in result.species_data.items():
            assert isinstance(value, Species)

    def test_verify_file_type(self, transp_reader, example_file):
        """Ensure verify_file_type completes without throwing an error"""
        transp_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, transp_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            transp_reader(filename)

    def test_read_file_is_not_netcdf(self, transp_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir / "input.gs2"
        with pytest.raises(OSError):
            transp_reader(filename)

    @pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
    def test_read_file_is_not_transp(self, transp_reader, filename):
        """Ensure failure when given a non-transp netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir / filename
        with pytest.raises(Exception):
            transp_reader(filename)

    def test_verify_file_does_not_exist(self, transp_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            transp_reader.verify_file_type(filename)

    def test_verify_file_is_not_netcdf(self, transp_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir / "input.gs2"
        with pytest.raises(ValueError):
            transp_reader.verify_file_type(filename)

    @pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
    def test_verify_file_is_not_transp(self, transp_reader, filename):
        """Ensure failure when given a non-transp netcdf file"""
        filename = template_dir / filename
        with pytest.raises(ValueError):
            transp_reader.verify_file_type(filename)
