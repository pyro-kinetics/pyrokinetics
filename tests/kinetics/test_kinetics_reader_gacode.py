from pyrokinetics.kinetics import Kinetics
from pyrokinetics.species import Species
from pyrokinetics import template_dir
import pytest

try:
    from pyrokinetics.kinetics import KineticsReaderGACODE
except ImportError:
    pytest.skip(allow_module_level=True)


class TestKineticsReaderGACODE:
    @pytest.fixture
    def gacode_reader(self):
        return KineticsReaderGACODE()

    @pytest.fixture
    def example_file(self):
        return template_dir / "input.gacode"

    def test_read(self, gacode_reader, example_file):
        """
        Ensure it can read the example GACODE file, and that it produces a Species dict.
        """
        result = gacode_reader(example_file)
        assert isinstance(result, Kinetics)
        for _, value in result.species_data.items():
            assert isinstance(value, Species)

    def test_verify_file_type(self, gacode_reader, example_file):
        """Ensure verify_file_type completes without throwing an error"""
        gacode_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, gacode_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            gacode_reader(filename)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_read_file_is_not_gacode(self, gacode_reader, filename):
        """Ensure failure when given a non-gacode netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir / filename
        with pytest.raises(ValueError):
            gacode_reader(filename)

    def test_verify_file_does_not_exist(self, gacode_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            gacode_reader.verify_file_type(filename)

    def test_verify_file_is_not_netcdf(self, gacode_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir / "input.gs2"
        with pytest.raises(ValueError):
            gacode_reader.verify_file_type(filename)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_verify_file_is_not_gacode(self, gacode_reader, filename):
        """Ensure failure when given a non-gacode netcdf file"""
        filename = template_dir / filename
        with pytest.raises(ValueError):
            gacode_reader.verify_file_type(filename)
