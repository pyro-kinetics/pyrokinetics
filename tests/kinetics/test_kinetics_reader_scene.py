from pyrokinetics.kinetics import Kinetics, KineticsReaderSCENE
from pyrokinetics.species import Species
from pyrokinetics import template_dir
import pytest


class TestKineticsReaderSCENE:
    @pytest.fixture
    def scene_reader(self):
        return KineticsReaderSCENE()

    @pytest.fixture
    def example_file(self):
        return template_dir / "scene.cdf"

    def test_read(self, scene_reader, example_file):
        """
        Ensure it can read the example SCENE file, and that it produces a Species dict.
        """
        result = scene_reader(example_file)
        assert isinstance(result, Kinetics)
        for key, value in result.species_data.items():
            assert key in {"electron", "deuterium", "tritium"}
            assert isinstance(value, Species)

    def test_verify_from_file(self, scene_reader, example_file):
        """Ensure verify completes without throwing an error"""
        scene_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, scene_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        # Sometimes xarray throws FileNotFoundError, sometimes ValueError
        # I can't figure out how to force it one way or the other
        with pytest.raises((FileNotFoundError, ValueError)):
            scene_reader(filename)

    def test_read_file_is_not_netcdf(self, scene_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir / "input.gs2"
        with pytest.raises(ValueError):
            scene_reader(filename)

    @pytest.mark.parametrize("filename", ["transp.cdf", "jetto.jsp"])
    def test_read_file_is_not_scene(self, scene_reader, filename):
        """Ensure failure when given a non-scene netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir / filename
        with pytest.raises(Exception):
            scene_reader(filename)

    def test_verify_file_does_not_exist(self, scene_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            scene_reader.verify_file_type(filename)

    def test_verify_file_is_not_netcdf(self, scene_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir / "input.gs2"
        with pytest.raises(ValueError):
            scene_reader.verify_file_type(filename)

    @pytest.mark.parametrize("filename", ["transp.cdf", "jetto.jsp"])
    def test_verify_file_is_not_scene(self, scene_reader, filename):
        """Ensure failure when given a non-scene netcdf file"""
        filename = template_dir / filename
        with pytest.raises(ValueError):
            scene_reader.verify_file_type(filename)
