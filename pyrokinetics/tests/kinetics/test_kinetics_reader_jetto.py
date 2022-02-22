from pyrokinetics.kinetics import KineticsReaderJETTO
from pyrokinetics.species import Species
from pyrokinetics import template_dir
import pytest


class TestKineticsReaderJETTO:
    @pytest.fixture
    def jetto_reader(self):
        return KineticsReaderJETTO()

    @pytest.fixture
    def example_file(self):
        return template_dir.joinpath("jetto.cdf")

    def test_read(self, jetto_reader, example_file):
        """
        Ensure it can read the example JETTO file, and that it produces a Species dict.
        """
        result = jetto_reader(example_file)
        assert isinstance(result, dict)
        for _, value in result.items():
            assert isinstance(value, Species)

    def test_verify(self, jetto_reader, example_file):
        """Ensure verify completes without throwing an error"""
        jetto_reader.verify(example_file)

    def test_read_file_does_not_exist(self, jetto_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir.joinpath("helloworld")
        with pytest.raises((FileNotFoundError, ValueError)):
            jetto_reader(filename)

    def test_read_file_is_not_netcdf(self, jetto_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir.joinpath("input.gs2")
        with pytest.raises(OSError):
            jetto_reader(filename)

    @pytest.mark.parametrize("filename", ["transp.cdf", "scene.cdf"])
    def test_read_file_is_not_jetto(self, jetto_reader, filename):
        """Ensure failure when given a non-jetto netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir.joinpath(filename)
        with pytest.raises(Exception):
            jetto_reader(filename)

    def test_verify_file_does_not_exist(self, jetto_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir.joinpath("helloworld")
        with pytest.raises((FileNotFoundError, ValueError)):
            jetto_reader.verify(filename)

    def test_verify_file_is_not_netcdf(self, jetto_reader):
        """Ensure failure when given a non-netcdf file"""
        filename = template_dir.joinpath("input.gs2")
        with pytest.raises(ValueError):
            jetto_reader.verify(filename)

    @pytest.mark.parametrize("filename", ["transp.cdf", "scene.cdf"])
    def test_verify_file_is_not_jetto(self, jetto_reader, filename):
        """Ensure failure when given a non-jetto netcdf file"""
        filename = template_dir.joinpath(filename)
        with pytest.raises(ValueError):
            jetto_reader.verify(filename)
