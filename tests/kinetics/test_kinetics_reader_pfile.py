from pyrokinetics.kinetics import Kinetics, KineticsReaderpFile
from pyrokinetics.species import Species
from pyrokinetics import template_dir
import pytest


class TestKineticsReaderpFile:
    @pytest.fixture
    def pfile_reader(self):
        return KineticsReaderpFile()

    @pytest.fixture
    def example_file(self):
        return template_dir / "pfile.txt"

    @pytest.fixture
    def example_geqdsk(self):
        return template_dir / "test.geqdsk"

    def test_read(self, pfile_reader, example_file, example_geqdsk):
        """
        Ensure it can read the example pfile file, and that it produces a Species dict.
        """
        result = pfile_reader(example_file, eq_file=example_geqdsk)
        assert isinstance(result, Kinetics)
        for _, value in result.species_data.items():
            assert isinstance(value, Species)

    def test_verify_file_type(self, pfile_reader, example_file):
        """Ensure verify_file_type completes without throwing an error"""
        pfile_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, pfile_reader, example_geqdsk):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises(FileNotFoundError):
            pfile_reader(filename, eq_file=example_geqdsk)

    def test_read_file_without_geqdsk(self, pfile_reader, example_file):
        """Ensure failure when no GEQDSK file given"""
        with pytest.raises(ValueError):
            pfile_reader(example_file, eq_file=None)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_read_file_is_not_pfile(self, pfile_reader, filename, example_geqdsk):
        """Ensure failure when given a non-pfile netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir / filename
        with pytest.raises(Exception):
            pfile_reader(filename, eq_file=example_geqdsk)

    def test_verify_file_does_not_exist(self, pfile_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            pfile_reader.verify_file_type(filename)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_verify_file_is_not_pfile(self, pfile_reader, filename):
        """Ensure failure when given a non-pfile netcdf file"""
        filename = template_dir / filename
        with pytest.raises(ValueError):
            pfile_reader.verify_file_type(filename)
