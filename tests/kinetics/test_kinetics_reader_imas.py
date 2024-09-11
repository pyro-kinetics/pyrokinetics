from pyrokinetics.equilibrium.equilibrium import read_equilibrium
from pyrokinetics.kinetics import Kinetics, KineticsReaderIMAS
from pyrokinetics.species import Species
from pyrokinetics import template_dir
import pytest


class TestKineticsReaderIMAS:
    @pytest.fixture
    def imas_reader(self):
        return KineticsReaderIMAS()

    @pytest.fixture
    def example_file(self):
        return template_dir / "core_profiles.h5"

    @pytest.fixture
    def example_eq(self):
        eq_file = template_dir / "test.geqdsk"
        return read_equilibrium(eq_file)

    def test_read(self, imas_reader, example_file, example_eq):
        """
        Ensure it can read the example imas file, and that it produces a Species dict.
        """
        result = imas_reader(example_file, eq=example_eq)
        assert isinstance(result, Kinetics)
        for _, value in result.species_data.items():
            assert isinstance(value, Species)

    def test_verify_file_type(self, imas_reader, example_file):
        """Ensure verify_file_type completes without throwing an error"""
        imas_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, imas_reader, example_eq):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises(FileNotFoundError):
            imas_reader(filename, eq=example_eq)

    def test_read_file_without_geqdsk(self, imas_reader, example_file):
        """Ensure failure when no GEQDSK file given"""
        with pytest.raises(ValueError):
            imas_reader(example_file, eq=None)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_read_file_is_not_imas(self, imas_reader, filename, example_eq):
        """Ensure failure when given a non-imas netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir / filename
        with pytest.raises(Exception):
            imas_reader(filename, eq=example_eq)

    def test_verify_file_does_not_exist(self, imas_reader):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises((FileNotFoundError, ValueError)):
            imas_reader.verify_file_type(filename)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_verify_file_is_not_imas(self, imas_reader, filename):
        """Ensure failure when given a non-imas netcdf file"""
        filename = template_dir / filename
        with pytest.raises(ValueError):
            imas_reader.verify_file_type(filename)
