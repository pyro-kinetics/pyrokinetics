import pytest

from pyrokinetics import template_dir
from pyrokinetics.equilibrium.equilibrium import read_equilibrium
from pyrokinetics.kinetics import Kinetics, KineticsReaderpFile
from pyrokinetics.kinetics.pfile import unique_species_name
from pyrokinetics.species import Species


class TestKineticsReaderpFile:
    @pytest.fixture
    def pfile_reader(self):
        return KineticsReaderpFile()

    @pytest.fixture
    def example_file(self):
        return template_dir / "pfile.txt"

    @pytest.fixture
    def example_eq(self):
        eq_file = template_dir / "test.geqdsk"
        return read_equilibrium(eq_file)

    def test_read(self, pfile_reader, example_file, example_eq):
        """
        Ensure it can read the example pfile file, and that it produces a Species dict.
        """
        result = pfile_reader(example_file, eq=example_eq)
        assert isinstance(result, Kinetics)
        for _, value in result.species_data.items():
            assert isinstance(value, Species)

    def test_verify_file_type(self, pfile_reader, example_file):
        """Ensure verify_file_type completes without throwing an error"""
        pfile_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, pfile_reader, example_eq):
        """Ensure failure when given a non-existent file"""
        filename = template_dir / "helloworld"
        with pytest.raises(FileNotFoundError):
            pfile_reader(filename, eq=example_eq)

    def test_read_file_without_geqdsk(self, pfile_reader, example_file):
        """Ensure failure when no GEQDSK file given"""
        with pytest.raises(ValueError):
            pfile_reader(example_file, eq=None)

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_read_file_is_not_pfile(self, pfile_reader, filename, example_eq):
        """Ensure failure when given a non-pfile netcdf file

        This could fail for any number of reasons during processing.
        """
        filename = template_dir / filename
        with pytest.raises(Exception):
            pfile_reader(filename, eq=example_eq)

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

    def test_read_multiple_impurities(self, pfile_reader, example_eq, tmp_path):
        """A pFile with two impurities keeps both, rather than the second
        overwriting the first under the shared name "impurity"."""
        lines = (template_dir / "pfile.txt").read_text().splitlines()

        # Duplicate the nz1 profile block as nz2, for a second impurity
        start = next(i for i, line in enumerate(lines) if "psinorm nz1" in line)
        npoints = int(lines[start].split()[0])
        nz2_block = [lines[start].replace("nz1", "nz2")] + lines[
            start + 1 : start + 1 + npoints
        ]

        # Add neon ahead of the existing carbon, so nz1 is neon and nz2 carbon
        species_header = next(
            i for i, line in enumerate(lines) if "N Z A of ION SPECIES" in line
        )
        nspecies = int(lines[species_header].split()[0])
        lines[species_header] = lines[species_header].replace(
            str(nspecies), str(nspecies + 1), 1
        )
        lines.insert(species_header + 1, " 10.000000   10.000000   20.000000")
        lines[species_header:species_header] = nz2_block

        filename = tmp_path / "pfile_two_impurities.txt"
        filename.write_text("\n".join(lines) + "\n")

        with pytest.warns(UserWarning, match="impurity2"):
            result = pfile_reader(filename, eq=example_eq)

        assert "impurity" in result.species_names
        assert "impurity2" in result.species_names

        impurity_charges = sorted(
            float(result.species_data[name].get_charge(0.5).m)
            for name in ("impurity", "impurity2")
        )
        assert impurity_charges == pytest.approx([6.0, 10.0])

    def test_read_missing_impurity_density(self, pfile_reader, example_eq, tmp_path):
        """An impurity listed without an nz profile is read with zero density
        and a warning, rather than crashing."""
        lines = (template_dir / "pfile.txt").read_text().splitlines()

        # Drop the nz1 block, leaving carbon listed in the N Z A block
        start = next(i for i, line in enumerate(lines) if "psinorm nz1" in line)
        npoints = int(lines[start].split()[0])
        del lines[start : start + 1 + npoints]

        filename = tmp_path / "pfile_no_nz1.txt"
        filename.write_text("\n".join(lines) + "\n")

        with pytest.warns(UserWarning, match="nz1"):
            result = pfile_reader(filename, eq=example_eq)

        impurity = result.species_data["impurity"]
        assert float(impurity.get_charge(0.5).m) == pytest.approx(6.0)
        assert float(impurity.get_dens(0.5).m) == 0.0

        # Species order is unchanged by the fix: impurities ahead of the main ion
        assert list(result.species_names) == [
            "electron",
            "impurity",
            "deuterium",
            "deuterium_fast",
        ]


def test_unique_species_name():
    """The first species keeps its name; later ones get the next free index."""
    assert unique_species_name("impurity", {}) == "impurity"

    result = {"impurity": None}
    with pytest.warns(UserWarning, match="impurity2"):
        assert unique_species_name("impurity", result) == "impurity2"

    result = {"impurity": None, "impurity2": None}
    with pytest.warns(UserWarning, match="impurity3"):
        assert unique_species_name("impurity", result) == "impurity3"
