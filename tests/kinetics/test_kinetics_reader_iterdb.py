import numpy as np
import pytest

from pyrokinetics import template_dir
from pyrokinetics.constants import deuterium_mass, electron_mass
from pyrokinetics.equilibrium import read_equilibrium
from pyrokinetics.kinetics import Kinetics, KineticsReaderITERDB, read_kinetics
from pyrokinetics.species import Species


class TestKineticsReaderITERDB:
    @pytest.fixture
    def iterdb_reader(self):
        return KineticsReaderITERDB()

    @pytest.fixture
    def example_file(self):
        return template_dir / "test.iterdb"

    def test_read(self, iterdb_reader, example_file):
        result = iterdb_reader(example_file, use_rhotor_as_rho=True)

        assert isinstance(result, Kinetics)
        assert result.kinetics_type == "ITERDB"
        assert np.array_equal(
            sorted(result.species_names), sorted(["electron", "deuterium"])
        )
        for _, value in result.species_data.items():
            assert isinstance(value, Species)

        electron = result.species_data["electron"]
        assert electron.species_type == "electron"
        assert electron.mass == electron_mass
        assert np.isclose(electron.get_charge(0.25).m, -1.0)
        assert np.isclose(electron.get_temp(0.25).m, 900.0)
        assert np.isclose(electron.get_dens(0.25).m, 4.5e19)
        assert np.isclose(electron.get_angular_velocity(0.25).m, 9000.0)

        ion = result.species_data["deuterium"]
        assert ion.species_type == "deuterium"
        assert ion.mass == deuterium_mass
        assert np.isclose(ion.get_charge(0.25).m, 1.0)
        assert np.isclose(ion.get_temp(0.25).m, 850.0)
        assert np.isclose(ion.get_dens(0.25).m, 4.4e19)
        assert np.isclose(ion.get_angular_velocity(0.25).m, 9000.0)

    @pytest.mark.parametrize("kinetics_type", ["ITERDB", None])
    def test_read_kinetics(self, example_file, kinetics_type):
        result = read_kinetics(
            example_file, kinetics_type, use_rhotor_as_rho=True, main_ion="hydrogen"
        )

        assert result.kinetics_type == "ITERDB"
        assert np.array_equal(
            sorted(result.species_names), sorted(["electron", "hydrogen"])
        )

    def test_read_with_time_index(self, iterdb_reader, example_file):
        result = iterdb_reader(example_file, time_index=0, use_rhotor_as_rho=True)

        assert np.isclose(
            result.species_data["electron"].get_angular_velocity(0.25).m, 9000.0
        )

    def test_read_with_time_and_time_index_fails(self, iterdb_reader, example_file):
        with pytest.raises(ValueError, match="Cannot set both"):
            iterdb_reader(example_file, time_index=0, time=1.0, use_rhotor_as_rho=True)

    def test_read_with_rotation_sign(self, iterdb_reader, example_file):
        result = iterdb_reader(example_file, rotation_sign=-1.0, use_rhotor_as_rho=True)

        assert np.isclose(
            result.species_data["electron"].get_angular_velocity(0.25).m, -9000.0
        )

    def test_read_with_equilibrium_maps_rhotor_to_equilibrium_psi_n(
        self, iterdb_reader, example_file
    ):
        eq = read_equilibrium(template_dir / "test.geqdsk", "GEQDSK")
        result = iterdb_reader(example_file, eq=eq)

        rho_tor_grid = eq["rho_tor"].data
        psi_n_grid = eq["psi_n"].data
        rho_tor_values = np.asarray(
            (
                rho_tor_grid.magnitude
                if hasattr(rho_tor_grid, "magnitude")
                else rho_tor_grid
            ),
            dtype=float,
        )
        psi_n_values = np.asarray(
            psi_n_grid.magnitude if hasattr(psi_n_grid, "magnitude") else psi_n_grid,
            dtype=float,
        )
        psi_n = float(np.interp(0.5, rho_tor_values, psi_n_values))

        electron = result.species_data["electron"]
        ion = result.species_data["deuterium"]

        assert np.isclose(electron.get_temp(psi_n).m, 800.0)
        assert np.isclose(electron.get_dens(psi_n).m, 4.0e19)
        assert np.isclose(electron.get_angular_velocity(psi_n).m, 8000.0)
        assert np.isclose(ion.get_temp(psi_n).m, 750.0)
        assert np.isclose(ion.get_dens(psi_n).m, 3.9e19)

    def test_verify_file_type(self, iterdb_reader, example_file):
        iterdb_reader.verify_file_type(example_file)

    def test_read_file_does_not_exist(self, iterdb_reader):
        filename = template_dir / "helloworld"
        with pytest.raises(FileNotFoundError):
            iterdb_reader(filename, use_rhotor_as_rho=True)

    def test_read_file_without_geqdsk(self, iterdb_reader, example_file):
        with pytest.raises(ValueError, match="Please load an Equilibrium"):
            iterdb_reader(example_file)

    def test_verify_file_is_not_iterdb(self, iterdb_reader):
        filename = template_dir / "input.gs2"
        with pytest.raises(ValueError):
            iterdb_reader.verify_file_type(filename)
