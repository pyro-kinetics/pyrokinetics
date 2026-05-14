<<<<<<< HEAD
=======
import netCDF4 as nc
import numpy as np
>>>>>>> origin/unstable
import pytest

from pyrokinetics import template_dir
from pyrokinetics.kinetics import Kinetics, KineticsReaderTRANSP
from pyrokinetics.species import Species


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

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
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

    @pytest.mark.parametrize("filename", ["jetto.jsp", "scene.cdf"])
    def test_verify_file_is_not_transp(self, transp_reader, filename):
        """Ensure failure when given a non-transp netcdf file"""
        filename = template_dir / filename
        with pytest.raises(ValueError):
            transp_reader.verify_file_type(filename)

    def test_centre_grid_axis_for_centre_grid_quantities(
        self, transp_reader, example_file
    ):
        """Regression: TE/NE/TI/OMEGA live on TRANSP's zone-centre grid X,
        while PLFLX (the source of psi_n) lives on the zone-boundary grid XB.
        The two are offset by half a cell, so each profile must be paired
        with the psi_n axis sampled on its own native grid. Pairing a
        centre-grid quantity with the XB-derived psi_n shifts the curve
        outward by half a cell and over-estimates TE on every interior
        surface."""
        result = transp_reader(example_file)

        with nc.Dataset(example_file) as ds:
            it = -1
            X = ds.variables["X"][it, :].data
            XB = ds.variables["XB"][it, :].data
            PLFLX = ds.variables["PLFLX"][it, :].data
            TE = ds.variables["TE"][it, :].data

        psi_n_xb = (PLFLX - PLFLX[0]) / (PLFLX[-1] - PLFLX[0])
        psi_n_x = np.interp(
            X,
            np.concatenate(([0.0], XB)),
            np.concatenate(([0.0], psi_n_xb)),
        )

        psi_query = 0.5
        te_centre = float(np.interp(psi_query, psi_n_x, TE))
        te_edge = float(np.interp(psi_query, psi_n_xb, TE))

        # The bug, by construction, would return ~te_edge; the fix returns
        # te_centre. Require the centre/edge gap to be large enough that
        # a tight tolerance discriminates between them.
        assert abs(te_centre - te_edge) > 5.0

        te_loader = (
            result.species_data["electron"].get_temp(psi_query).to("eV").magnitude
        )
        assert te_loader == pytest.approx(te_centre, abs=1.0)
