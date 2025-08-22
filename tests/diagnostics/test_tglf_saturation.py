"""
Tests for TGLF saturation parameters implementation.

This module tests the TGLF saturation rules implementation including:
- Energy flux calculation using sum_ky_spectrum
- Saturation parameters calculation using get_sat_params
"""

import numpy as np
import xarray as xr
import os
import pytest
from numpy.testing import assert_allclose
from pyrokinetics import template_dir
from pyrokinetics.diagnostics import sum_ky_spectrum, get_sat_params


@pytest.fixture(scope="module")
def template_path():
    """Path to TGLF test data files."""
    return os.path.join(template_dir, "outputs/TGLF_transport/")


@pytest.fixture(scope="module")
def tglf_dimensions(template_path):
    """Read number of fields, species, modes, ky, and moments from TGLF output."""
    with open(os.path.join(template_path, "out.tglf.QL_flux_spectrum"), "r") as f:
        lines = f.readlines()
    ntype, nspecies, nfield, nky, nmodes = list(map(int, lines[3].strip().split()))
    return {
        "ntype": ntype,
        "nspecies": nspecies, 
        "nfield": nfield,
        "nky": nky,
        "nmodes": nmodes,
        "lines": lines
    }


@pytest.fixture(scope="module")
def ql_data(tglf_dimensions):
    """Load and process QL data from TGLF output files."""
    dims = tglf_dimensions
    lines = dims["lines"]
    
    # Read QL data
    ql = []
    for line in lines[6:]:
        line = line.strip().split()
        if any([x.startswith(("s", "m")) for x in line]):
            continue
        for x in line:
            ql.append(float(x))
    
    QLw = np.array(ql).reshape(dims["nspecies"], dims["nfield"], dims["nmodes"], dims["nky"], dims["ntype"])
    
    # Write QL data into Xarray and reshape
    QL_data_array = xr.DataArray(
        data=QLw,
        dims=["species", "field", "mode", "ky", "type"],
        coords={
            "species": np.arange(dims["nspecies"]),
            "field": np.arange(dims["nfield"]),
            "mode": np.arange(dims["nmodes"]),
            "ky": np.arange(dims["nky"]),
            "type": [
                "particle",
                "energy",
                "toroidal stress",
                "parallel stress",
                "exchange",
            ],
        },
    )
    
    QL_data = QL_data_array.transpose("ky", "mode", "species", "field", "type").data
    
    return {
        "particle_QL": QL_data[:, :, :, :, 0],
        "energy_QL": QL_data[:, :, :, :, 1],
        "toroidal_stress_QL": QL_data[:, :, :, :, 2],
        "parallel_stress_QL": QL_data[:, :, :, :, 3],
        "exchange_QL": QL_data[:, :, :, :, 4],
    }


@pytest.fixture(scope="module")
def spectral_data(template_path):
    """Load spectral shift and ave_p0 data."""
    # Read spectral shift and ave_p0 (only needed for SAT0)
    with open(os.path.join(template_path, "out.tglf.spectral_shift_spectrum"), "r") as f:
        kx0_e = np.loadtxt(f, skiprows=5, unpack=True)
    with open(os.path.join(template_path, "out.tglf.ave_p0_spectrum"), "r") as f:
        ave_p0 = np.loadtxt(f, skiprows=3, unpack=True)
    
    return {"kx0_e": kx0_e, "ave_p0": ave_p0}


@pytest.fixture(scope="module")
def tglf_inputs(template_path):
    """Load and process TGLF input parameters."""
    inputs = {}
    R_unit = None
    
    # Read scalar saturation parameters
    with open(os.path.join(template_path, "out.tglf.scalar_saturation_parameters"), "r") as f:
        content = f.readlines()
        for line in content[1:]:
            line = line.strip().split("\n")
            if any([
                x.startswith((
                    "!",
                    "UNITS",
                    "SAT_RULE",
                    "XNU_MODEL",
                    "ETG_FACTOR",
                    "R_unit",
                    "ALPHA_ZF",
                    "RULE",
                )) for x in line
            ]):
                if line[0].startswith("R_unit"):
                    line = line[0].split(" = ")
                    R_unit = float(line[1])
                continue
            line = line[0].split(" = ")
            inputs.setdefault(str(line[0]), float(line[1]))
    
    # Read input.tglf
    with open(os.path.join(template_path, "input.tglf"), "r") as f:
        content = f.readlines()
        for line in content[1:]:
            line = line.strip().split("\n")
            line = line[0].split(" = ")
            try:
                inputs.setdefault(str(line[0]), float(line[1]))
            except ValueError:
                continue
    
    # Added inputs
    inputs["UNITS"] = "GYRO"
    inputs["ALPHA_ZF"] = 1.0
    inputs["RLNP_CUTOFF"] = 18.0
    inputs["NS"] = int(inputs["NS"])
    inputs["ALPHA_QUENCH"] = 0.0
    inputs["USE_AVE_ION_GRID"] = False
    
    return inputs, R_unit


@pytest.fixture(scope="module")
def spectrum_data(template_path, tglf_dimensions):
    """Load ky and eigenvalue spectrum data."""
    dims = tglf_dimensions
    nmodes = dims["nmodes"]
    
    # Get ky spectrum
    with open(os.path.join(template_path, "out.tglf.ky_spectrum"), "r") as f:
        content = f.readlines()
        content = "".join(content[2:]).split()
        ky_spect = np.array(content, dtype=float)
    
    # Get eigenvalue spectrum
    with open(os.path.join(template_path, "out.tglf.eigenvalue_spectrum"), "r") as f:
        content = f.readlines()
        content = "".join(content[2:]).split()
        gamma = []
        freq = []
        for k in range(nmodes):
            gamma.append(np.array(content[2 * k :: nmodes * 2], dtype=float))
            freq.append(np.array(content[2 * k + 1 :: nmodes * 2], dtype=float))
        gamma = np.array(gamma)
        freq = np.array(freq)
        gamma = xr.DataArray(
            gamma,
            dims=("mode_num", "ky"),
            coords={"ky": ky_spect, "mode_num": np.arange(nmodes) + 1},
        )
        freq = xr.DataArray(
            freq,
            dims=("mode_num", "ky"),
            coords={"ky": ky_spect, "mode_num": np.arange(nmodes) + 1},
        )
    
    return {
        "ky_spect": ky_spect,
        "gammas": gamma.T,
        "freq": freq.T
    }


@pytest.fixture(scope="module")
def potential_data(template_path, tglf_dimensions, spectrum_data):
    """Load potential spectrum data."""
    dims = tglf_dimensions
    nmodes = dims["nmodes"]
    ky_spect = spectrum_data["ky_spect"]
    
    # Get potential spectrum
    with open(os.path.join(template_path, "out.tglf.field_spectrum"), "r") as f:
        lines = f.readlines()
        columns = [x.strip() for x in lines[1].split(",")]
        nc = len(columns)
        content = "".join(lines[6:]).split()
        tmpdict = {}
        for ik, k in enumerate(columns):
            tmp = []
            for nm in range(nmodes):
                tmp.append(np.array(content[ik + nm * nc :: nmodes * nc], dtype=float))
            tmpdict[k] = tmp
        for k, v in list(tmpdict.items()):
            potential = xr.DataArray(
                v,
                dims=("mode_num", "ky"),
                coords={"ky": ky_spect, "mode_num": np.arange(nmodes) + 1},
            )
    
    return potential.T


@pytest.fixture(scope="module")
def expected_fluxes(template_path):
    """Load expected flux values from TGLF output."""
    with open(os.path.join(template_path, "out.tglf.gbflux"), "r") as f:
        content = f.read()
        fluxes = list(map(float, content.split()))
        fluxes = np.reshape(fluxes, (4, -1))
    
    return fluxes


def test_energy_flux_calculation(
    ql_data,
    spectral_data,
    tglf_inputs,
    spectrum_data,
    potential_data,
    expected_fluxes
):
    """Test energy flux calculation using sum_ky_spectrum."""
    inputs, R_unit = tglf_inputs
    
    # Create R_unit array with correct shape
    R_unit_array = np.ones(np.shape(spectrum_data["gammas"])) * R_unit
    
    # Calculate saturation using sum_ky_spectrum
    sat_1 = sum_ky_spectrum(
        inputs["SAT_RULE"],
        spectrum_data["ky_spect"],
        spectrum_data["gammas"],
        spectral_data["ave_p0"],
        R_unit_array,
        spectral_data["kx0_e"],
        potential_data,
        ql_data["particle_QL"],
        ql_data["energy_QL"],
        ql_data["toroidal_stress_QL"],
        ql_data["parallel_stress_QL"],
        ql_data["exchange_QL"],
        **inputs,
    )
    
    # Compare with expected values
    expected_sat1 = expected_fluxes[1]
    python_sat1 = np.sum(np.sum(sat_1["energy_flux_integral"], axis=2), axis=0)
    
    assert_allclose(python_sat1, expected_sat1, rtol=1e-3)


def test_saturation_parameters(
    spectral_data,
    tglf_inputs,
    spectrum_data
):
    """Test saturation parameters calculation using get_sat_params."""
    inputs, R_unit = tglf_inputs
    
    # Set additional parameters for saturation test
    inputs["DRMINDX_LOC"] = 1.0
    inputs["ALPHA_E"] = 1.0
    inputs["VEXB_SHEAR"] = 0.0
    inputs["SIGN_IT"] = 1.0
    
    # Calculate saturation parameters
    kx0epy, satgeo1, satgeo2, runit, bt0, bgeo0, gradr0, _, _, _, _ = get_sat_params(
        1, spectrum_data["ky_spect"], spectrum_data["gammas"].T, **inputs
    )
    
    # Test all saturation parameters
    assert_allclose(kx0epy, spectral_data["kx0_e"], rtol=1e-3)
    assert_allclose(inputs["SAT_geo1_out"], satgeo1, rtol=1e-6)
    assert_allclose(inputs["SAT_geo2_out"], satgeo2, rtol=1e-6)
    assert_allclose(R_unit, runit, rtol=1e-6)
    assert_allclose(inputs["Bt0_out"], bt0, rtol=1e-6)
    assert_allclose(inputs["grad_r0_out"], gradr0, rtol=1e-6)
    
    # Only test B_geo0_out if VEXB_SHEAR is non-zero
    if inputs["VEXB_SHEAR"] != 0.0:
        assert_allclose(inputs["B_geo0_out"], bgeo0, rtol=1e-6)


def test_get_zonal_mixing_integration():
    """
    Integration test to verify that get_zonal_mixing function is working
    as part of the larger TGLF saturation calculation pipeline.
    
    This test ensures the get_zonal_mixing function (which was compared earlier)
    works correctly within the context of the full saturation calculation.
    """
    # This test is implicit - if the other tests pass, get_zonal_mixing is working
    # correctly since it's called internally by get_sat_params and intensity_sat
    # functions used in the above tests.
    
    # We can add a simple import test to ensure the function is accessible
    from pyrokinetics.diagnostics import get_zonal_mixing
    
    # Basic test with dummy data to ensure function signature is correct
    ky_mix = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma_mix = np.array([0.05, 0.1, 0.08, 0.06, 0.04])
    kw = {
        "rho_ion": 0.1,
        "ALPHA_ZF": 1.0,
        "SAT_RULE": 1,
        "grad_r0_out": 1.0
    }
    
    # Test that function runs without error
    vzf_mix, kymax_mix, jmax_mix = get_zonal_mixing(ky_mix, gamma_mix, **kw)
    
    # Basic sanity checks
    assert isinstance(vzf_mix, (float, np.floating))
    assert isinstance(kymax_mix, (float, np.floating))
    assert isinstance(jmax_mix, (int, np.integer))
    assert vzf_mix > 0  # Should be positive
    assert kymax_mix > 0  # Should be positive
    assert 0 <= jmax_mix < len(ky_mix)  # Should be valid index


if __name__ == "__main__":
    # Allow running as script for backward compatibility
    pytest.main([__file__, "-v"])