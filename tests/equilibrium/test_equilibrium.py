import warnings
from itertools import product
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from omas import cocos_transform
from pyrokinetics import template_dir
from pyrokinetics.equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
    supported_equilibrium_types,
)
from pyrokinetics.normalisation import ureg as units

# Test using a known equilibrium, without dependence on any specific file type.
# This is not a valid solution to the Grad-Shafranov equation!
# Expect same results regardless of input units or COCOS conventions
# TODO use hypothesis library here?


@pytest.fixture(scope="module")
def expected() -> Dict[str, Any]:
    """
    Defines expected coords, data_vars, attrs, and units for both the ``circular_eq``
    fixture, which builds an ``Equilibrium`` directly from these values, and the
    ``parametrized_eq``, which modifies the inputs to simulate a variety of input
    units and COCOS types.

    Returns
    -------
    Dict[str, Any]
    """
    # Define default units
    len_units = units.m
    psi_units = units.weber
    F_units = units.m * units.tesla
    FF_prime_units = F_units**2 / units.weber
    p_units = units.pascal
    p_prime_units = units.pascal / units.weber
    q_units = units.dimensionless
    B_units = units.tesla
    I_units = units.ampere

    # Create set of COCOS 11 args
    n_R = 101
    n_Z = 121
    n_psi = 61
    R_max = 5.0
    R_min = 1.0
    Z_max = 3.0
    Z_min = -1.0

    R = np.linspace(R_min, R_max, n_R)
    Z = np.linspace(-2.0, 2.0, n_Z)
    R_axis = 0.5 * (R_max + R_min)
    Z_axis = 0.5 * (Z_max + Z_min)
    radial_grid = np.sqrt((R - R_axis)[:, np.newaxis] ** 2 + (Z - Z_axis) ** 2)
    psi_offset = -5.1
    psi_RZ = 2 * np.pi * (radial_grid) + psi_offset
    psi_axis = np.min(psi_RZ)
    psi_lcfs = np.min(psi_RZ[0])
    a_minor = 0.5 * (R_max - R_min)

    psi = np.linspace(psi_axis, psi_lcfs, n_psi)
    F = psi**2
    F_prime = 2 * psi
    FF_prime = F * F_prime
    p = 3000 + 100 * psi
    p_prime = 100 * np.ones(n_psi)
    q = np.linspace(2.0, 7.0, n_psi)
    R_major = R_axis * np.ones(n_psi)
    r_minor = np.linspace(0, a_minor, n_psi)
    Z_mid = Z_axis * np.ones(n_psi)
    B_0 = 2.5
    I_p = 1e6

    # Get vals we can use to test spline functions
    # See tests 'test_circular_eq_func' and 'test_circular_eq_psi_n'
    indices = [0, n_psi // 4, n_psi // 2, 3 * n_psi // 4, -1]
    psi_n_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    psi_vals = psi[indices]
    F_vals = F[indices]
    F_prime_vals = F_prime[indices]
    FF_prime_vals = FF_prime[indices]
    p_vals = p[indices]
    p_prime_vals = p_prime[indices]
    q_vals = q[indices]
    q_prime_vals = np.ones(3) * (q[-1] - q[0]) / (psi[-1] - psi[0])
    R_major_vals = R_major[indices]
    R_major_prime_vals = np.zeros(3)
    r_minor_vals = r_minor[indices]
    r_minor_prime_vals = np.ones(3) * (r_minor[-1] - r_minor[0]) / (psi[-1] - psi[0])
    Z_mid_vals = Z_mid[indices]
    Z_mid_prime_vals = np.zeros(3)
    rho_vals = r_minor_vals / a_minor

    return dict(
        len_units=len_units,
        psi_units=psi_units,
        F_units=F_units,
        FF_prime_units=FF_prime_units,
        p_units=p_units,
        p_prime_units=p_prime_units,
        q_units=q_units,
        B_units=B_units,
        I_units=I_units,
        n_R=n_R,
        n_Z=n_Z,
        n_psi=n_psi,
        R=R * len_units,
        Z=Z * len_units,
        psi_RZ=psi_RZ * psi_units,
        psi_axis=psi_axis * psi_units,
        psi_lcfs=psi_lcfs * psi_units,
        a_minor=a_minor * len_units,
        psi=psi * psi_units,
        F=F * F_units,
        F_prime=F_prime * F_units / psi_units,
        FF_prime=FF_prime * FF_prime_units,
        p=p * p_units,
        p_prime=p_prime * p_prime_units,
        q=q * q_units,
        R_major=R_major * len_units,
        r_minor=r_minor * len_units,
        Z_mid=Z_mid * len_units,
        B_0=B_0 * B_units,
        I_p=I_p * I_units,
        R_axis=R_major[0] * len_units,
        Z_axis=Z_mid[0] * len_units,
        dR=((R[-1] - R[0]) / (n_R - 1)) * len_units,
        dZ=((Z[-1] - Z[0]) / (n_Z - 1)) * len_units,
        indices=indices,
        psi_n_vals=psi_n_vals * units.dimensionless,
        psi_vals=psi_vals * psi_units,
        F_vals=F_vals * F_units,
        F_prime_vals=F_prime_vals * F_units / psi_units,
        FF_prime_vals=FF_prime_vals * FF_prime_units,
        p_vals=p_vals * p_units,
        p_prime_vals=p_prime_vals * p_prime_units,
        q_vals=q_vals * q_units,
        q_prime_vals=q_prime_vals * q_units / psi_units,
        R_major_vals=R_major_vals * len_units,
        R_major_prime_vals=R_major_prime_vals * len_units / psi_units,
        r_minor_vals=r_minor_vals * len_units,
        r_minor_prime_vals=r_minor_prime_vals * len_units / psi_units,
        Z_mid_vals=Z_mid_vals * len_units,
        Z_mid_prime_vals=Z_mid_prime_vals * len_units / psi_units,
        rho_vals=rho_vals * units.dimensionless,
    )


@pytest.fixture(scope="module")
def circular_eq(expected) -> Equilibrium:
    """
    Builds an ``Equilibrium`` directly from the ``expected`` fixture. The values
    contained within the ``Equilibrium`` should be compared directly to the ``expected``
    values.

    Returns
    -------
    Equilibrium

    Raises
    ------
    RuntimeError
        The inputs should be interpretted as COCOS 11. If ``Equilibrium`` throws an
        ``EquilibriumCOCOSWarning``, then it has unintentionally modified the inputs.
        If this occurs, an error is raised.
    """
    warnings.simplefilter("error", EquilibriumCOCOSWarning)
    try:
        eq = Equilibrium(
            R=expected["R"],
            Z=expected["Z"],
            psi_RZ=expected["psi_RZ"],
            psi=expected["psi"],
            F=expected["F"],
            FF_prime=expected["FF_prime"],
            p=expected["p"],
            p_prime=expected["p_prime"],
            q=expected["q"],
            R_major=expected["R_major"],
            r_minor=expected["r_minor"],
            Z_mid=expected["Z_mid"],
            psi_lcfs=expected["psi_lcfs"],
            a_minor=expected["a_minor"],
            B_0=expected["B_0"],
            I_p=expected["I_p"],
        )
    except EquilibriumCOCOSWarning as w:
        raise RuntimeError("circular_eq inputs are not COCOS 11") from w
    finally:
        warnings.simplefilter("default", EquilibriumCOCOSWarning)
    return eq


_units = [units.m, units.cm]
_cocos = list(range(1, 9)) + list(range(11, 19))
_params = [{"len_units": u, "cocos": c} for u, c in product(_units, _cocos)]


def _ids(param):
    unit_part = "cm" if param["len_units"] == units.cm else "m"
    cocos_part = param["cocos"]
    return f"{unit_part}_{cocos_part}"


@pytest.fixture(params=_params, ids=_ids, scope="module")
def parametrized_eq(request, expected):
    """
    Builds an ``Equilibrium`` using modified values from the ``expected`` fixture.
    Parametrized over the length units scale and the input COCOS. The resulting
    ``Equilibrium`` should be compared directly with the values in the ``expected``
    fixture, as ``Equilibrium`` is expected to rectify differences in units or COCOS
    automatically.

    Returns
    -------
    Equilibrium
    """
    # TODO: Currently this sets the optional cocos argument to the Equilibrium
    # initialiser, which asserts that the inputs are of a given COCOS. It would be
    # preferable to instead test the automatic COCOS detection.

    # TODO: The inputs are modified using omas.cocos_transform to generate inputs in
    # a difference COCOS. As this is the same function used within Equilibrium, the
    # test is effectively circular, and may not detect if there is a defect in the
    # omas function. The modification should be done manually in this fixture.

    # Determine units and multiplicative factors
    cocos = request.param["cocos"]
    cocos_factors = cocos_transform(11, cocos)

    len_units = request.param["len_units"] / expected["len_units"]
    len_factor = (1.0 if len_units == units.dimensionless else 100.0) * len_units

    psi_units = 1.0 if cocos >= 10 else 1.0 / units.radian
    psi_factor = cocos_factors["PSI"] * psi_units
    F_factor = cocos_factors["F"] * len_factor
    FF_prime_factor = cocos_factors["F_FPRIME"] * len_factor**2 / psi_units
    p_prime_factor = cocos_factors["PPRIME"] / psi_units
    q_factor = cocos_factors["Q"]
    B_factor = cocos_factors["BT"]
    I_factor = cocos_factors["IP"]

    eq = Equilibrium(
        R=expected["R"] * len_factor,
        Z=expected["Z"] * len_factor,
        psi_RZ=expected["psi_RZ"] * psi_factor,
        psi=expected["psi"] * psi_factor,
        F=expected["F"] * F_factor,
        FF_prime=expected["FF_prime"] * FF_prime_factor,
        p=expected["p"],
        p_prime=expected["p_prime"] * p_prime_factor,
        q=expected["q"] * q_factor,
        R_major=expected["R_major"] * len_factor,
        r_minor=expected["r_minor"] * len_factor,
        Z_mid=expected["Z_mid"] * len_factor,
        psi_lcfs=expected["psi_lcfs"] * psi_factor,
        a_minor=expected["a_minor"] * len_factor,
        B_0=expected["B_0"] * B_factor,
        I_p=expected["I_p"] * I_factor,
        clockwise_phi=(cocos % 2 == 0),
        cocos=cocos,
    )
    return eq


def test_parametrized_eq_dims(parametrized_eq, expected):
    dims = parametrized_eq.dims
    assert dims["R_dim"] == expected["n_R"]
    assert dims["Z_dim"] == expected["n_Z"]
    assert dims["psi_dim"] == expected["n_psi"]


def test_parametrized_eq_coords(parametrized_eq, expected):
    coords = parametrized_eq.coords
    # Check dims/shapes
    assert coords["R"].dims == ("R_dim",)
    assert coords["Z"].dims == ("Z_dim",)
    assert coords["psi"].dims == ("psi_dim",)
    assert_array_equal(coords["R"].shape, (expected["n_R"],))
    assert_array_equal(coords["Z"].shape, (expected["n_Z"],))
    assert_array_equal(coords["psi"].shape, (expected["n_psi"],))
    # Check units
    # Expect meters are used regardless of whether the object was created with cm
    # Expect psi in webers regardless of COCOS
    assert coords["R"].data.units == expected["len_units"]
    assert coords["Z"].data.units == expected["len_units"]
    assert coords["psi"].data.units == expected["psi_units"]
    # check values
    assert_allclose(coords["R"].data.magnitude, expected["R"].magnitude)
    assert_allclose(coords["Z"].data.magnitude, expected["Z"].magnitude)
    assert_allclose(coords["psi"].data.magnitude, expected["psi"].magnitude)


def test_parametrized_eq_data_vars(parametrized_eq, expected):
    data_vars = parametrized_eq.data_vars
    # Check dims
    assert data_vars["psi_RZ"].dims == ("R_dim", "Z_dim")
    assert data_vars["F"].dims == ("psi_dim",)
    assert data_vars["FF_prime"].dims == ("psi_dim",)
    assert data_vars["p"].dims == ("psi_dim",)
    assert data_vars["p_prime"].dims == ("psi_dim",)
    assert data_vars["q"].dims == ("psi_dim",)
    assert data_vars["R_major"].dims == ("psi_dim",)
    assert data_vars["r_minor"].dims == ("psi_dim",)
    assert data_vars["Z_mid"].dims == ("psi_dim",)
    assert_array_equal(data_vars["psi_RZ"].shape, (expected["n_R"], expected["n_Z"]))
    assert_array_equal(data_vars["F"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["FF_prime"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["p"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["p_prime"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["q"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["R_major"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["r_minor"].shape, (expected["n_psi"],))
    assert_array_equal(data_vars["Z_mid"].shape, (expected["n_psi"],))

    # Check units
    assert data_vars["psi_RZ"].data.units == expected["psi_units"]
    assert data_vars["F"].data.units == expected["F_units"]
    assert data_vars["FF_prime"].data.units == expected["FF_prime_units"]
    assert data_vars["p"].data.units == expected["p_units"]
    assert data_vars["p_prime"].data.units == expected["p_prime_units"]
    assert data_vars["q"].data.units == expected["q_units"]
    assert data_vars["R_major"].data.units == expected["len_units"]
    assert data_vars["r_minor"].data.units == expected["len_units"]
    assert data_vars["Z_mid"].data.units == expected["len_units"]
    # Check values
    assert_allclose(data_vars["psi_RZ"].data.magnitude, expected["psi_RZ"].magnitude)
    assert_allclose(data_vars["F"].data.magnitude, expected["F"].magnitude)
    assert_allclose(
        data_vars["FF_prime"].data.magnitude, expected["FF_prime"].magnitude
    )
    assert_allclose(data_vars["p"].data.magnitude, expected["p"].magnitude)
    assert_allclose(data_vars["p_prime"].data.magnitude, expected["p_prime"].magnitude)
    assert_allclose(data_vars["q"].data.magnitude, expected["q"].magnitude)
    assert_allclose(data_vars["R_major"].data.magnitude, expected["R_major"].magnitude)
    assert_allclose(data_vars["r_minor"].data.magnitude, expected["r_minor"].magnitude)
    assert_allclose(data_vars["Z_mid"].data.magnitude, expected["Z_mid"].magnitude)


def test_parametrized_eq_attrs(parametrized_eq, expected):
    eq = parametrized_eq
    # Check units
    assert eq.R_axis.units == expected["len_units"]
    assert eq.Z_axis.units == expected["len_units"]
    assert eq.psi_axis.units == expected["psi_units"]
    assert eq.psi_lcfs.units == expected["psi_units"]
    assert eq.a_minor.units == expected["len_units"]
    assert eq.dR.units == expected["len_units"]
    assert eq.dZ.units == expected["len_units"]
    assert eq.B_0.units == expected["B_units"]
    assert eq.I_p.units == expected["I_units"]
    # Check values
    assert np.isclose(eq.R_axis, expected["R_axis"])
    assert np.isclose(eq.Z_axis, expected["Z_axis"])
    assert np.isclose(eq.psi_axis, expected["psi_axis"])
    assert np.isclose(eq.psi_lcfs, expected["psi_lcfs"])
    assert np.isclose(eq.a_minor, expected["a_minor"])
    assert np.isclose(eq.dR, expected["dR"])
    assert np.isclose(eq.dZ, expected["dZ"])
    assert np.isclose(eq.B_0, expected["B_0"])
    assert np.isclose(eq.I_p, expected["I_p"])
    assert eq.eq_type == "None"


@pytest.mark.parametrize(
    "func",
    (
        "psi",
        "F",
        "F_prime",
        "FF_prime",
        "p",
        "p_prime",
        "q",
        "q_prime",
        "R_major",
        "R_major_prime",
        "r_minor",
        "r_minor_prime",
        "Z_mid",
        "Z_mid_prime",
        "rho",
    ),
)
def test_circular_eq_func(circular_eq, expected, func):
    eq = circular_eq
    eq_func = getattr(eq, func)
    expected_vals = expected[f"{func}_vals"]
    assert eq_func(0).units == expected_vals.units
    for psi_n, expect in zip(expected["psi_n_vals"], expected_vals):
        assert np.isclose(eq_func(psi_n), expect)


def test_circular_eq_psi_n(circular_eq, expected):
    eq = circular_eq
    assert eq.psi_n(0).units == expected["psi_n_vals"].units
    for psi, expect in zip(expected["psi_vals"], expected["psi_n_vals"]):
        assert np.isclose(eq.psi_n(psi), expect)


@pytest.mark.parametrize(
    "key",
    [
        "R",
        "Z",
        "psi_RZ",
        "psi",
        "F",
        "FF_prime",
        "p",
        "p_prime",
        "q",
        "R_major",
        "r_minor",
        "Z_mid",
        "psi_lcfs",
        "a_minor",
        "B_0",
        "I_p",
    ],
)
def test_equilibrium_bad_units(expected, key):
    """Test to ensure Equilibrium raises an exception when given incorrect units"""
    new_args = {}
    for k in expected:
        if key == k:
            new_args[k] = expected[k] * units.s
        else:
            new_args[k] = expected[k]
    with pytest.raises(Exception):
        Equilibrium(
            R=new_args["R"],
            Z=new_args["Z"],
            psi_RZ=new_args["psi_RZ"],
            psi=new_args["psi"],
            F=new_args["F"],
            FF_prime=new_args["FF_prime"],
            p=new_args["p"],
            p_prime=new_args["p_prime"],
            q=new_args["q"],
            R_major=new_args["R_major"],
            r_minor=new_args["r_minor"],
            Z_mid=new_args["Z_mid"],
            psi_lcfs=new_args["psi_lcfs"],
            a_minor=new_args["a_minor"],
            B_0=new_args["B_0"],
            I_p=new_args["I_p"],
        )


def test_circular_eq_flux_surface(circular_eq, expected):
    fs = circular_eq.flux_surface(0.5)
    radius = np.hypot(fs["R"] - fs.R_major, fs["Z"] - fs.Z_mid).data.magnitude
    R_max = expected["R"][-1].magnitude
    R_min = expected["R"][0].magnitude
    R_width = 0.5 * (R_max - R_min)
    expected = 0.5 * R_width * np.ones(len(fs["R"]))
    assert_allclose(radius, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "quantity,normalised",
    product(
        [
            "F",
            "FF_prime",
            "p",
            "p_prime",
            "q",
            "R_major",
            "r_minor",
            "Z_mid",
        ],
        [True, False],
    ),
)
def test_circular_eq_plot(circular_eq, quantity, normalised):
    eq = circular_eq
    psi = eq["psi_n" if normalised else "psi"]
    # Test plot with no provided axes, provide kwargs
    ax = eq.plot(quantity, psi_n=normalised, label="plot 1")
    # Plot again on same ax with new label
    ax = eq.plot(quantity, ax=ax, psi_n=normalised, label="plot_2")
    # Test correct labels
    assert psi.long_name in ax.get_xlabel()
    assert eq[quantity].long_name in ax.get_ylabel()
    # Ensure the correct data is plotted
    for line in ax.lines:
        assert_allclose(line.get_xdata(), psi.data.magnitude)
        assert_allclose(line.get_ydata(), eq[quantity].data.magnitude)
    # Remove figure so it doesn't sit around in memory
    plt.close(ax.get_figure())


def test_circular_eq_plot_bad_quantity(circular_eq):
    with pytest.raises(ValueError):
        circular_eq.plot("hello world")


def test_circular_eq_plot_quantity_on_wrong_grid(circular_eq):
    with pytest.raises(ValueError):
        circular_eq.plot("psi_RZ")


def test_circular_eq_plot_contour(circular_eq):
    eq = circular_eq
    # Test plot with no provided axes, provide kwargs
    ax = eq.contour(levels=50)
    # Plot again on same ax
    ax = eq.contour(ax=ax)
    # Test correct labels
    assert eq["R"].long_name in ax.get_xlabel()
    assert eq["Z"].long_name in ax.get_ylabel()
    # Remove figure so it doesn't sit around in memory
    plt.close(ax.get_figure())


@pytest.mark.parametrize(
    "quantity",
    [
        "R",
        "Z",
        "B_poloidal",
    ],
)
def test_circular_eq_flux_surface_plot(circular_eq, quantity):
    fs = circular_eq.flux_surface(0.5)
    # Test plot with no provided axes, provide kwargs
    ax = fs.plot(quantity, label="plot 1")
    # Plot again on same ax with new label
    ax = fs.plot(quantity, ax=ax, label="plot_2")
    # Test correct labels
    assert fs["theta"].long_name in ax.get_xlabel()
    assert fs[quantity].long_name in ax.get_ylabel()
    # Ensure the correct data is plotted
    for line in ax.lines:
        assert_allclose(line.get_xdata(), fs["theta"].data.magnitude)
        assert_allclose(line.get_ydata(), fs[quantity].data.magnitude)
    # Remove figure so it doesn't sit around in memory
    plt.close(ax.get_figure())


def test_circular_eq_flux_surface_plot_bad_quantity(circular_eq):
    fs = circular_eq.flux_surface(0.7)
    with pytest.raises(ValueError):
        fs.plot("hello world")


def test_circular_eq_flux_surface_plot_path(circular_eq):
    fs = circular_eq.flux_surface(0.5)
    # Test plot with no provided axes, provide kwargs
    ax = fs.plot_path(label="plot 1")
    # Plot again on same ax with new label
    ax = fs.plot_path(ax=ax, label="plot_2")
    # Test correct labels
    assert fs["R"].long_name in ax.get_xlabel()
    assert fs["Z"].long_name in ax.get_ylabel()
    # Ensure the correct data is plotted
    for line in ax.lines:
        assert_allclose(line.get_xdata(), fs["R"].data.magnitude)
        assert_allclose(line.get_ydata(), fs["Z"].data.magnitude)


def test_circular_eq_netcdf_round_trip(tmp_path, circular_eq):
    eq = circular_eq
    dir_ = tmp_path / "circular_eq_netcdf_round_trip"
    dir_.mkdir()
    file_ = dir_ / "my_netcdf.nc"
    eq.to_netcdf(file_)
    eq2 = read_equilibrium(file_)
    # Test coords
    for k, v in eq.coords.items():
        assert k in eq2.coords
        assert_allclose(v.data.magnitude, eq2[k].data.magnitude)
        assert v.data.units == eq2[k].data.units
    # Test data vars
    for k, v in eq.data_vars.items():
        assert k in eq2.data_vars
        assert_allclose(v.data.magnitude, eq2[k].data.magnitude)
        assert v.data.units == eq2[k].data.units
    # Test attributes
    for k, v in eq.attrs.items():
        if hasattr(v, "magnitude"):
            assert np.isclose(v, eq2.attrs[k])
            assert getattr(eq, k).units == getattr(eq2, k).units
        else:
            assert v == eq2.attrs[k]


@pytest.mark.parametrize(
    "filename, eq_type",
    [
        ("transp_eq.cdf", "TRANSP"),
        ("transp_eq.geqdsk", "GEQDSK"),
        ("test.geqdsk", "GEQDSK"),
    ],
)
def test_filetype_inference(filename, eq_type):
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    eq = read_equilibrium(template_dir / filename)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    assert eq.eq_type == eq_type


def test_supported_equilibrium_types():
    eq_types = supported_equilibrium_types()
    assert "GEQDSK" in eq_types
    assert "TRANSP" in eq_types
    assert "Pyrokinetics" in eq_types
