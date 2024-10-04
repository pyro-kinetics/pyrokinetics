from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryFourierGENE
from pyrokinetics.normalisation import SimulationNormalisation
from pyrokinetics.equilibrium import read_equilibrium
from pyrokinetics.units import ureg

import numpy as np
import pytest

atol = 1e-2
rtol = 1e-3


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    n_moments = 32

    cN = np.array([1.0, *[0.0] * (n_moments - 1)])

    sN = np.array([*[0.0] * n_moments])

    lg = LocalGeometryFourierGENE(
        {
            "cN": cN,
            "sN": sN,
            "a_minor": 1.0,
            "Rmaj": 0.0,
            "Z0": 0.0,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    np.testing.assert_allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation(generate_miller):
    length = 129
    theta = np.linspace(0.0, 2 * np.pi, length)

    Rmaj = 3.0
    elongation = 5.0
    miller = generate_miller(
        theta=theta, kappa=elongation, delta=0.0, Rmaj=Rmaj, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    R, Z = fourier.get_flux_surface(theta)
    lref = fourier.Rmaj.units

    assert np.isclose(np.min(R), 2.0 * lref, atol=atol)
    assert np.isclose(np.max(R), 4.0 * lref, atol=atol)
    assert np.isclose(np.min(Z), -5.0 * lref, atol=atol)
    assert np.isclose(np.max(Z), 5.0 * lref, atol=atol)


def test_flux_surface_triangularity(generate_miller):
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=1.0, delta=0.5, Rmaj=3.0, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)
    lref = fourier.Rmaj.units

    R, Z = fourier.get_flux_surface(fourier.theta_eq)

    assert np.isclose(np.min(R), 2.0 * lref, atol=atol)
    assert np.isclose(np.max(R), 4.0 * lref, atol=atol)
    assert np.isclose(np.min(Z), -1.0 * lref, atol=atol)
    assert np.isclose(np.max(Z), 1.0 * lref, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 2.5 * lref, atol=atol)
    assert np.isclose(Z[top_corner], 1.0 * lref, atol=atol)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 2.5 * lref, atol=atol)
    assert np.isclose(Z[bottom_corner], -1.0 * lref, atol=atol)


def test_flux_surface_long_triangularity(generate_miller):
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=2.0, delta=0.5, Rmaj=1.0, rho=2.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)
    lref = fourier.Rmaj.units

    high_res_theta = np.linspace(-np.pi, np.pi, length)
    R, Z = fourier.get_flux_surface(high_res_theta)

    assert np.isclose(np.min(R), -1.0 * lref, atol=atol)
    assert np.isclose(np.max(R), 3.0 * lref, atol=atol)
    assert np.isclose(np.min(Z), -4.0 * lref, atol=atol)
    assert np.isclose(np.max(Z), 4.0 * lref, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 0.0 * lref, atol=atol)

    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 0.0 * lref, atol=atol)


def test_default_bunit_over_b0(generate_miller):
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)
    miller = generate_miller(theta)

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    assert np.isclose(fourier.get_bunit_over_b0(), 1.0141851056742153)


@pytest.mark.parametrize(
    ["parameters", "expected"],
    [
        (
            {"kappa": 1.0, "delta": 0.0, "s_kappa": 0.0, "s_delta": 0.0, "shift": 0.0},
            lambda theta: np.ones(theta.shape),
        ),
        (
            {"kappa": 1.0, "delta": 0.0, "s_kappa": 1.0, "s_delta": 0.0, "shift": 0.0},
            lambda theta: 1.0 / (np.sin(theta) ** 2 + 1),
        ),
        (
            {"kappa": 2.0, "delta": 0.5, "s_kappa": 0.5, "s_delta": 0.2, "shift": 0.1},
            lambda theta: 2.0
            * np.sqrt(
                0.25
                * (0.523598775598299 * np.cos(theta) + 1) ** 2
                * np.sin(theta + 0.523598775598299 * np.sin(theta)) ** 2
                + np.cos(theta) ** 2
            )
            / (
                2.0
                * (0.585398163397448 * np.cos(theta) + 0.5)
                * np.sin(theta)
                * np.sin(theta + 0.523598775598299 * np.sin(theta))
                + 0.2 * np.cos(theta)
                + 2.0 * np.cos(0.523598775598299 * np.sin(theta))
            ),
        ),
    ],
)
def test_grad_r(generate_miller, parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 129
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(theta, dict=parameters)

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    np.testing.assert_allclose(
        ureg.Quantity(fourier.get_grad_r(theta=fourier.theta_eq)).magnitude,
        expected(theta),
        atol=atol,
    )


def test_load_from_eq():
    """Golden answer test"""

    norms = SimulationNormalisation("test_load_from_eq_fouriergene")
    eq = read_equilibrium(template_dir / "test.geqdsk", "GEQDSK")

    fourier = LocalGeometryFourierGENE()
    fourier.from_global_eq(eq, 0.5, norms)

    assert fourier["local_geometry"] == "FourierGENE"

    units = norms.units

    expected = {
        "B0": 2.197104321877944 * units.tesla,
        "rho": 0.6847974215474699 * norms.lref,
        "Rmaj": 1.8498509607744338 * norms.lref,
        "a_minor": 1.5000747773827081 * units.meter,
        "beta_prime": -0.9189081293324618 * norms.bref**2 * norms.lref**-1,
        "bt_ccw": 1 * units.dimensionless,
        "bunit_over_b0": 3.5737590048481054 * units.dimensionless,
        "dpsidr": 1.874010706550275 * units.tesla * units.meter,
        "Fpsi": 6.096777229999999 * units.tesla * units.meter,
        "ip_ccw": 1 * units.dimensionless,
        "q": 4.29996157 * units.dimensionless,
        "shat": 0.7706147138551124 * units.dimensionless,
        "cN": [
            1.10827623e00,
            -5.30195594e-02,
            -5.73146297e-01,
            1.06053809e-01,
            2.01208245e-01,
            -9.13165947e-02,
            -6.42359294e-02,
            6.13266100e-02,
            1.12528646e-02,
            -3.45849942e-02,
            6.62203349e-03,
            1.60792526e-02,
            -9.81697677e-03,
            -5.30272581e-03,
            7.68178186e-03,
            -2.02630146e-05,
            -4.64347808e-03,
            1.79327331e-03,
            2.02415118e-03,
            -1.98798740e-03,
            -5.06056953e-04,
            1.38667639e-03,
            -3.25603489e-04,
            -8.53108926e-04,
            4.22477892e-04,
            2.14166033e-04,
            -5.39729148e-04,
            -9.69026453e-05,
            2.17492554e-04,
            -1.94840528e-04,
            -2.28742844e-04,
            4.41312850e-05,
        ]
        * norms.lref,
        "sN": [
            0.00000000e00,
            -4.32975896e-06,
            9.03004175e-05,
            -2.34144020e-05,
            -6.47799889e-05,
            3.64033670e-05,
            2.94531152e-05,
            -3.64123777e-05,
            -7.34591313e-06,
            2.42953644e-05,
            -7.60372114e-06,
            -1.57993781e-05,
            8.49253786e-06,
            3.59872685e-06,
            -1.10889325e-05,
            -1.76773820e-06,
            4.19216834e-06,
            -5.10700437e-06,
            -5.66699265e-06,
            8.16710401e-07,
            -1.68437420e-06,
            -5.47790459e-06,
            -2.45904134e-06,
            -1.21640368e-06,
            -4.03232065e-06,
            -3.97966506e-06,
            -2.31837290e-06,
            -3.21063594e-06,
            -4.12636128e-06,
            -3.44535012e-06,
            -3.31754195e-06,
            -4.09954249e-06,
        ]
        * norms.lref,
    }

    for key, value in expected.items():
        np.testing.assert_allclose(
            fourier[key].to(value.units).magnitude,
            value.magnitude,
            rtol=rtol,
            atol=atol,
        )

    fourier.R, fourier.Z = fourier.get_flux_surface(fourier.theta_eq)
    assert np.isclose(
        min(fourier.R).to("meter"),
        1.7476563059555796 * units.meter,
        rtol=rtol,
        atol=atol,
    )
    assert np.isclose(
        max(fourier.R).to("meter"),
        3.8023514986250713 * units.meter,
        rtol=rtol,
        atol=atol,
    )
    assert np.isclose(
        min(fourier.Z).to("meter"),
        -3.112945604763297 * units.meter,
        rtol=rtol,
        atol=atol,
    )
    assert np.isclose(
        max(fourier.Z).to("meter"),
        3.112868609690877 * units.meter,
        rtol=rtol,
        atol=atol,
    )
    assert all(fourier.theta <= 2 * np.pi)
    assert all(fourier.theta >= 0)


@pytest.mark.parametrize(
    ["parameters", "expected"],
    [
        (
            {
                "kappa": 1.0,
                "delta": 0.0,
                "s_kappa": 0.0,
                "s_delta": 0.0,
                "shift": 0.0,
                "dpsidr": 1.0,
                "Rmaj": 1.0,
            },
            lambda theta: 1 / (1 + 0.5 * np.cos(theta)),
        ),
        (
            {
                "kappa": 1.0,
                "delta": 0.0,
                "s_kappa": 1.0,
                "s_delta": 0.0,
                "shift": 0.0,
                "dpsidr": 3.0,
                "Rmaj": 2.5,
            },
            lambda theta: 3 / ((2.5 + 0.5 * np.cos(theta)) * (np.sin(theta) ** 2 + 1)),
        ),
        (
            {
                "kappa": 2.0,
                "delta": 0.5,
                "s_kappa": 0.5,
                "s_delta": 0.2,
                "shift": 0.1,
                "dpsidr": 0.3,
                "Rmaj": 2.5,
            },
            lambda theta: 0.3
            * np.sqrt(
                0.25
                * (0.523598775598299 * np.cos(theta) + 1.0) ** 2
                * np.sin(theta + 0.523598775598299 * np.sin(theta)) ** 2
                + np.cos(theta) ** 2
            )
            / (
                (0.5 * np.cos(theta + 0.523598775598299 * np.sin(theta)) + 2.5)
                * (
                    (0.585398163397448 * np.cos(theta) + 0.5)
                    * np.sin(theta)
                    * np.sin(theta + 0.523598775598299 * np.sin(theta))
                    + 0.1 * np.cos(theta)
                    + np.cos(0.523598775598299 * np.sin(theta))
                )
            ),
        ),
    ],
)
def test_b_poloidal(generate_miller, parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 129
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(theta, dict=parameters)

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    np.testing.assert_allclose(
        ureg.Quantity(fourier.get_b_poloidal(fourier.theta_eq)).magnitude,
        expected(theta),
        atol=atol,
    )


def test_tracer_efit_eqdsk():
    norms = SimulationNormalisation("test_tracer_efit_eqdsk", convention="gene")
    eq = read_equilibrium(template_dir / "transp_eq.geqdsk", "GEQDSK")

    fourier = LocalGeometryFourierGENE()
    fourier.from_global_eq(eq, 0.7145650753687218, norms)

    assert fourier["local_geometry"] == "FourierGENE"

    units = norms.units

    expected = {
        "rho": 0.49751406615085186 * norms.lref,
        "Rmaj": 1.0 * norms.lref,
        "beta_prime": -0.3607381577760629 * norms.bref**2 * norms.lref**-1,
        "bunit_over_b0": 1.8408524506797692 * units.dimensionless,
        "dpsidr": -0.4677771048432373 * norms.bref * norms.lref,
        "q": -1.9578769 * units.dimensionless,
        "shat": 3.3706510676698 * units.dimensionless,
        "cN": [
            0.5593502238215297,
            -0.026258332256481668,
            -0.10095116973284755,
            0.0338181228357667,
            0.017364040261600166,
            -0.010200866436236813,
            -0.0012455048919194733,
            0.0034592576249762867,
            -0.00043339502131845866,
            -0.0009917202357077413,
            0.00037658108227625706,
            0.0002269513324167718,
            -0.00023556501859256333,
            -8.145819257558843e-05,
            4.040987208307903e-05,
            -4.5470947662145435e-05,
            -7.206683756443687e-05,
            -4.366641371203621e-05,
            -5.5485403305590186e-05,
            -7.525782599651817e-05,
            -7.703072925772132e-05,
            -7.414815101774807e-05,
            -8.736711002546979e-05,
            -0.00010170665728449291,
            -0.00010563613793289249,
            -0.00010936593651974333,
            -0.00012306001704291686,
            -0.0001352249965471105,
            -0.00014024169725474587,
            -0.00015117056728272655,
            -0.00016503896461651,
            -0.00017046503618015562,
        ]
        * norms.lref,
        "sN": [
            0.0,
            0.000562560674884449,
            7.332448105238586e-05,
            -0.000661442701234799,
            0.00027695098149489175,
            0.00028598794894225694,
            -0.00021433965183750362,
            -7.861631127132802e-05,
            0.00010871521549575655,
            -4.651715941574952e-06,
            -4.840018360102326e-05,
            1.6426297956026335e-05,
            1.2902782040376272e-05,
            -1.0051783596369384e-05,
            -1.640647067867107e-06,
            4.384871547568506e-06,
            9.974940411432935e-07,
            -2.602312419494581e-06,
            -1.2997138185527722e-06,
            1.6691071331780338e-06,
            8.96409300942047e-07,
            -4.290122968958642e-07,
            -9.398389886559878e-07,
            1.5188203039932896e-07,
            1.344308281205175e-06,
            -9.632221027807203e-07,
            -1.7962523031942508e-07,
            1.592317882159835e-07,
            5.655167867208379e-07,
            -4.470904632512818e-07,
            -1.7965899329478915e-07,
            8.337593725959175e-07,
        ]
        * norms.lref,
        "dcNdr": [
            1.1458257566316417,
            -0.4562526426433849,
            -0.2750386678934288,
            0.25475855728233027,
            0.043987502311783014,
            -0.09220736478559927,
            0.014579749375623905,
            0.036790600413702676,
            -0.014226675473183154,
            -0.011461384703859523,
            0.009196056224173976,
            0.0027082820397365624,
            -0.004608663337953287,
            9.790100489883016e-05,
            0.0018646286665856913,
            -0.0005800767076541941,
            -0.0005569041725722642,
            0.0005097430383892417,
            0.00027363661167693144,
            -0.00020538282354294547,
            -0.00010211380837084047,
            0.00030359162826479807,
            0.0002748282342274107,
            -0.00018206472002074936,
            -0.0001570846010593356,
            0.0005062249364765719,
            0.00039053130990287854,
            -0.0005290307008882101,
            9.654363952948924e-05,
            0.0017438658791260157,
            -0.0005929776901697778,
            -0.00129554602277948,
        ]
        * units.dimensionless,
        "dsNdr": [
            0.0,
            0.003374609039209439,
            -3.612263580951674e-06,
            -0.005368506424962792,
            0.0033390958283972733,
            0.0024802308581067557,
            -0.0029014383669182615,
            -0.000438076477428933,
            0.001778961750599597,
            -0.00048070319019235565,
            -0.0009224371618317253,
            0.00042806472659618873,
            0.0003525801119944322,
            -0.0002337314608015051,
            -5.492363425831255e-05,
            0.00018064665441227897,
            -2.9851489248933165e-05,
            -0.0001200378308036098,
            2.864984428148849e-05,
            7.244756486573892e-05,
            2.1017220012316886e-05,
            -5.685758112451971e-05,
            -3.794237493592746e-05,
            7.757975138733211e-05,
            -9.143023286303709e-07,
            -7.534933029221941e-05,
            1.5747680925510215e-05,
            6.694256428146742e-05,
            -2.1074474177790174e-06,
            -4.330297681173282e-05,
            7.411187439478203e-06,
            1.4165849884644198e-05,
        ]
        * units.dimensionless,
    }

    for key, value in expected.items():
        if "N" in key:
            atol = 0.02
            rtol = 0.1
        else:
            atol = 0.001
            rtol = 0.15

        np.testing.assert_allclose(
            fourier[key].to(value.units).magnitude,
            value.magnitude,
            rtol=rtol,
            atol=atol,
        )
