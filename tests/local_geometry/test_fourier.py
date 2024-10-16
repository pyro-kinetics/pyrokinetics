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
        "beta_prime": -0.43155100493936366 * norms.bref**2 * norms.lref**-1,
        "bunit_over_b0": 1.8716344800206974 * units.dimensionless,
        "dpsidr": 0.47559909435737896 * norms.bref * norms.lref,
        "q": -1.9578769 * units.dimensionless,
        "shat": 3.713279750031135 * units.dimensionless,
        "cN": [
            0.5865913223773318,
            -0.027537147900860575,
            -0.10586762573286063,
            0.035465110318577256,
            0.018209692076775072,
            -0.010697662175488658,
            -0.001306162633827051,
            0.00362772806420857,
            -0.0004545019342801974,
            -0.001040018327905015,
            0.0003949210809741235,
            0.00023800416363134997,
            -0.0002470373477603923,
            -8.542531470699412e-05,
            4.237788650617461e-05,
            -4.768544318558952e-05,
            -7.557658823778995e-05,
            -4.5793026036209606e-05,
            -5.81876161151774e-05,
            -7.892298204322548e-05,
            -8.078222804715317e-05,
            -7.775926441966109e-05,
            -9.162200428202203e-05,
            -0.00010665990653134058,
            -0.00011078075810453364,
            -0.00011469220283466038,
            -0.0001290532032610999,
            -0.00014181063341873028,
            -0.00014707165411155843,
            -0.00015853277462028758,
            -0.00017307658131745615,
            -0.00017876690977050862,
        ]
        * norms.lref,
        "sN": [
            0.0,
            -0.0016144872467520539,
            -0.0033908488738985743,
            0.001242575969577981,
            0.0013320295214382803,
            -0.0006289763252570288,
            -0.00033491260383606316,
            0.00031952839476446485,
            5.180046831900659e-05,
            -0.00015554856037304713,
            5.406892108025965e-06,
            5.403179576915652e-05,
            -2.3963779919574236e-05,
            -2.0412050818685445e-05,
            -8.722286435861292e-06,
            -5.56280715756868e-06,
            8.97447638017129e-06,
            -1.9560626678022537e-05,
            -2.356370873134238e-05,
            1.1172479417242202e-05,
            -4.376664346354685e-08,
            -2.9441451192380378e-05,
            -1.1702582604667124e-05,
            9.090344727503698e-06,
            -1.156543824327516e-05,
            -2.55444368092725e-05,
            -9.115732584420515e-06,
            3.5986984018474526e-06,
            -1.6757967378808148e-05,
            -1.4349038695235438e-05,
            -7.0757412503262265e-06,
            -3.7519489443125294e-06,
        ]
        * norms.lref,
        "dcNdr": [
            1.1649858162354552,
            -0.46388192411535756,
            -0.27963775777624506,
            0.2590185298606985,
            0.044723044425825746,
            -0.0937492182649791,
            0.014823544759047137,
            0.03740579713893232,
            -0.014464566151431366,
            -0.011653037570135732,
            0.009349827305709092,
            0.0027535701075840253,
            -0.004685726974146843,
            9.953655995879811e-05,
            0.0018958090503641309,
            -0.0005897756198697491,
            -0.0005662183290359582,
            0.0005182673310388051,
            0.0002782143331069528,
            -0.00020881951598682305,
            -0.00010382226639334815,
            0.0003086720446278327,
            0.00027942252530923923,
            -0.0001851129765877197,
            -0.00015970720618705818,
            0.0005146918486715422,
            0.00039705597807019206,
            -0.0005378756192318649,
            9.816286196271342e-05,
            0.0017730224299545062,
            -0.0006028950998913554,
            -0.0013172063112284328,
        ]
        * units.dimensionless,
        "dsNdr": [
            0.0,
            0.0034310379143453586,
            -3.67267230423721e-06,
            -0.005458276461450309,
            0.0033949308690309197,
            0.002521704284343824,
            -0.0029499550192574795,
            -0.00044540184336446176,
            0.0018087088020190903,
            -0.0004887412727924835,
            -0.0009378617956098928,
            0.0004352226580463687,
            0.00035847583717238033,
            -0.00023763992685116637,
            -5.584198142767433e-05,
            0.00018366752067521013,
            -3.0350891776908045e-05,
            -0.00012204516950577724,
            2.9129335732101444e-05,
            7.365889773789957e-05,
            2.1368180018600513e-05,
            -5.7807932396935e-05,
            -3.857650768341131e-05,
            7.887639402640479e-05,
            -9.295973272340349e-07,
            -7.660868559367262e-05,
            1.6010674790029376e-05,
            6.8061581197337e-05,
            -2.142219109201988e-06,
            -4.402704475660483e-05,
            7.534802754674087e-06,
            1.4402883161479222e-05,
        ]
        * units.dimensionless,
    }

    for key, value in expected.items():
        if "N" in key:
            atol = 0.02
            rtol = 0.05
        else:
            atol = 0.001
            rtol = 0.15

        np.testing.assert_allclose(
            fourier[key].to(value.units).magnitude,
            value.magnitude,
            rtol=rtol,
            atol=atol,
        )
