import numpy as np
import pytest

from pyrokinetics import template_dir
from pyrokinetics.constants import deuterium_mass, electron_mass, hydrogen_mass
from pyrokinetics.equilibrium import read_equilibrium
from pyrokinetics.kinetics import read_kinetics
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.normalisation import ureg as units

tritium_mass = 1.5 * deuterium_mass
carbon_mass = 6 * deuterium_mass


@pytest.fixture
def scene_file():
    return template_dir.joinpath("scene.cdf")


@pytest.fixture
def jetto_file():
    return template_dir.joinpath("jetto.jsp")


@pytest.fixture
def transp_file():
    return template_dir.joinpath("transp.cdf")


@pytest.fixture
def pfile_file():
    return template_dir.joinpath("pfile.txt")


@pytest.fixture
def gacode_file():
    return template_dir.joinpath("input.gacode")


@pytest.fixture
def imas_file():
    return template_dir.joinpath("core_profiles.h5")


@pytest.fixture
def eliteinp_file():
    return template_dir.joinpath("test.eliteinp")


@pytest.fixture
def equilibrium():
    eq_file = template_dir / "test.geqdsk"
    return read_equilibrium(eq_file)


def check_species(
    species,
    name,
    charge,
    mass,
    midpoint_density,
    midpoint_density_gradient,
    midpoint_temperature,
    midpoint_temperature_gradient,
    midpoint_angular_velocity,
    midpoint_angular_velocity_gradient,
):
    assert species.species_type == name
    assert species.mass == mass
    assert np.isclose(species.get_charge(0.5).m, charge)
    assert np.isclose(species.get_dens(0.5).m, midpoint_density)
    assert np.isclose(species.get_norm_dens_gradient(0.5).m, midpoint_density_gradient)
    assert np.isclose(species.get_temp(0.5).m, midpoint_temperature)
    assert np.isclose(
        species.get_norm_temp_gradient(0.5).m, midpoint_temperature_gradient
    )
    assert np.isclose(species.get_angular_velocity(0.5).m, midpoint_angular_velocity)
    assert np.isclose(
        species.get_norm_ang_vel_gradient(0.5).m, midpoint_angular_velocity_gradient
    )


@pytest.mark.parametrize("kinetics_type", ["SCENE", None])
def test_read_scene(scene_file, kinetics_type):
    scene = read_kinetics(scene_file, kinetics_type)
    assert scene.kinetics_type == "SCENE"

    assert scene.nspec == 3
    assert np.array_equal(
        sorted(scene.species_names), sorted(["electron", "deuterium", "tritium"])
    )
    check_species(
        scene.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=1.5116787e20,
        midpoint_density_gradient=0.4247526509961558,
        midpoint_temperature=12174.554122236143,
        midpoint_temperature_gradient=2.782385669107711,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        scene.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=7.558393477464637e19,
        midpoint_density_gradient=0.4247526509961558,
        midpoint_temperature=12174.554122236143,
        midpoint_temperature_gradient=2.782385669107711,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        scene.species_data["tritium"],
        "tritium",
        1,
        tritium_mass,
        midpoint_density=7.558393477464637e19,
        midpoint_density_gradient=0.4247526509961558,
        midpoint_temperature=12174.554122236143,
        midpoint_temperature_gradient=2.782385669107711,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )


@pytest.mark.parametrize("kinetics_type", ["IMAS", None])
def test_read_imas(imas_file, kinetics_type, equilibrium):
    imas = read_kinetics(imas_file, kinetics_type, eq=equilibrium)
    assert imas.kinetics_type == "IMAS"

    assert imas.nspec == 5
    assert np.array_equal(
        sorted(imas.species_names),
        sorted(["electron", "deuterium", "carbon", "tungsten", "nickel"]),
    )

    check_species(
        imas.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=1.30324595e19,
        midpoint_density_gradient=0.9976205393237828,
        midpoint_temperature=1155.3048880626693,
        midpoint_temperature_gradient=2.095455413229988,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )

    check_species(
        imas.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=1.1267877863926372e19,
        midpoint_density_gradient=1.034623039315129,
        midpoint_temperature=1155.3048880626693,
        midpoint_temperature_gradient=2.095455413229988,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )

    check_species(
        imas.species_data["carbon"],
        "carbon",
        6,
        carbon_mass,
        midpoint_density=2.784259094632337e17,
        midpoint_density_gradient=0.7265520251672052,
        midpoint_temperature=1155.3048880626693,
        midpoint_temperature_gradient=2.095455413229988,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )


@pytest.mark.parametrize("kinetics_type", ["JETTO", None])
def test_read_jetto(jetto_file, kinetics_type):
    jetto = read_kinetics(jetto_file, kinetics_type)
    assert jetto.kinetics_type == "JETTO"

    assert jetto.nspec == 4
    assert np.array_equal(
        sorted(jetto.species_names),
        sorted(["electron", "deuterium", "deuterium_fast", "impurity1"]),
    )
    check_species(
        jetto.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=3.98442302e19,
        midpoint_density_gradient=0.24934713314306212,
        midpoint_temperature=2048.70870657,
        midpoint_temperature_gradient=1.877960703115299,
        midpoint_angular_velocity=30084.42620196,
        midpoint_angular_velocity_gradient=1.3539597923978433,
    )
    check_species(
        jetto.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=3.36404139e19,
        midpoint_density_gradient=0.15912588033334082,
        midpoint_temperature=1881.28998733,
        midpoint_temperature_gradient=1.2290413714311896,
        midpoint_angular_velocity=30084.42620196,
        midpoint_angular_velocity_gradient=1.3539597923978433,
    )
    check_species(
        jetto.species_data["impurity1"],
        "impurity1",
        5.99947137,
        12 * hydrogen_mass,
        midpoint_density=8.99100966e17,
        midpoint_density_gradient=0.37761249,
        midpoint_temperature=1881.28998733,
        midpoint_temperature_gradient=1.2290413714311896,
        midpoint_angular_velocity=30084.42620196,
        midpoint_angular_velocity_gradient=1.3539597923978433,
    )
    check_species(
        jetto.species_data["deuterium_fast"],
        "deuterium_fast",
        1.0,
        deuterium_mass,
        midpoint_density=8.09686716e17,
        midpoint_density_gradient=3.1423607191445337,
        midpoint_temperature=21305.937104805435,
        midpoint_temperature_gradient=0.4752751206880977,
        midpoint_angular_velocity=30084.42620196,
        midpoint_angular_velocity_gradient=1.3539597923978433,
    )


@pytest.mark.parametrize("kinetics_type", ["TRANSP", None])
def test_read_transp(transp_file, kinetics_type):
    transp = read_kinetics(transp_file, kinetics_type)
    assert transp.kinetics_type == "TRANSP"

    assert transp.nspec == 4
    assert np.array_equal(
        sorted(transp.species_names),
        sorted(["electron", "deuterium", "impurity", "deuterium_fast"]),
    )

    # Pinned values updated when bugfix/transp-loading paired centre-grid
    # profile variables (TE/TI/NE/ion densities/...) with the centre-grid
    # psi_n axis instead of the boundary-grid axis they had inherited via
    # PLFLX. The shifts (a few % at psi_n=0.5) all go in the direction
    # expected from removing a half-cell outward offset.
    check_species(
        transp.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=3.4532005909052506e19,
        midpoint_density_gradient=0.2885046125159644,
        midpoint_temperature=353.7364570791424,
        midpoint_temperature_gradient=2.2718458924226206,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=3.19764453603482e19,
        midpoint_density_gradient=0.1956175052528843,
        midpoint_temperature=422.72237842625526,
        midpoint_temperature_gradient=2.0618668656492343,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["impurity"],
        "impurity",
        6,
        12 * hydrogen_mass,
        midpoint_density=3.453200457862165e17,
        midpoint_density_gradient=0.28850678824124265,
        midpoint_temperature=422.2498819381078,
        midpoint_temperature_gradient=2.0864210737280366,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["deuterium_fast"],
        "deuterium_fast",
        1,
        1 * deuterium_mass,
        midpoint_density=4.8363892008797286e17,
        midpoint_density_gradient=6.429663907065453,
        midpoint_temperature=16562.809245931603,
        midpoint_temperature_gradient=0.8959782862330974,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )


@pytest.mark.parametrize("kinetics_type", ["TRANSP", None])
def test_read_transp_kwargs(transp_file, kinetics_type):
    transp = read_kinetics(transp_file, kinetics_type, time_index=10)
    assert transp.kinetics_type == "TRANSP"

    assert transp.nspec == 4
    assert np.array_equal(
        sorted(transp.species_names),
        sorted(["electron", "deuterium", "impurity", "deuterium_fast"]),
    )

    # Pinned values updated alongside bugfix/transp-loading; see the comment
    # in test_read_transp.
    check_species(
        transp.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=2.7216183165144388e19,
        midpoint_density_gradient=0.4939696683847586,
        midpoint_temperature=350.4251240498061,
        midpoint_temperature_gradient=3.106614577126959,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=2.533214254546288e19,
        midpoint_density_gradient=0.4287052999407568,
        midpoint_temperature=258.16003222229585,
        midpoint_temperature_gradient=5.498546361970726,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["impurity"],
        "impurity",
        6,
        12 * hydrogen_mass,
        midpoint_density=2.721618633501551e17,
        midpoint_density_gradient=0.4939909176144518,
        midpoint_temperature=259.35315019419335,
        midpoint_temperature_gradient=5.486208997594045,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["deuterium_fast"],
        "deuterium_fast",
        1,
        1 * deuterium_mass,
        midpoint_density=2.510709021491931e17,
        midpoint_density_gradient=7.078944767844185,
        midpoint_temperature=26390.58244649499,
        midpoint_temperature_gradient=0.6088534509569599,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )


@pytest.mark.parametrize("kinetics_type", ["pFile", None])
def test_read_pFile(pfile_file, equilibrium, kinetics_type):
    pfile = read_kinetics(pfile_file, kinetics_type, eq=equilibrium)
    assert pfile.kinetics_type == "pFile"

    assert pfile.nspec == 4
    assert np.array_equal(
        sorted(pfile.species_names),
        sorted(["deuterium", "deuterium_fast", "electron", "impurity"]),
    )
    check_species(
        pfile.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=7.63899297e19,
        midpoint_density_gradient=1.10742399,
        midpoint_temperature=770.37876268,
        midpoint_temperature_gradient=3.1457586490506135,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )
    check_species(
        pfile.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=5.99025662e19,
        midpoint_density_gradient=1.7807398428788435,
        midpoint_temperature=742.54533496,
        midpoint_temperature_gradient=2.410566291534264,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )
    check_species(
        pfile.species_data["impurity"],
        "impurity",
        6,
        6 * deuterium_mass,
        midpoint_density=1.4747197923435095e18,
        midpoint_density_gradient=-3.4544821178030762,
        midpoint_temperature=742.54533496,
        midpoint_temperature_gradient=2.410566291534264,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )
    check_species(
        pfile.species_data["deuterium_fast"],
        "deuterium_fast",
        1,
        deuterium_mass,
        midpoint_density=7.63899297e18,
        midpoint_density_gradient=1.1074239891222437,
        midpoint_temperature=1379.36939199,
        midpoint_temperature_gradient=3.0580150015690317,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )


@pytest.mark.parametrize("kinetics_type", ["GACODE", None])
def test_read_gacode(gacode_file, equilibrium, kinetics_type):
    gacode = read_kinetics(gacode_file, kinetics_type)
    assert gacode.kinetics_type == "GACODE"

    assert gacode.nspec == 4
    assert np.array_equal(
        sorted(gacode.species_names),
        sorted(["deuterium", "deuterium_fast", "electron", "impurity1"]),
    )

    check_species(
        gacode.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=3.90344513e19,
        midpoint_density_gradient=0.24618685944052837,
        midpoint_temperature=2.0487168760575134,
        midpoint_temperature_gradient=2.4720257831420644,
        midpoint_angular_velocity=30084.64386986,
        midpoint_angular_velocity_gradient=1.78730003,
    )
    check_species(
        gacode.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=3.364036907223085e19,
        midpoint_density_gradient=0.20733395590044212,
        midpoint_temperature=1.8812833840152388,
        midpoint_temperature_gradient=1.6103159725032943,
        midpoint_angular_velocity=30084.64386986,
        midpoint_angular_velocity_gradient=1.78730003,
    )
    check_species(
        gacode.species_data["impurity1"],
        "impurity1",
        6,
        6 * deuterium_mass,
        midpoint_density=8.990927564612032e17,
        midpoint_density_gradient=0.48831969476757303,
        midpoint_temperature=1.8812833840152388,
        midpoint_temperature_gradient=1.6103159725032943,
        midpoint_angular_velocity=30084.64386986,
        midpoint_angular_velocity_gradient=1.78730003,
    )
    check_species(
        gacode.species_data["deuterium_fast"],
        "deuterium_fast",
        1,
        deuterium_mass,
        midpoint_density=8.097878630410934e17,
        midpoint_density_gradient=4.272559775542998,
        midpoint_temperature=21.309682394342914,
        midpoint_temperature_gradient=0.6266355862448794,
        midpoint_angular_velocity=30084.64386986,
        midpoint_angular_velocity_gradient=1.78730003,
    )


@pytest.mark.parametrize("kinetics_type", ["ELITEINP", None])
def test_read_eliteinp(eliteinp_file, equilibrium, kinetics_type):
    eliteinp = read_kinetics(eliteinp_file, kinetics_type, eq=equilibrium)
    assert eliteinp.kinetics_type == "ELITEINP"

    assert eliteinp.nspec == 3
    assert np.array_equal(
        sorted(eliteinp.species_names),
        sorted(["deuterium", "electron", "impurity"]),
    )
    check_species(
        eliteinp.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=1.8610162031277978e19,
        midpoint_density_gradient=0.008087278210801626,
        midpoint_temperature=1009.2293020628986,
        midpoint_temperature_gradient=-0.5443295612297642,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        eliteinp.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=1.4888129625030572e19,
        midpoint_density_gradient=0.008087278276291673,
        midpoint_temperature=1009.2293020628986,
        midpoint_temperature_gradient=-0.5443295612297642,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )
    check_species(
        eliteinp.species_data["impurity"],
        "impurity",
        6,
        12 * hydrogen_mass,
        midpoint_density=6.203387343762956e17,
        midpoint_density_gradient=0.008087278159888247,
        midpoint_temperature=1009.2293020628986,
        midpoint_temperature_gradient=-0.5443295612297642,
        midpoint_angular_velocity=0.0,
        midpoint_angular_velocity_gradient=0.0,
    )


@pytest.mark.parametrize(
    "filename,kinetics_type",
    [
        ("scene.cdf", "SCENE"),
        ("jetto.jsp", "JETTO"),
        ("transp.cdf", "TRANSP"),
        ("input.gacode", "GACODE"),
    ],
)
def test_filetype_inference(filename, kinetics_type):
    kinetics = read_kinetics(template_dir.joinpath(filename))
    assert kinetics.kinetics_type == kinetics_type


def test_filetype_inference_pfile(pfile_file, equilibrium):
    kinetics = read_kinetics(pfile_file, eq=equilibrium)
    assert kinetics.kinetics_type == "pFile"


def test_bad_kinetics_type(scene_file):
    kinetics = read_kinetics(scene_file)
    with pytest.raises(ValueError):
        kinetics.kinetics_type = "helloworld"


# Compare JETTO and GACODE files with the same Equilibrium
# Compare only the flux surface at ``psi_n=0.5``.
@pytest.fixture(scope="module")
def kin_gacode():
    kin = read_kinetics(template_dir / "input.gacode")
    return kin


@pytest.fixture(scope="module")
def kin_jetto():
    kin = read_kinetics(template_dir / "jetto.jsp", time_index=-1)
    return kin


@pytest.mark.parametrize(
    "attr, unit",
    [
        ("get_charge", "coulomb"),
        ("get_dens", "meter ** -3"),
        ("get_temp", "eV"),
    ],
)
def test_compare_gacode_jetto_attrs(kin_gacode, kin_jetto, attr, unit):
    """
    Compare attributes between equivalent flux surfaces from GACODE and JETTO
    files. Only checks that values are within 5%. Can't compare gradients as
    gacode file uses different equilibrium
    """
    for sp1, sp2 in zip(
        sorted(kin_gacode.species_names), sorted(kin_jetto.species_names)
    ):
        assert np.isclose(
            getattr(kin_gacode.species_data[sp1], attr)(0.5).to(unit).magnitude,
            getattr(kin_jetto.species_data[sp2], attr)(0.5).to(unit).magnitude,
            rtol=5e-2,
        )


@pytest.fixture(scope="module")
def setup_hydrogenic_pfile(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pfile_hydrogenic")
    with open(template_dir / "pfile.txt", "r") as f:
        lines = f.readlines()
    lines[-2] = " 1.000000   1.000000   1.500000\n"
    print(lines)
    hydrogenic_pfile = tmp_path / "pfile_hydrogenic.txt"
    with open(hydrogenic_pfile, "w") as f:
        f.writelines(lines)

    return hydrogenic_pfile


@pytest.mark.parametrize("kinetics_type", ["pFile", None])
def test_read_pFile_hydrogenic(setup_hydrogenic_pfile, equilibrium, kinetics_type):
    # Rename the ion species in the original pyro object
    hydrogenic_pfile = setup_hydrogenic_pfile

    pfile = read_kinetics(hydrogenic_pfile, kinetics_type, eq=equilibrium)
    assert pfile.kinetics_type == "pFile"

    assert pfile.nspec == 4
    assert np.array_equal(
        sorted(pfile.species_names),
        sorted(["deuterium_fast", "electron", "hydrogenic", "impurity"]),
    )
    check_species(
        pfile.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=7.63899297e19,
        midpoint_density_gradient=1.10742399,
        midpoint_temperature=770.37876268,
        midpoint_temperature_gradient=3.1457586490506135,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )
    check_species(
        pfile.species_data["hydrogenic"],
        "hydrogenic",
        1,
        deuterium_mass * 0.75,
        midpoint_density=5.99025662e19,
        midpoint_density_gradient=1.7807398428788435,
        midpoint_temperature=742.54533496,
        midpoint_temperature_gradient=2.410566291534264,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )
    check_species(
        pfile.species_data["impurity"],
        "impurity",
        6,
        6 * deuterium_mass,
        midpoint_density=1.4747197923435095e18,
        midpoint_density_gradient=-3.4544821178030762,
        midpoint_temperature=742.54533496,
        midpoint_temperature_gradient=2.410566291534264,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )
    check_species(
        pfile.species_data["deuterium_fast"],
        "deuterium_fast",
        1,
        deuterium_mass,
        midpoint_density=7.63899297e18,
        midpoint_density_gradient=1.1074239891222437,
        midpoint_temperature=1379.36939199,
        midpoint_temperature_gradient=3.0580150015690317,
        midpoint_angular_velocity=16882.124102721187,
        midpoint_angular_velocity_gradient=4.165436791612331,
    )


def test_kinetics_pressure_consistent_with_manual_local_species(scene_file):
    psi_n = 0.5
    kinetics = read_kinetics(scene_file, "SCENE")

    local_species = LocalSpecies()

    for name in kinetics.species_names:
        species = kinetics.species_data[name]

        local_species.add_species(
            name=name,
            species_data={
                "name": name,
                "mass": species.get_mass(),
                "z": species.get_charge(psi_n),
                "dens": species.get_dens(psi_n),
                "temp": species.get_temp(psi_n),
                "omega0": species.get_angular_velocity(psi_n),
                "nu": 0.0 / units.second,
                "inverse_lt": species.get_norm_temp_gradient(psi_n),
                "inverse_ln": species.get_norm_dens_gradient(psi_n),
                "domega_drho": 0.0 / units.second,
            },
        )

    np.testing.assert_allclose(
        local_species.pressure.to("pascal").magnitude,
        kinetics.get_total_pressure(psi_n).to("pascal").magnitude,
    )


def test_kinetics_pressure_gradient_consistent_with_manual_local_species(scene_file):
    psi_n = 0.5
    kinetics = read_kinetics(scene_file, "SCENE")

    local_species = LocalSpecies()

    for name in kinetics.species_names:
        species = kinetics.species_data[name]

        local_species.add_species(
            name=name,
            species_data={
                "name": name,
                "mass": species.get_mass(),
                "z": species.get_charge(psi_n),
                "dens": species.get_dens(psi_n),
                "temp": species.get_temp(psi_n),
                "omega0": species.get_angular_velocity(psi_n),
                "nu": 0.0 / units.second,
                "inverse_lt": species.get_norm_temp_gradient(psi_n),
                "inverse_ln": species.get_norm_dens_gradient(psi_n),
                "domega_drho": 0.0 / units.second,
            },
        )

    np.testing.assert_allclose(
        local_species.inverse_lp.magnitude,
        kinetics.get_norm_total_pressure_gradient(psi_n).magnitude,
    )


def test_kinetics_total_pressure_prime_finite_difference(scene_file, equilibrium):
    psi_n = 0.5
    delta = 1.0e-4

    kinetics = read_kinetics(scene_file, "SCENE")

    p_plus = kinetics.get_total_pressure(psi_n + delta)
    p_minus = kinetics.get_total_pressure(psi_n - delta)

    psi_plus = equilibrium.psi(psi_n + delta)
    psi_minus = equilibrium.psi(psi_n - delta)

    expected = (p_plus - p_minus) / (psi_plus - psi_minus)

    actual = kinetics.get_total_pressure_prime(psi_n, eq=equilibrium)

    np.testing.assert_allclose(
        actual.to("pascal / weber").magnitude,
        expected.to("pascal / weber").magnitude,
        rtol=1e-3,
    )
