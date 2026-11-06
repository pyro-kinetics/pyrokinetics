import sys
from pathlib import Path

import f90nml
import numpy as np
import pytest

from pyrokinetics import Pyro, read_equilibrium, template_dir
from pyrokinetics.gk_code import GKInputGENE
from pyrokinetics.gk_code.gene import read_gene_geometry_data
from pyrokinetics.gk_code.gk_input import read_gk_input
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

docs_dir = Path(__file__).parent.parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_JETTO  # noqa

template_file = template_dir / "input.gene"
geometry_file = template_dir / "outputs" / "GENE_linear" / "miller_0001"


@pytest.fixture
def default_gene():
    return GKInputGENE()


@pytest.fixture
def gene():
    return GKInputGENE(template_file)


def test_read(gene):
    """Ensure a gene file can be read, and that the 'data' attribute is set"""
    params = ["general", "box", "geometry"]
    assert np.all(np.isin(params, list(gene.data)))


def test_read_str():
    """Ensure a gene file can be read as a string, and that the 'data' attribute is set"""
    params = ["general", "box", "geometry"]
    with open(template_file, "r") as f:
        gene = GKInputGENE.from_str(f.read())
        assert np.all(np.isin(params, list(gene.data)))


def test_verify_file_type(gene):
    """Ensure that 'verify_file_type' does not raise exception on GENE file"""
    gene.verify_file_type(template_file)


@pytest.mark.parametrize(
    "filename", ["input.gs2", "input.cgyro", "transp.cdf", "helloworld"]
)
def test_verify_file_type_bad_inputs(gene, filename):
    """Ensure that 'verify_file_type' raises exception on non-GENE file"""
    with pytest.raises(Exception):
        gene.verify_file_type(template_dir / filename)


def test_is_nonlinear(gene):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    gene.data["general"]["nonlinear"] = 0
    assert gene.is_linear()
    assert not gene.is_nonlinear()
    gene.data["general"]["nonlinear"] = 1
    assert not gene.is_linear()
    assert gene.is_nonlinear()


def test_add_flags(gene):
    gene.add_flags({"foo": {"bar": "baz"}})
    assert gene.data["foo"]["bar"] == "baz"


def test_get_local_geometry(gene):
    # TODO test it has the correct values
    local_geometry = gene.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(gene):
    local_species = gene.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    assert len(gene.data["species"]) == 2
    # Ensure you can index gene.data["species"] (doesn't work on some f90nml versions)
    assert gene.data["species"][0]
    assert gene.data["species"][1]
    assert local_species["electron"]
    assert local_species["ion1"]
    # TODO test it has the correct values


def test_get_numerics(gene):
    # TODO test it has the correct values
    numerics = gene.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, gene):
    """Ensure a gene file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gene.get_local_geometry()
    local_species = gene.get_local_species()
    numerics = gene.get_numerics()

    # Set output path
    filename = tmp_path / "input.in"

    # Write out a new input file
    gene_writer = GKInputGENE()
    gene_writer.set(local_geometry, local_species, numerics)

    # Ensure you can index gene.data["species"] (doesn't work on some f90nml versions)
    assert len(gene_writer.data["species"]) == 2
    assert gene_writer.data["species"][0]
    assert gene_writer.data["species"][1]

    # Write to disk
    gene_writer.write(filename)

    # Ensure a new file exists
    assert Path(filename).exists()

    # Ensure it is a valid file
    GKInputGENE().verify_file_type(filename)
    gene_reader = GKInputGENE(filename)
    new_local_geometry = gene_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gene_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gene_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_species_order(tmp_path):
    pyro = example_JETTO.main(tmp_path)

    # Reverse species order so electron is last
    pyro.local_species.names = pyro.local_species.names[::-1]
    pyro.gk_code = "GENE"

    pyro.write_gk_file(file_name=tmp_path / "input.in")

    assert Path(tmp_path / "input.in").exists()


def test_drop_species(tmp_path):
    pyro = example_JETTO.main(tmp_path)
    pyro.gk_code = "GENE"

    n_species = pyro.local_species.nspec
    assert len(pyro.gk_input.data["species"]) == n_species

    pyro.local_species.merge_species(
        base_species="deuterium",
        merge_species=["deuterium", "impurity1"],
        keep_base_species_z=True,
        keep_base_species_mass=True,
    )

    pyro.update_gk_code()
    n_species = pyro.local_species.nspec
    assert len(pyro.gk_input.data["species"]) == n_species


def _write_geometry_file(path, extra_header_lines=()):
    """Copy the reference geometry file, optionally adding entries to its
    ``&parameters`` header, so the header length changes."""
    lines = geometry_file.read_text().split("\n")
    end_of_header = lines.index("/")
    lines[end_of_header:end_of_header] = list(extra_header_lines)
    path.write_text("\n".join(lines))
    return path


def test_read_gene_geometry_data():
    """All gridpoints rows of data should be read, none dropped."""
    data = read_gene_geometry_data(geometry_file)
    gridpoints = f90nml.read(geometry_file)["parameters"]["gridpoints"]
    assert len(data) == gridpoints


@pytest.mark.parametrize(
    "extra_header_lines",
    [
        (),
        ("edge_opt =   0.0000000000000000E+00",),
        ("edge_opt =   0.0000000000000000E+00", "my_parameter =   1"),
    ],
)
def test_read_gene_geometry_data_header_length(tmp_path, extra_header_lines):
    """The header holds a different number of entries depending on what GENE was
    asked to do, so its length has to be found rather than assumed."""
    reference = read_gene_geometry_data(geometry_file)
    path = _write_geometry_file(tmp_path / "miller_0001", extra_header_lines)

    np.testing.assert_allclose(read_gene_geometry_data(path), reference)


def test_read_gene_geometry_data_row_count_mismatch(tmp_path):
    """A header length that doesn't line up with 'gridpoints' should be reported
    rather than silently returning short data."""
    path = tmp_path / "miller_0001"
    lines = geometry_file.read_text().split("\n")
    del lines[lines.index("/") + 1]
    path.write_text("\n".join(lines))

    with pytest.raises(ValueError, match="gridpoints"):
        read_gene_geometry_data(path)


def test_read_gene_geometry_data_no_header(tmp_path):
    path = tmp_path / "miller_0001"
    path.write_text("1.0 2.0\n3.0 4.0\n")

    with pytest.raises(ValueError, match="namelist"):
        read_gene_geometry_data(path)


eq_file = template_dir / "transp_eq.geqdsk"
tracer_psi_n = 0.7145650753687218


@pytest.fixture(scope="module")
def tracer_efit_file(tmp_path_factory):
    """GENE tracer_efit input that references transp_eq.geqdsk, with no GENE geometry
    file beside it and no local geometry in it. Reference values are arbitrary, as
    physical results shouldn't depend on them."""
    eq = read_equilibrium(eq_file)
    nml = f90nml.read(template_file)
    nml["geometry"] = f90nml.Namelist(
        magn_geometry="tracer_efit", geomfile=eq_file.name, minor_r=1.0, dpdx_pm=-2
    )
    nml["box"]["x0"] = float(eq.rho_tor(tracer_psi_n).m)
    nml["units"] = f90nml.Namelist(Lref=1.0, Bref=2.0, Tref=1.0, nref=1.0, mref=2.0)
    path = tmp_path_factory.mktemp("tracer_efit") / "input.gene"
    nml.write(path)
    return path


def test_tracer_efit_requires_equilibrium(tracer_efit_file):
    with pytest.raises(FileNotFoundError, match="eq_file"):
        GKInputGENE(tracer_efit_file)


def _si(quantity, pyro, unit):
    return quantity.to(pyro.norms.pyrokinetics, pyro.norms.context).to(unit).m


def test_tracer_efit_geometry_from_equilibrium(tracer_efit_file):
    """Geometry from the equilibrium at x0 should match a local geometry fitted to the
    same flux surface, compared in SI units."""
    pyro = Pyro(eq_file=eq_file, gk_file=tracer_efit_file, gk_code="GENE")
    reference = Pyro(eq_file=eq_file)
    reference.load_local_geometry(tracer_psi_n, "MXH")

    geometry, expected = pyro.local_geometry, reference.local_geometry
    assert np.isclose(geometry.psi_n, tracer_psi_n)
    for key, unit in [("rho", "meter"), ("Rmaj", "meter"), ("dpsidr", "weber/meter")]:
        assert np.isclose(
            _si(geometry[key], pyro, unit),
            _si(expected[key], reference, unit),
            rtol=1e-3,
        )
    for key in ["q", "shat"]:
        assert np.isclose(abs(geometry[key].m), abs(expected[key].m), rtol=1e-3)

    # Bunit = (q / r) dpsi/dr exactly, which fitted local geometries only approximate
    fs = pyro.eq.flux_surface(tracer_psi_n)
    bunit = abs(fs.q) * fs.psi_gradient / (2 * np.pi * fs.r_minor)
    b0 = abs(fs.F) / fs.R_major
    assert np.isclose(geometry.bunit_over_b0.m, (bunit / b0).to("").m, rtol=1e-3)


def test_tracer_efit_gradients(tracer_efit_file):
    """GENE gradients are with respect to x = rho_tor, so 1/Ln = omn drho_tor/dr"""
    pyro = Pyro(eq_file=eq_file, gk_file=tracer_efit_file, gk_code="GENE")
    eq = pyro.eq
    dpsi_n = np.array([-1e-4, 1e-4]) + tracer_psi_n
    drhotor_dr = np.diff(eq.rho_tor(dpsi_n).m) / np.diff(eq.r_minor(dpsi_n).m)

    omn = pyro.gk_input.data["species"][0]["omn"]
    inverse_ln = _si(pyro.local_species.ion1.inverse_ln, pyro, "1 / meter")
    assert np.isclose(inverse_ln, omn * drhotor_dr[0], rtol=1e-3)


def test_tracer_efit_x0_change(tracer_efit_file):
    gene = read_gk_input(
        tracer_efit_file, "GENE", equilibrium=read_equilibrium(eq_file)
    )
    rho = gene.get_gene_geometry()["rho"]
    gene.add_flags({"box": {"x0": 0.5}})
    assert gene.get_gene_geometry()["rho"] < rho


def test_gene_geometry_file_without_suffix(tmp_path):
    """GENE parameters files may have no suffix, with the geometry file at .dat"""
    gene = GKInputGENE()
    gene.data = f90nml.Namelist({"geometry": {"magn_geometry": "tracer_efit"}})
    gene.original_filename = tmp_path / "parameters"
    (tmp_path / "tracer_efit.dat").touch()
    assert gene._gene_geometry_filename() == tmp_path / "tracer_efit.dat"


def test_tracer_efit_write_warns(tracer_efit_file, tmp_path):
    pyro = Pyro(eq_file=eq_file, gk_file=tracer_efit_file, gk_code="GENE")
    with pytest.warns(UserWarning, match="magn_geometry = 'tracer_efit'"):
        pyro.write_gk_file(tmp_path / "input.gene", gk_code="GENE")
