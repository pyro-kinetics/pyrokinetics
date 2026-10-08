import shutil

import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.local_geometry import MetricTerms


def _em_copy(tmp_path):
    """GFTM_linear template with USE_BPER/USE_BPAR on and phi = apar = sigma"""
    d = tmp_path / "gftm"
    shutil.copytree(template_dir / "outputs" / "GFTM_linear", d)
    inp = d / "input.gftm"
    inp.write_text(
        inp.read_text()
        .replace("USE_BPER = F", "USE_BPER = T")
        .replace("USE_BPAR = F", "USE_BPAR = T")
    )
    lines = (d / "out.gftm.wavefunction").read_text().splitlines()
    nmode, _, ntheta = (int(x) for x in lines[0].split())
    rows = np.array([[float(x) for x in line.split()] for line in lines[2:]])
    out = [f"{nmode} 3 {ntheta}", lines[1]]
    for row in rows:
        modes = row[1:].reshape(nmode, 2)
        out.append(" ".join(f"{x:.17e}" for x in [row[0], *np.tile(modes, 3).ravel()]))
    (d / "out.gftm.wavefunction").write_text("\n".join(out) + "\n")
    return inp


def test_gftm_bpar_is_sigma_times_kperp(tmp_path):
    pyro = Pyro(gk_file=_em_copy(tmp_path))
    pyro.load_gk_output()
    data = pyro.gk_output.data
    ef = np.asarray(data.eigenfunctions.isel(mode=0).pint.dequantify())
    phi, apar, bpar = ef
    theta = np.asarray(data.theta.pint.dequantify())
    ky = float(data.ky.pint.dequantify()[0])

    # Expected k_perp rho_unit, independently from the metric on a fine grid
    metric = MetricTerms(pyro.local_geometry, ntheta=256)
    th, kp = metric.k_perp(ky=1.0, theta0=0.0, nperiod=5)
    k_perp = ky * np.interp(theta, th, getattr(kp, "m", kp))

    ok = np.abs(phi) > 1e-6 * np.abs(phi).max()
    np.testing.assert_allclose(apar[ok] / phi[ok], 1.0, rtol=1e-10)
    np.testing.assert_allclose(bpar[ok] / phi[ok], k_perp[ok], rtol=3e-2)  # reader metric: 4*NXGRID points
    # k_perp grows away from the outboard midplane through magnetic shear
    assert k_perp[np.argmax(np.abs(theta))] > k_perp[np.argmin(np.abs(theta))]

    # Eigenvalues are untouched by the field factor
    ref = Pyro(gk_file=template_dir / "outputs" / "GFTM_linear" / "input.gftm")
    ref.load_gk_output()
    np.testing.assert_array_equal(
        np.asarray(data.growth_rate.pint.dequantify()),
        np.asarray(ref.gk_output.data.growth_rate.pint.dequantify()),
    )
