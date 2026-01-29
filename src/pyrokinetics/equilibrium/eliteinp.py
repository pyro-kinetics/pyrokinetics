from typing import Optional

import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator

from ..constants import electron_charge
from ..file_utils import FileReader
from ..typing import PathLike
from ..units import UnitSpline, UnitCloughTocher2DInterpolator
from ..units import ureg as units
from .equilibrium import Equilibrium


def read_eqin(filename_or_file):
    def token_generator(fp):
        for line in fp:
            for tok in line.split():
                yield tok

    with open(filename_or_file) as f:
        f.readline()
        tokens = token_generator(f)

        try:
            npsi, npol = int(next(tokens)), int(next(tokens))
        except (StopIteration, ValueError):
            raise IOError("Second line should contain Npsi and Npol")

        data_dict = {"npsi": npsi, "npol": npol}

        while True:
            try:
                varname = next(tokens).rstrip(":")
            except StopIteration:
                break

            try:
                if varname.lower() in {"r", "z"} or varname.startswith("B"):
                    data = np.array([float(next(tokens)) for _ in range(npsi * npol)])
                    data = data.reshape((npsi, npol), order="F")
                elif varname in {"Zeff", "Zimp", "Aimp", "Amain"}:
                    data = float(next(tokens))
                else:
                    data = np.array([float(next(tokens)) for _ in range(npsi)])
            except (StopIteration, ValueError):
                raise IOError(f"Error while reading {varname}")

            data_dict[varname] = data

    necessary_keys = ["Psi", "fpol", "q", "R", "z"]
    if not all(key in data_dict.keys() for key in necessary_keys):
        raise ValueError(
            "Missing equilibrium data in ELITEINP file."
            " Currently only HELENA ELITEINP files are supported, please"
            " raise issue if another source (like) SCENE is needed"
        )

    if "Psi" in data_dict:
        data_dict["Psi"] = data_dict["Psi"] * units.weber / units.radian

    if "dp/dpsi" in data_dict:
        data_dict["p_prime"] = (
            data_dict["dp/dpsi"] * units.pascal / (units.weber / units.radian)
        )

    if "ffp" in data_dict:
        data_dict["ff_prime"] = data_dict["ffp"] * (
            units.tesla**2 * units.meter**2 * units.radian / units.weber
        )

    if "Bt" not in data_dict and "fpol" in data_dict and "R" in data_dict:
        fpol = data_dict["fpol"]
        R = data_dict["R"]  # shape (npsi, npol)
        data_dict["Bt"] = fpol[:, None] * R  # shape (npsi, npol)
    else:
        raise ValueError("Missing Bt data in ELITEINP file")

    if "p" not in data_dict:
        if "ne" not in data_dict or "Te" not in data_dict:
            raise ValueError(
                "No electron density and temperature data found in ELITEINP"
            )

        if "nMainIon" not in data_dict or "Ti" not in data_dict:
            raise ValueError(
                "No main ion density and temperature data found in ELITEINP"
            )

        data_dict["p"] = (
            data_dict["ne"] * (data_dict["Te"] * electron_charge.m)
        ) * units.pascal

        data_dict["p"] += (
            data_dict["nMainIon"] * (data_dict["Ti"] * electron_charge.m)
        ) * units.pascal

        if "nZ" in data_dict and "Ti" in data_dict:
            data_dict["p"] += (
                data_dict["nZ"] * (data_dict["Ti"] * electron_charge.m)
            ) * units.pascal

    return data_dict


class EquilibriumReaderELITEINP(FileReader, file_type="ELITEINP", reads=Equilibrium):
    def read_from_file(
        self,
        filename: PathLike,
        clockwise_phi: bool = False,
        cocos: Optional[int] = None,
        grid_length: Optional[int] = None,
        interpolator: int = 0,
    ) -> Equilibrium:
        data = read_eqin(filename)

        # Units
        len_units = units.meter
        psi = data["Psi"]
        psi_n = psi / psi[-1]
        npsi = len(psi)
        ntheta = data["npol"]
        F = data["fpol"] * units.tesla * units.meter
        p = data["p"]
        q = data["q"] * units.dimensionless

        FF_prime = F * UnitSpline(psi, F)(psi, derivative=1)
        p_prime = UnitSpline(psi, p)(psi, derivative=1)

        # Flux surface contours
        R2D = data["R"] * len_units
        Z2D = data["z"] * len_units
        Bpol2D = data["Bp"] * units.tesla

        # R_major can be obtained from the flux surfaces
        R_major = (np.max(R2D, axis=1) + np.min(R2D, axis=1)) / 2
        r_minor = (np.max(R2D, axis=1) - np.min(R2D, axis=1)) / 2
        Z_mid = (np.max(Z2D, axis=1) + np.min(Z2D, axis=1)) / 2
        a_minor = r_minor[-1]

        # Flatten known grid and add axis once
        surface_coords = np.stack((R2D[1:, :-1].m.ravel(), Z2D[1:, :-1].m.ravel()), -1)
        surface_coords = np.append(
            surface_coords, np.array([[R2D[0, 0].m, Z2D[0, 0].m]]), axis=0
        )

        Z_range = (np.max(Z2D, axis=1) - np.min(Z2D, axis=1)) / 2
        Zind_upper = np.argmax(Z2D, axis=-1)
        Zind_lower = np.argmin(Z2D, axis=-1)
        rows = np.arange(Z2D.shape[0])

        R_upper = R2D[rows, Zind_upper]
        R_lower = R2D[rows, Zind_lower]

        theta = np.arcsin((Z2D - Z_mid[:, None]) / (Z_range[:, None]))
        theta = np.where(
            (R2D <= R_upper[:, None]) & (Z2D > Z_mid[:, None]), np.pi - theta, theta
        )
        theta = np.where(
            (R2D <= R_lower[:, None]) & (Z2D <= Z_mid[:, None]), np.pi - theta, theta
        )
        theta = np.where(
            (R2D > R_lower[:, None]) & (Z2D <= Z_mid[:, None]), 2 * np.pi + theta, theta
        )

        # Remove axis NaN
        theta[0, :] = np.linspace(0, 2 * np.pi, ntheta, endpoint=True)
        theta[:, 0] = np.where(
            theta[:, 0] < np.pi, theta[:, 0], theta[:, 0] - 2 * np.pi
        )

        # Set up R, Z and Bpol interpolator
        surface_psin_theta = np.repeat(psi_n.m, ntheta)
        psin_theta = np.stack((surface_psin_theta.ravel(), theta.m.ravel()), -1)
        R2D_interp = UnitCloughTocher2DInterpolator(psin_theta, R2D.ravel())

        Z2D_interp = UnitCloughTocher2DInterpolator(psin_theta, Z2D.ravel())
        Bpol2D_interp = UnitCloughTocher2DInterpolator(psin_theta, Bpol2D.ravel())

        surface_interps = {
            "R": R2D_interp,
            "Z": Z2D_interp,
            "Bpol": Bpol2D_interp,
        }

        # Psi(R,Z) interpolator
        surface_psi = np.repeat(psi[1:].magnitude, ntheta - 1)
        surface_psi = np.append(surface_psi, psi[0].m)
        psi_next = 2 * psi[-1] - psi[-2]

        if interpolator == 0:
            psi_interp = CloughTocher2DInterpolator(
                surface_coords, surface_psi, fill_value=psi_next.m
            )
        else:
            psi_interp = RBFInterpolator(surface_coords, surface_psi, neighbors=256)

        if grid_length:
            npsi = grid_length

        R = np.linspace(min(R2D[-1, :]), max(R2D[-1, :]), npsi).m
        Z = np.linspace(min(Z2D[-1, :]), max(Z2D[-1, :]), npsi).m
        RZ_coords = np.stack([x.ravel() for x in np.meshgrid(R, Z)], -1)

        psi_RZ = psi_interp(RZ_coords).reshape((npsi, npsi)).T * psi.units

        # Magnetic field strength at axis
        B_0 = F[0] / R_major[0]

        # Infer total current
        Ip = data.get("Ip")
        if Ip is None:
            try:
                dpsi = np.gradient(psi.magnitude)
                I_profile = (
                    2
                    * np.pi
                    * np.mean(data["R"])
                    * np.mean(data["Bt"])
                    / (mu_0 * data["q"])
                    * dpsi
                )
                Ip = np.trapezoid(I_profile, psi.magnitude)
            except Exception as e:
                raise IOError(f"Could not infer total current: {e}")

        Ip *= units.ampere

        return Equilibrium(
            R=R,
            Z=Z,
            psi_RZ=psi_RZ,
            psi=psi,
            F=F,
            FF_prime=FF_prime,
            p=p,
            p_prime=p_prime,
            q=q,
            R_major=R_major,
            r_minor=r_minor,
            Z_mid=Z_mid,
            psi_lcfs=psi[-1],
            a_minor=a_minor,
            B_0=B_0,
            I_p=Ip,
            clockwise_phi=clockwise_phi,
            cocos=cocos,
            eq_type="ELITEINP",
            surface_interps=surface_interps,
        )

    def verify_file_type(self, filename: PathLike) -> None:
        with open(filename, "r") as f:
            head = f.read(50).upper()
        if "HELENA GENERATED INPUT" not in head and "Psi:" not in head:
            raise ValueError(
                f"{filename} does not appear to be an ELITEINP .eqin file."
                " Currently only HELENA ELITEINP files are supported, please"
                " raise issue if another source (like) SCENE is needed"
            )
