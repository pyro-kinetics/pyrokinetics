from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict

import git
import idspy_toolkit as idspy
import numpy as np
import pint
from idspy_dictionaries import ids_gyrokinetics_local as gkids
from idspy_toolkit import ids_to_hdf5
from xmltodict import unparse as dicttoxml

from pyrokinetics import __version__ as pyro_version

from ..gk_code.gk_output import GKOutput
from ..normalisation import convert_dict
from ..pyro import Pyro

if TYPE_CHECKING:
    import xarray as xr

imas_pyro_field_names = {
    "phi": "phi_potential",
    "apar": "a_field_parallel",
    "bpar": "b_field_parallel",
}

imas_pyro_flux_names = {
    "particle": "particles",
    "heat": "energy",
    "momentum": "momentum_tor_perpendicular",
}

imas_pyro_moment_names = {
    "density": "density",
    "velocity": "j_parallel",
}


def pyro_to_ids(
    pyro: Pyro,
    comment: str = None,
    time_interval: [float, float] = None,
    format: str = "hdf5",
    file_name: str = None,
    reference_values: Dict = {},
):
    """
    Return a Gyrokinetics IDS structure from idspy_toolkit
    GKDB/IMAS/OMAS gyrokinetics schema as described in:

    https://gitlab.com/gkdb/gkdb/raw/master/doc/general/IOGKDB.pdf

    Requires species and geometry data to already exist
    Parameters
    ----------
    pyro : Pyro
        pyro object with data loaded
    comment : str
        String describing run
    name : str
        Name for IDS
    time_interval : Float
        Final fraction of data over which to average fluxes (ignored if linear)
    format : str
        File format to save IDS in (currently hdf5 support)
    file_name : str
        Filename to save ids under
    reference_values : dict
        If normalised quantities aren't defined, can be set via dictionary here

    Returns
    -------
    ids : Gyrokinetics IDS
        Populated IDS

    """

    # generate a gyrokinetics IDS
    # this initialises the root structure and fill the whole structure with IMAS default values
    ids = gkids.GyrokineticsLocal()
    idspy.fill_default_values_ids(ids)

    ids = pyro_to_imas_mapping(
        pyro,
        comment=comment,
        reference_values=reference_values,
        ids=ids,
        time_interval=time_interval,
    )

    if file_name is not None:
        if format == "hdf5":
            ids_to_hdf5(ids, filename=file_name)
        else:
            raise ValueError(f"Format {format} not supported when writing IDS")

    return ids


def ids_to_pyro(ids_path, file_format="hdf5"):
    ids = gkids.GyrokineticsLocal()
    idspy.fill_default_values_ids(ids)

    if file_format == "hdf5":
        idspy.hdf5_to_ids(ids_path, ids)

    try:
        gk_input_dict = ids.linear.wavevector[0].eigenmode[0].code.parameters
    except IndexError:
        gk_input_dict = ids.non_linear.code.parameters

    gk_code = ids.code.name

    pyro = Pyro()

    pyro.read_gk_dict(gk_dict=gk_input_dict, gk_code=gk_code)

    # Set up reference values
    units = pyro.norms.units

    if pyro.local_geometry.Rmaj.units == "lref_minor_radius":
        lref_minor_radius = (
            ids.normalizing_quantities.r / pyro.local_geometry.Rmaj.m * units.meter
        )
    else:
        lref_minor_radius = None

    if ids.normalizing_quantities.t_e != 9e40:
        reference_values = {
            "tref_electron": ids.normalizing_quantities.t_e * units.eV,
            "nref_electron": ids.normalizing_quantities.n_e * units.meter**-3,
            "bref_B0": ids.normalizing_quantities.b_field_tor * units.tesla,
            "lref_major_radius": ids.normalizing_quantities.r * units.meter,
            "lref_minor_radius": lref_minor_radius,
        }

        pyro.set_reference_values(**reference_values)

    original_theta_geo = pyro.local_geometry.theta
    original_lg = pyro.local_geometry

    if pyro.local_geometry.local_geometry != "MXH":
        pyro.switch_local_geometry("MXH")

        # Original local_geometry theta grid using MXH theta definition
        mxh_theta_geo = pyro.local_geometry.theta_eq

        # Revert local geometry
        pyro.local_geometry = original_lg
    else:
        mxh_theta_geo = original_theta_geo

    if lref_minor_radius is None:
        output_convention = pyro.gk_code.lower()
    else:
        output_convention = "pyrokinetics"

    pyro.load_gk_output(
        ids_path,
        gk_type="IDS",
        ids=ids,
        original_theta_geo=original_theta_geo,
        mxh_theta_geo=mxh_theta_geo,
        output_convention=output_convention,
    )

    return pyro


def pyro_to_imas_mapping(
    pyro,
    comment=None,
    time_interval: [float, float] = [0.5, 1.0],
    reference_values: Dict = {},
    ids=None,
):
    """
    Return a dictionary mapping from pyro to ids data format

    Parameters
    ----------
    pyro : Pyro
        pyro object with data loaded
    comment : str
        String describing run
    time_interval : Float
        Final fraction of data over which to average fluxes (ignored if linear)
    format : str
        File format to save IDS in (currently hdf5 support)
    file_name : str
        Filename to save ids under
    reference_values : dict
        If normalised quantities aren't defined, can be set via dictionary here

    Returns
    -------
    data : dict
        Dictionary containing mapping from pyro to ids
    """
    if comment is None:
        raise ValueError("A comment is needed for IMAS upload")

    # Convert output to IMAS norm before switching geometry as this can tweak Bunit/B0
    norms = pyro.norms
    original_convention = norms.default_convention

    if pyro.gk_output:
        pyro.gk_output.to(norms.imas)
        original_theta_output = pyro.gk_output["theta"].data
    else:
        original_theta_output = pyro.local_geometry.theta

    # Convert gk output theta to local geometry theta
    original_theta_geo = pyro.local_geometry.theta

    if pyro.local_geometry.local_geometry != "MXH":
        pyro.switch_local_geometry("MXH")

        # Original local_geometry theta grid using MXH theta definition
        mxh_theta_geo = pyro.local_geometry.theta_eq

        # Need to interpolate on theta mod 2pi and then add back on each period
        theta_interval = original_theta_output // (2 * np.pi)
        theta_mod = original_theta_output % (2 * np.pi)
        mxh_theta_output = (
            np.interp(theta_mod, original_theta_geo, mxh_theta_geo)
            + theta_interval * 2 * np.pi
        )
    else:
        mxh_theta_output = original_theta_output

    geometry = pyro.local_geometry

    aspect_ratio = geometry.Rmaj.m

    species_list = [pyro.local_species[name] for name in pyro.local_species.names]

    numerics = pyro.numerics

    ids_properties = {
        "provider": "pyrokinetics",
        "creation_date": str(datetime.now()),
        "comment": comment,
        "homogeneous_time": 1,
    }

    ids_properties = gkids.IdsProperties(**ids_properties)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    code_library = [
        {
            "name": "pyrokinetics",
            "commit": sha,
            "version": pyro_version,
            "repository": "https://github.com/pyro-kinetics/pyrokinetics",
        }
    ]

    code_library = [gkids.Library(**cl) for cl in code_library]

    xml_gk_input = dicttoxml({"root": pyro.gk_input.data})

    code_output = gkids.CodePartialConstant(
        **{
            "parameters": xml_gk_input,
            "output_flag": 0,
        }
    )
    code = {
        "name": pyro.gk_code,
        "commit": None,
        "version": None,
        "repository": None,
        "library": code_library,
        "parameters": xml_gk_input,
    }

    code = gkids.Code(**code)

    try:
        normalizing_quantities = {
            "r": (1.0 * norms.gene.lref).to("meter").m,
            "b_field_tor": (1.0 * norms.bref).to("tesla").m,
            "n_e": (1.0 * norms.nref).to("meter**-3").m,
            "t_e": (1.0 * norms.tref).to("eV").m,
        }
    except pint.DimensionalityError:
        if reference_values:
            normalizing_quantities = {
                "r": reference_values["lref_major_radius"].to("meter").m,
                "b_field_tor": reference_values["bref_B0"].to("tesla").m,
                "n_e": reference_values["nref_electron"].to("meter**-3").m,
                "t_e": reference_values["tref_electron"].to("eV").m,
            }
        else:
            normalizing_quantities = {}

    normalizing_quantities = gkids.InputNormalizing(**normalizing_quantities)

    model = gkids.Model(
        **{
            "adiabatic_electrons": None,
            "include_a_field_parallel": int(numerics.apar),
            "include_b_field_parallel": int(numerics.bpar),
            "include_full_curvature_drift": None,
            "include_coriolis_drift": None,
            "include_centrifugal_effects": None,
            "collisions_pitch_only": None,
            "collisions_momentum_conservation": None,
            "collisions_energy_conservation": None,
            "collisions_finite_larmor_radius": None,
        }
    )

    flux_surface = convert_dict(
        {
            "r_minor_norm": geometry.rho,
            "ip_sign": geometry.ip_ccw,
            "b_field_tor_sign": geometry.bt_ccw,
            "q": geometry.q,
            "magnetic_shear_r_minor": geometry.shat,
            "pressure_gradient_norm": -geometry.beta_prime * aspect_ratio,
            "dgeometric_axis_r_dr_minor": geometry.shift,
            "dgeometric_axis_z_dr_minor": geometry.dZ0dr,
            "elongation": geometry.kappa,
            "delongation_dr_minor_norm": geometry.s_kappa
            * geometry.kappa
            / geometry.rho
            * aspect_ratio,
            "shape_coefficients_c": geometry.cn,
            "dc_dr_minor_norm": geometry.dcndr * aspect_ratio,
            "shape_coefficients_s": geometry.sn,
            "ds_dr_minor_norm": geometry.dcndr * aspect_ratio,
        },
        norms.imas,
    )

    flux_surface = gkids.FluxSurface(**flux_surface)

    species_all = convert_dict(
        {
            "velocity_tor_norm": pyro.local_species.electron.omega0,
            "shearing_rate_norm": pyro.numerics.gamma_exb,
            "beta_reference": numerics.beta,
            "debye_length_norm": 0.0,
            "angle_pol": np.linspace(-np.pi, np.pi, pyro.numerics.ntheta + 1),
        },
        norms.imas,
    )

    species_all = gkids.InputSpeciesGlobal(**species_all)

    species = [
        convert_dict(
            {
                "charge_norm": species.z,
                "mass_norm": species.mass,
                "density_norm": species.dens,
                "density_log_gradient_norm": species.inverse_ln,
                "temperature_norm": species.temp,
                "temperature_log_gradient_norm": species.inverse_lt,
                "velocity_tor_gradient_norm": species.domega_drho,
                "potential_energy_norm": None,
                "potential_energy_gradient_norm": None,
            },
            norms.imas,
        )
        for species in species_list
    ]

    species = [gkids.Species(**spec) for spec in species]

    collisionality = np.empty((len(species_list), len(species)))
    for isp1, spec1 in enumerate(species_list):
        for isp2, spec2 in enumerate(species_list):
            collisionality[isp1, isp2] = (
                spec1.nu.to(norms.imas).m
                * (spec2.dens / spec1.dens)
                * (spec2.z / spec1.z) ** 2
            )

    collisions = {"collisionality_norm": collisionality}

    collisions = gkids.Collisions(**collisions)

    data = {
        "ids_properties": ids_properties,
        "code": code,
        "normalizing_quantities": normalizing_quantities,
        "model": model,
        "flux_surface": flux_surface,
        "species_all": species_all,
        "species": species,
        "collisions": collisions,
    }

    if pyro.gk_output:
        # Assign new theta coord
        gk_output = pyro.gk_output.data.assign_coords(theta=mxh_theta_output)
        data["time"] = gk_output.time.data

        if not numerics.nonlinear:
            wavevector = []
            for ky in gk_output["ky"].data:
                for kx in gk_output["kx"].data:
                    wavevector.append(
                        {
                            "binormal_wavevector_norm": ky,
                            "radial_wavevector_norm": kx,
                            "eigenmode": get_eigenmode(
                                kx, ky, pyro.numerics.nperiod, gk_output, code_output
                            ),
                        }
                    )

            wavevector = [gkids.Wavevector(**wv) for wv in wavevector]

            linear = {"wavevector": wavevector}

            data["linear"] = gkids.GyrokineticsLinear(**linear)
        else:

            non_linear = {
                "binormal_wavevector_norm": gk_output["ky"].data,
                "radial_wavevector_norm": gk_output["kx"].data,
                "angle_pol": gk_output["theta"].data,
                "time_norm": gk_output["time"].data,
                "time_interval_norm": time_interval,
                "quasi_linear": 0,
                "code": code_output,
                "fields_4d": get_nonlinear_fields(gk_output),
            }

            non_linear.update(get_nonlinear_fluxes(gk_output, time_interval))

            data["non_linear"] = gkids.GyrokineticsNonLinear(**non_linear)

    for key in data.keys():
        setattr(ids, key, data[key])

    if pyro.gk_output:
        pyro.gk_output.to(getattr(norms, original_convention.name))
        pyro.gk_output.data = pyro.gk_output.data.assign_coords(
            theta=original_theta_output
        )

    return ids


def get_eigenmode(
    kx: float,
    ky: float,
    nperiod: int,
    gk_output: GKOutput,
    code_eigenmode: dict,
):
    """
    Returns dictionary with the structure of the Wavevector->Eigenmode IDS for a given kx and ky
    Parameters
    ----------
    kx : float
        Radial wavenumber to examine
    ky : float
        Bi-normal wavenumber to examine
    nperiod : int
        Number of poloidal turns
    gk_output : xr.Dataset
        Dataset of gk_output
    code_eigenmode : dict
        Dict of code inputs and status

    Returns
    -------
    eigenmode : dict
        Dictionary in the format of Eigenmode IDS
    """
    gk_output = gk_output.sel(kx=kx, ky=ky)

    gk_frame = "particle"

    if "mode" in gk_output:
        eigenmode = [
            {
                "poloidal_turns": nperiod,
                "angle_pol": gk_output["theta"].data,
                "time_norm": gk_output["time"].data,
                "initial_value_run": 1,
                "growth_rate_norm": gk_output["growth_rate"]
                .isel(time=-1, missing_dims="ignore")
                .sel(mode=mode)
                .data.m,
                "frequency_norm": gk_output["mode_frequency"]
                .isel(time=-1)
                .sel(mode=mode)
                .data.m,
                "growth_rate_tolerance": 0.0,
                "fields": get_linear_fields(gk_output),
                "linear_weights": get_linear_weights(gk_output.sel(mode=mode)),
                f"moments_norm_{gk_frame}": get_linear_moments(
                    gk_output.sel(mode=mode)
                ),
                "code": code_eigenmode,
            }
            for mode in gk_output["mode"].data
        ]
    else:
        eigenmode = [
            {
                "poloidal_turns": nperiod,
                "angle_pol": gk_output["theta"].data,
                "time_norm": gk_output["time"].data,
                "initial_value_run": 1,
                "growth_rate_norm": gk_output["growth_rate"]
                .isel(time=-1, missing_dims="ignore")
                .data.m,
                "frequency_norm": gk_output["mode_frequency"]
                .isel(time=-1, missing_dims="ignore")
                .data.m,
                "growth_rate_tolerance": gk_output.growth_rate_tolerance.data.m,
                "fields": get_linear_fields(gk_output),
                "linear_weights": get_linear_weights(gk_output),
                f"moments_norm_{gk_frame}": get_linear_moments(gk_output),
                "code": code_eigenmode,
            }
        ]

    eigenmode = [gkids.Eigenmode(**em) for em in eigenmode]

    return eigenmode


def get_linear_fields(gk_output: xr.Dataset):
    """
    Calculates "perturbed" quantities of field to be stored in the Wavevector->Eigenmode IDS
    Parameters
    ----------
    gk_output : xr.Dataset
        Dataset containing fields for a given kx and ky

    Returns
    -------
    weight : dict
        Dictionary of QL weights for different fields
    parity : dict
        Dictionary of parity for different fields
    norm : dict
        Dictionary of normalised eigenfunctions for different fields
    """

    theta_star = (
        np.abs(gk_output["phi"])
        .isel(time=-1, missing_dims="ignore")
        .argmax(dim="theta")
        .data
    )

    fields = {}

    for field in gk_output["field"].data:
        field_name = imas_pyro_field_names[field]
        field_data_norm = gk_output[field]

        # Normalised
        if field_data_norm.data.ndim == 1:
            fields[f"{field_name}_perturbed_norm"] = np.expand_dims(
                field_data_norm.data.m, axis=-1
            )
        else:
            fields[f"{field_name}_perturbed_norm"] = field_data_norm.data.m

        # Weights
        fields[f"{field_name}_perturbed_weight"] = np.reshape(
            np.sqrt((np.abs(field_data_norm) ** 2).integrate(coord="theta") / 2 * np.pi)
            .isel(time=-1, missing_dims="ignore")
            .data.m,
            (1,),
        )

        # Parity can have / 0 when a_par initialised as 0
        parity_data = (
            (
                np.abs(field_data_norm.roll(theta=theta_star).integrate(coord="theta"))
                / np.abs(field_data_norm).integrate(coord="theta")
            )
            .isel(time=-1, missing_dims="ignore")
            .data.m
        )

        fields[f"{field_name}_perturbed_parity"] = np.reshape(
            np.nan_to_num(parity_data), (1,)
        )

    fields = gkids.EigenmodeFields(**fields)

    return fields


def get_linear_weights(gk_output: GKOutput):
    """
    Gets linear weights needed for Wavevector->Eigenmode->linear_weights
    Parameters
    ----------
    gk_output : xr.Dataset
        Dataset containing fields for a given kx and ky

    Returns
    -------
    linear_weights : Dict
        Dictionary of linear weights
    """

    linear_weights = {}

    for flux in imas_pyro_flux_names.keys():
        for field in gk_output.field.data:
            linear_weights[
                f"{imas_pyro_flux_names[flux]}_{imas_pyro_field_names[field]}"
            ] = (
                gk_output[flux]
                .isel(time=-1, missing_dims="ignore")
                .sel(field=field)
                .data.m
            )

    linear_weights = gkids.Fluxes(**linear_weights)
    return linear_weights


def get_linear_moments(gk_output: GKOutput):
    """
    Gets moments needed for Wavevector->Eigenmode->Flux_moments
    Parameters
    ----------
    gk_output : xr.Dataset
        Dataset containing fields for a given kx and ky

    Returns
    -------
    flux_moments : Dict
        Dictionary of flux_moments
    """
    moments = {}

    for moment in imas_pyro_moment_names.keys():
        if moment in gk_output:
            moments[imas_pyro_moment_names[moment]] = gk_output[moment].data.m

    moments = gkids.MomentsLinear(**moments)

    return moments


def get_nonlinear_fields(gk_output: GKOutput):
    """
    Retreives full 4D nonlinear fields(kx, ky, theta, time)

    Parameters
    ----------
    gk_output : GKOutput
        Pyrokinetics GKOutput loaded with data

    Returns
    -------
    fields_4d: gkids.GyrokineticsFieldsNl4D
        4D fields GK IDS
    """

    fields = {}

    for field in gk_output["field"].data:
        field_name = imas_pyro_field_names[field]
        field_data_norm = gk_output[field]

        # Normalised
        fields[f"{field_name}_perturbed_norm"] = field_data_norm.data.m

    fields_4d = gkids.GyrokineticsFieldsNl4D(**fields)

    return fields_4d


def get_nonlinear_fluxes(gk_output: GKOutput, time_interval: [float, float]):
    """
    Loads nonlinear fluxes from pyrokinetics GKOutput

    Parameters
    ----------
     gk_output : GKOutput
        Pyrokinetics GKOutput loaded with data
    time_interval : float
        Final fraction of time over which to average fluxes

    Returns
    -------
    fluxes : Dict
        Dictionary of fluxes
        1) averaged over time
        2) summed over ky
    """

    fluxes = {}
    fluxes_2d_k_x_sum = {}
    fluxes_2d_k_x_k_y_sum = {}

    min_time = gk_output.time[-1].data * time_interval[0]
    max_time = gk_output.time[-1].data * time_interval[1]

    for pyro_flux, imas_flux in imas_pyro_flux_names.items():
        flux = gk_output[pyro_flux]

        time_average = flux.sel(time=slice(min_time, max_time)).mean(dim="time")
        sum_ky = flux.sum(dim="ky")

        for pyro_field in gk_output["field"].data:

            imas_field = imas_pyro_field_names[pyro_field]
            fluxes_2d_k_x_sum[f"{imas_flux}_{imas_field}"] = time_average.sel(
                field=pyro_field
            ).data.m
            fluxes_2d_k_x_k_y_sum[f"{imas_flux}_{imas_field}"] = sum_ky.sel(
                field=pyro_field
            ).data.m

    fluxes["fluxes_2d_k_x_sum"] = gkids.FluxesNl2DSumKx(**fluxes_2d_k_x_sum)
    fluxes["fluxes_2d_k_x_k_y_sum"] = gkids.FluxesNl2DSumKxKy(**fluxes_2d_k_x_k_y_sum)

    return fluxes
