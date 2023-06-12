import pint
from typing import Dict, Optional
from datetime import datetime
from idspy_dictionaries import ids_gyrokinetics as gkids
import idspy_toolkit as idspy
from dicttoxml import dicttoxml

from pyrokinetics import __version__ as pyro_version
from ..normalisation import convert_dict
from ..pyro import Pyro
import git
from itertools import product
import numpy as np

imas_pyro_field_names = {
    "phi": "phi_potential",
    "apar": "a_field_parallel",
    "bpar": "b_field_parallel",
}

imas_pyro_moment_names = {
    "particle": "particles",
    "heat": "energy",
    "momentum": "momentum_tor_perpendicular",
}


def pyro_to_ids(
    pyro: Pyro,
    comment: str = None,
    name: str = None,
    format: str = "json",
    file_name: str = None,
    ref_dict: Dict = {},
):
    """

    Parameters
    ----------
    pyro
    comment
    name
    format
    file_name
    ref_dict

    Returns
    -------

    """

    pyro_ids_dict = pyro_to_imas_mapping(
        pyro,
        comment=comment,
        name=name,
        format="hdf5",
        file_name="test.ids",
        ref_dict=ref_dict,
    )

    # generate a gyrokinetics IDS
    # this initialises the root structure and fill the whole structure with IMAS default values
    ids = gkids.Gyrokinetics()
    idspy.fill_default_values_ids(ids)

    # Values that be set simply with setattr
    setattr_keys = [
        "ids_properties",
        "normalizing_quantities",
        "flux_surface",
        "model",
        "species_all",
        "code",
        "collisions",
    ]

    for attr_key in setattr_keys:
        ids_attr = getattr(ids, attr_key)
        pyro_dict = pyro_ids_dict[attr_key]
        for key, value in pyro_dict.items():
            setattr(ids_attr, key, value)
        print(ids_attr)
        print("")

    # Set up code library
    ids.code.library = gkids.Library(**pyro_ids_dict["code"]["library"])
    print(ids.code)
    print("")
    # Set up tag
    names = pyro_ids_dict["tag"]["names"]
    comments = pyro_ids_dict["tag"]["comments"]
    ids.tag = [
        gkids.EntryTag(name=name, comment=comment)
        for name, comment in zip(names, comments)
    ]
    print(ids.tag)
    print("")

    # Set up time
    ids.time = pyro_ids_dict["time"]
    print(ids.time)
    print("")

    # Set up species
    ids.species = [
        gkids.Species(**species_data) for species_data in pyro_ids_dict["species"]
    ]
    print(ids.species)
    print("")

    # Set up fluxes
    fluxes = pyro_ids_dict["fluxes_integrated_norm"]
    ids.fluxes_integrated_norm = [gkids.Fluxes(**flux) for flux in fluxes.values()]
    print(ids.fluxes_integrated_norm)
    print("")

    wavevector = pyro_ids_dict["wavevector"]
    ids.wavevector = [gkids.Wavevector(**wv) for wv in wavevector.values()]

    for wv in ids.wavevector:
        eigenmodes = wv.eigenmode
        wv.eigenmode = [gkids.Eigenmode(**eigenmode) for eigenmode in eigenmodes]

        for em in wv.eigenmode:
            flux_moments = em.fluxes_moments
            em.fluxes_moments = [
                gkids.FluxesMoments(**flux_moment) for flux_moment in flux_moments
            ]

            flux_key = list(flux_moments[0].keys())[0]
            for fm in em.fluxes_moments:
                flux_values = getattr(fm, flux_key)
                setattr(fm, flux_key, gkids.Fluxes(**flux_values))

    print(ids.wavevector)

    return ids


def ids_to_pyro(ids_path, file_format="HDF5"):

    ids = gkids.Gyrokinetics()
    idspy.fill_default_values_ids(ids)

    if file_format == "HDF5":
        idspy.hdf5_to_ids(ids_path, ids)


def pyro_to_imas_mapping(
    pyro,
    comment=None,
    name=None,
    time_interval=0.5,
    format="json",
    file_name=None,
    ref_dict: Dict = {},
    times={},
):
    """Return a JSON-compatible structure that conforms to the
    GKDB/IMAS/OMAS gyrokinetics schema as described in:

    https://gitlab.com/gkdb/gkdb/raw/master/doc/general/IOGKDB.pdf
    https://gafusion.github.io/omas/schema/schema_gyrokinetics.html

    Currently only a partial implementation.

    Requires species and geometry data to already exist
    """
    if comment is None:
        raise ValueError("A comment is needed for IMAS upload")

    pyro.switch_local_geometry("MXH")
    geometry = pyro.local_geometry

    aspect_ratio = geometry.Rmaj

    species_list = [pyro.local_species[name] for name in pyro.local_species.names]

    numerics = pyro.numerics
    norms = pyro.norms

    # Convert output to IMAS norm
    pyro.gk_output.to(norms.imas)

    gk_output = pyro.gk_output.data

    ids_properties = {
        "comment": comment,
        "homogeneous_time": 1,
        "provider": "pyrokinetics",
        "creation_date": str(datetime.now()),
        "version_put": None,
    }

    tag = {
        "names": [name],
        "comments": [comment],
    }

    # TODO If reference values don't exist, what to set to?
    try:
        normalizing_quantities = {
            "t_e": (1.0 * norms.tref).to("eV"),
            "n_e": (1.0 * norms.nref).to("meter**-3"),
            "r": (1.0 * norms.gene.lref).to("meter"),
            "b_field_tor": (1.0 * norms.bref).to("tesla"),
        }
    except pint.DimensionalityError:
        normalizing_quantities = {
            "t_e": ref_dict["tref"],
            "n_e": ref_dict["nref"],
            "r": ref_dict["lref"],
            "b_field_tor": ref_dict["bref"],
        }

    flux_surface = convert_dict(
        {
            "r_minor_norm": geometry.rho / aspect_ratio,
            "elongation": geometry.kappa,
            "delongation_dr_minor_norm": geometry.kappa
            * geometry.kappa
            / geometry.rho
            * aspect_ratio,
            "dgeometric_axis_r_dr_minor": geometry.shift,
            "dgeometric_axis_z_dr_minor": geometry.dZ0dr,
            "q": geometry.q,
            "magnetic_shear_r_minor": geometry.shat,
            "pressure_gradient_norm": geometry.beta_prime * aspect_ratio,
            "ip_sign": geometry.ip_ccw,
            "b_field_tor_sign": geometry.bt_ccw,
            "shape_coefficients_c": geometry.cn,
            "shape_coefficients_s": geometry.sn,
            "dc_dr_minor_norm": geometry.dcndr * aspect_ratio,
            "ds_dr_minor_norm": geometry.dcndr * aspect_ratio,
        },
        norms.imas,
    )

    time = gk_output.time.data

    model = {
        "include_centrifugal_effects": None,
        "include_a_field_parallel": numerics.apar,
        "include_b_field_parallel": numerics.bpar,
        "include_full_curvature_drift": None,
        "collisions_pitch_only": None,
        "collisions_momentum_conservation": None,
        "collisions_energy_conservation": None,
        "collisions_finite_larmor_radius": None,
        "non_linear_run": int(numerics.nonlinear),
        "time_interval_norm": [max(time.data) * time_interval, max(time.data)],
    }

    species_all = convert_dict(
        {
            "beta_reference": numerics.beta,
            "velocity_tor_norm": pyro.local_species.electron.vel,
            "zeff": pyro.local_species.zeff,
            "debye_length_reference": None,
            "shearing_rate_norm": None,
        },
        norms.imas,
    )

    species = [
        convert_dict(
            {
                "charge_norm": species.z,
                "mass_norm": species.mass,
                "temperature_norm": species.temp,
                "temperature_log_gradient_norm": species.inverse_lt,
                "density_norm": species.dens,
                "density_log_gradient_norm": species.inverse_ln,
                "velocity_tor_gradient_norm": species.inverse_lv,
            },
            norms.imas,
        )
        for species in species_list
    ]

    collisions = {
        "collisionality_norm": [
            float(species.nu.to(norms.imas).m) for species in species_list
        ]
    }

    # Nonlinear fluxes
    fluxes_integrated_norm = {
        species.name: {
            "particles_phi_potential": None,
            "particles_a_field_parallel": None,
            "particles_b_field_parallel": None,
            "energy_phi_potential": None,
            "energy_a_field_parallel": None,
            "energy_b_field_parallel": None,
            "momentum_tor_parallel_phi_potential": None,
            "momentum_tor_parallel_a_field_parallel": None,
            "momentum_tor_parallel_b_field_parallel": None,
            "momentum_tor_perpendicular_phi_potential": None,
            "momentum_tor_perpendicular_a_field_parallel": None,
            "momentum_tor_perpendicular_b_field_parallel": None,
        }
        for species in species_list
    }

    # Linear moments of dist fn
    moments_norm_particle = {
        "density": None,
        "j_parallel": None,
        "pressure_parallel": None,
        "pressure_perpendicular": None,
        "heat_flux_parallel": None,
        "v_parallel_energy_perpendicular": None,
        "v_perpendicular_square_energy": None,
    }

    code_eigenmode = {"name": pyro.gk_code, "output_flag": -1}

    # TODO how does this work for nonlinear runs?
    if numerics.nonlinear:
        wavevector = {
            f"kx_{i[0]}_ky_{i[0]}": {
                "radial_component_norm": i[0],
                "binormal_component_norm": i[1],
                "eigenmode": None,
            }
            for i in product(gk_output.kx.data, gk_output.ky.data)
        }
    else:
        wavevector = {
            f"kx_{gk_output.kx.data[0]}_ky_{gk_output.ky.data[0]}": {
                "radial_component_norm": gk_output.kx.data[0],
                "binormal_component_norm": gk_output.ky.data[0],
                "eigenmode": None,
            }
        }

    xml_gk_input = dicttoxml(pyro.gk_input.data)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    code_library = {
        "name": "pyrokinetics",
        "commit": sha,
        "version": pyro_version,
        "repository": "https://github.com/pyro-kinetics/pyrokinetics",
        "parameters": xml_gk_input,
    }

    # TODO Need to have parameters in XML format?
    code = {
        "name": pyro.gk_code,
        "commit": None,
        "version": None,
        "repository": None,
        "parameters": xml_gk_input,
        "output_flag": -1,
        "library": code_library,
    }

    data = {
        "ids_properties": ids_properties,
        "tag": tag,
        "normalizing_quantities": normalizing_quantities,
        "flux_surface": flux_surface,
        "model": model,
        "species_all": species_all,
        "species": species,
        "collisions": collisions,
        "wavevector": wavevector,
        "fluxes_integrated_norm": fluxes_integrated_norm,
        "code": code,
        "time": time,
    }

    if not gk_output:
        return data

    nperiod = pyro.numerics.nperiod

    fluxes_integrated_norm_dict = {}
    if numerics.nonlinear:
        fluxes_integrated_norm_dict["particle"] = (
            gk_output["particle"].where(time > time_interval * time).mean(dim="time")
        )
        fluxes_integrated_norm_dict["heat"] = (
            gk_output["heat"].where(time > time_interval * time).mean(dim="time")
        )
        fluxes_integrated_norm_dict["momentum"] = (
            gk_output["momentum"].where(time > time_interval * time).mean(dim="time")
        )

        if "ky" in fluxes_integrated_norm_dict["particle"].coords:
            fluxes_integrated_norm_dict["particle"] = fluxes_integrated_norm_dict[
                "particle"
            ].sum(dim="ky", keep_attrs=False)
            fluxes_integrated_norm_dict["heat"] = fluxes_integrated_norm_dict[
                "heat"
            ].sum(dim="ky")
            fluxes_integrated_norm_dict["momentum"] = fluxes_integrated_norm_dict[
                "momentum"
            ].sum(dim="ky")

        for species in species_list:
            for field, field_name in imas_pyro_field_names.items():
                if getattr(numerics, field):
                    for flux, moment_name in imas_pyro_moment_names.items():
                        flux_data = fluxes_integrated_norm_dict[flux]
                        fluxes_integrated_norm[species.name][
                            f"{moment_name}_{field_name}"
                        ] = float(
                            flux_data.sel(field=field, species=species.name).data.m
                        )

        for i in product(gk_output.kx, gk_output.ky):
            wavevector[f"kx_{i[0]}_ky_{i[1]}"]["eigenmode"] = [get_eigenmode(
                i[0], i[1], nperiod, gk_output, time, time_interval, code_eigenmode
            ) ]
    else:
        # Select eigenmode if eigensolver used
        if "mode" in gk_output.coords:
            wavevector[f"kx_{gk_output.kx.data[0]}_ky_{gk_output.ky.data[0]}"][
                "eigenmode"
            ] = [
                get_eigenmode(
                    gk_output.kx,
                    gk_output.ky,
                    nperiod,
                    gk_output.sel(mode=mode),
                    time,
                    time_interval,
                    code_eigenmode
                )
                for mode in gk_output.mode
            ]
        else:
            wavevector[f"kx_{gk_output.kx.data[0]}_ky_{gk_output.ky.data[0]}"][
                "eigenmode"
            ] = [
                get_eigenmode(
                    gk_output.kx, gk_output.ky, nperiod, gk_output, time, time_interval, code_eigenmode
                )
            ]

    return data


def get_eigenmode(kx, ky, nperiod, gk_output, time, time_interval, code_eigenmode):

    gk_output = gk_output.sel(kx=kx, ky=ky).squeeze(dim=("kx", "ky"))

    eigenmode = {
        "poloidal_turns": nperiod,
        "growth_rate_norm": gk_output["growth_rate"].isel(time=-1).data.m,
        "frequency_norm": gk_output["mode_frequency"].isel(time=-1).data.m,
        "growth_rate_tolerance": gk_output.growth_rate_tolerance,
        "poloidal_angle": gk_output["theta"].data,
        "time_norm": gk_output["time"].data,
        "initial_value_run": 1,
        "fluxes_moments": [{spec: None} for spec in gk_output.species.data],
    }

    weight, parity, norm = get_perturbed(gk_output)

    for field in gk_output["field"].data:
        field_name = imas_pyro_field_names[field]
        eigenmode[f"{field_name}_perturbed_weight"] = weight[field]
        eigenmode[f"{field_name}_perturbed_parity"] = parity[field]
        eigenmode[f"{field_name}_perturbed_norm"] = norm[field]

    eigenmode["fluxes_moments"] = get_flux_moments(gk_output, time, time_interval)

    eigenmode["code"] = code_eigenmode

    return eigenmode


def get_perturbed(gk_output):

    if gk_output.linear:
        field_squared = (
            gk_output["phi"].pint.dequantify() ** 2
            + gk_output["apar"].pint.dequantify() ** 2
            + gk_output["bpar"].pint.dequantify() ** 2
        )
        amplitude = np.sqrt(field_squared.integrate(coord="theta") / 2 * np.pi)
    else:
        amplitude = 1.0

    theta_star = np.abs(gk_output["phi"]).isel(time=-1).argmax(dim="theta").data
    phi_field_star = gk_output["phi"].isel(theta=theta_star)

    if gk_output.linear:
        phase = np.abs(phi_field_star) / phi_field_star
    else:
        phase = 1.0

    parity = {}
    weight = {}
    norm = {}
    for field in gk_output["field"].data:
        field_data_norm = gk_output[field] / amplitude * phase

        norm[field] = field_data_norm.data.m
        weight[field] = np.sqrt(
            (np.abs(field_data_norm) ** 2).integrate(coord="theta") / 2 * np.pi
        ).data.m
        parity[field] = (
            np.abs(field_data_norm.roll(theta=theta_star).integrate(coord="theta"))
            / np.abs(field_data_norm.integrate(coord="theta"))
        ).data.m

    return weight, parity, norm


def get_flux_moments(gk_output, time, time_interval):

    # TODO Code dependent here (particle/gyrocenter/gyrocenter_rotating_frame)
    gk_frame = "particle"

    pyro_flux_moments = {}

    for flux_mom in imas_pyro_moment_names.keys():
        pyro_flux_moments[flux_mom] = (
            gk_output[flux_mom]
            .where(time > time_interval * time)
            .isel(time=-1)
            .squeeze(drop=True)
        )
        if "ky" in pyro_flux_moments[flux_mom].dims:
            pyro_flux_moments[flux_mom] = pyro_flux_moments[flux_mom].sum(dim="ky")

    flux_moments = [{f"fluxes_norm_{gk_frame}": {}} for spec in gk_output.species]
    for i, spec in enumerate(gk_output.species.data):
        spec_flux_moments = {}
        for field in gk_output.field.data:
            field_name = imas_pyro_field_names[field]
            for moment in gk_output.moment.data:
                moment_name = imas_pyro_moment_names[moment]
                imas_moment_name = f"{moment_name}_{field_name}"
                spec_flux_moments[imas_moment_name] = (
                    pyro_flux_moments[moment].sel(field=field, species=spec).data.m
                )

        flux_moments[i][f"fluxes_norm_{gk_frame}"] = spec_flux_moments

    return flux_moments
