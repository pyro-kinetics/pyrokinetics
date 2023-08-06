import numpy as np
import xarray as xr
import os
from numpy.testing import assert_allclose
from pyrokinetics import template_dir
from pyrokinetics.diagnostics import sum_ky_spectrum, get_sat_params

template_path = os.path.join(template_dir, 'outputs/TGLF_transport/')

# Read number of fields, species, modes, ky, and moments
with open(os.path.join(template_path, 'out.tglf.QL_flux_spectrum'), 'r') as f:
    lines = f.readlines()
ntype, nspecies, nfield, nky, nmodes = list(map(int, lines[3].strip().split()))

# Read QL data
ql = []
for line in lines[6:]:
    line = line.strip().split()
    if any([x.startswith(("s", "m")) for x in line]):
        continue
    for x in line:
        ql.append(float(x))
QLw = np.array(ql).reshape(nspecies, nfield, nmodes, nky, ntype)

# Write QL data into Xarray and reshape
QL_data_array = xr.DataArray(
    data=QLw,
    dims=["species", 'field', 'mode', 'ky', "type"],
    coords={
        "species": np.arange(nspecies),
        "field": np.arange(nfield),
        "mode": np.arange(nmodes),
        "ky": np.arange(nky),
        "type": ['particle', 'energy', 'toroidal stress', 'parallel stress', 'exchange'],
    },
)

QL_data = QL_data_array.transpose('ky', 'mode', 'species', 'field', 'type').data
particle_QL = QL_data[:, :, :, :, 0]
energy_QL = QL_data[:, :, :, :, 1]
toroidal_stress_QL = QL_data[:, :, :, :, 2]
parallel_stress_QL = QL_data[:, :, :, :, 3]
exchange_QL = QL_data[:, :, :, :, 4]

# Read spectral shift and ave_p0 (only needed for SAT0)
with open(os.path.join(template_path, 'out.tglf.spectral_shift_spectrum'), 'r') as f:
    kx0_e = np.loadtxt(f, skiprows=5, unpack=True)
with open(os.path.join(template_path, 'out.tglf.ave_p0_spectrum'), 'r') as f:
    ave_p0 = np.loadtxt(f, skiprows=3, unpack=True)

# Read scalar saturation parameters
inputs = {}
with open(os.path.join(template_path, 'out.tglf.scalar_saturation_parameters'), 'r') as f:
    content = f.readlines()
    for line in content[1:]:
        line = line.strip().split('\n')
        if any([x.startswith(('!', 'UNITS', 'SAT_RULE', 'XNU_MODEL', 'ETG_FACTOR', 'R_unit', 'ALPHA_ZF', 'RULE')) for x in line]):
            if line[0].startswith('R_unit'):
                line = line[0].split(' = ')
                R_unit = float(line[1])
            continue
        line = line[0].split(' = ')
        inputs.setdefault(str(line[0]), float(line[1]))

# Read input.tglf
with open(os.path.join(template_path, 'input.tglf'), 'r') as f:
    content = f.readlines()
    for line in content[1:]:
        line = line.strip().split('\n')
        line = line[0].split(' = ')
        try:
            inputs.setdefault(str(line[0]), float(line[1]))
        except ValueError:
            continue
# Added inputs
inputs['UNITS'] = 'GYRO'
inputs['ALPHA_ZF'] = 1.0
inputs['RLNP_CUTOFF'] = 18.0
inputs['NS'] = int(inputs['NS'])
inputs['ALPHA_QUENCH'] = 0.0

# Get ky spectrum
with open(os.path.join(template_path, 'out.tglf.ky_spectrum'), 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:]).split()
    ky_spect = np.array(content, dtype=float)

# Get eigenvalue spectrum
with open(os.path.join(template_path, 'out.tglf.eigenvalue_spectrum'), 'r') as f:
    content = f.readlines()
    content = ''.join(content[2:]).split()
    gamma = []
    freq = []
    for k in range(nmodes):
        gamma.append(np.array(content[2 * k :: nmodes * 2], dtype=float))
        freq.append(np.array(content[2 * k + 1 :: nmodes * 2], dtype=float))
    gamma = np.array(gamma)
    freq = np.array(freq)
    gamma = xr.DataArray(gamma, dims=('mode_num', 'ky'), coords={'ky': ky_spect, 'mode_num': np.arange(nmodes) + 1})
    freq = xr.DataArray(freq, dims=('mode_num', 'ky'), coords={'ky': ky_spect, 'mode_num': np.arange(nmodes) + 1})
gammas = gamma.T
R_unit = np.ones(np.shape(gammas)) * R_unit

# Get potential spectrum
with open(os.path.join(template_path, 'out.tglf.field_spectrum'), 'r') as f:
    lines = f.readlines()
    columns = [x.strip() for x in lines[1].split(',')]
    nc = len(columns)
    content = ''.join(lines[6:]).split()
    tmpdict = {}
    for ik, k in enumerate(columns):
        tmp = []
        for nm in range(nmodes):
            tmp.append(np.array(content[ik + nm * nc :: nmodes * nc], dtype=float))
        tmpdict[k] = tmp
    for k, v in list(tmpdict.items()):
        potential = xr.DataArray(v, dims=('mode_num', 'ky'), coords={'ky': ky_spect, 'mode_num': np.arange(nmodes) + 1})
potential = potential.T

with open(os.path.join(template_path, 'out.tglf.gbflux'), 'r') as f:
    content = f.read()
    fluxes = list(map(float, content.split()))
    fluxes = np.reshape(fluxes, (4, -1))

sat_1 = sum_ky_spectrum(inputs['SAT_RULE'],
    ky_spect,
    gammas,
    ave_p0,
    R_unit,
    kx0_e,
    potential,
    particle_QL,
    energy_QL,
    toroidal_stress_QL,
    parallel_stress_QL,
    exchange_QL,
    **inputs)

expected_sat1 = fluxes[1]
python_sat1 = np.sum(np.sum(sat_1['energy_flux_integral'], axis=2), axis=0)

assert_allclose(python_sat1, expected_sat1, rtol=1e-3)

inputs['DRMINDX_LOC'] = 1.0
inputs['ALPHA_E'] = 1.0
inputs['VEXB_SHEAR'] = 0.0
inputs["SIGN_IT"] = 1.0
kx0epy, satgeo1, satgeo2, runit, bt0, bgeo0, gradr0, _, _, _, _ = get_sat_params(1, ky_spect, gammas.T, **inputs)
assert_allclose(kx0epy, kx0_e, rtol=1e-3)
assert_allclose(inputs['SAT_geo1_out'], satgeo1, rtol=1e-6)
assert_allclose(inputs['SAT_geo2_out'], satgeo2, rtol=1e-6)
assert_allclose(R_unit[0,0], runit,  rtol=1e-6)
assert_allclose(inputs['Bt0_out'], bt0, rtol=1e-6)
assert_allclose(inputs['grad_r0_out'], gradr0, rtol=1e-6)

if inputs['VEXB_SHEAR'] != 0.0:
    assert_allclose(inputs['B_geo0_out'], bgeo0, rtol=1e-6)