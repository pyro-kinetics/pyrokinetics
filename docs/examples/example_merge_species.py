from pyrokinetics import Pyro, template_dir
from pint import UnitRegistry
import numpy as np

# data path
spr_path = '/home/ab1209/pyrokinetics/SPR-JETTO/qmin_264_imp'
eqb_file = spr_path + '/jetto.eqdsk_out'
kin_file = spr_path + '/jetto.jsp'
psi_n = 0.5


# load pyro object
pyro = Pyro(
        eq_file=eqb_file,
        kinetics_file=kin_file,
        )
kin = pyro.kinetics
eqb = pyro.eq
ureg = UnitRegistry()


# inform species
for count,species in enumerate(kin.species_data.keys()):

    charge = kin.species_data[f'{species}'].charge(psi_n*ureg.dimensionless)
    print(f'{species}, Z: {charge.magnitude}')


# check with users species to merge
spec_list = []
nspec = int(input("Enter number of species to merge : "))
print('Enter species name, e.g. electrons')
for i in range(nspec):
    spec_name = str(input())
    spec_list.append(spec_name)

print('merging following species: ', spec_list)
merge_spec = dict((ispec, kin.species_data[ispec]) for ispec in spec_list)


# merge species routine

sum_ns = sum([merge_spec[f'{ispec}'].dens(psi_n*ureg.dimensionless) for ispec in spec_list])

Z_m = sum([merge_spec[f'{ispec}'].charge(psi_n*ureg.dimensionless)
            * merge_spec[f'{ispec}'].dens(psi_n*ureg.dimensionless)
                for ispec in spec_list]
        ) / sum_ns

n_m = sum([merge_spec[f'{ispec}'].charge(psi_n*ureg.dimensionless)
            * merge_spec[f'{ispec}'].dens(psi_n*ureg.dimensionless)
                for ispec in spec_list]
        ) / Z_m


M_m = sum([merge_spec[f'{ispec}'].mass
            * merge_spec[f'{ispec}'].dens(psi_n*ureg.dimensionless)
                for ispec in spec_list]
        ) / sum_ns


R_Ln_m = sum([merge_spec[f'{ispec}'].charge(psi_n*ureg.dimensionless)
            * merge_spec[f'{ispec}'].dens(psi_n*ureg.dimensionless)
            * merge_spec[f'{ispec}'].get_norm_dens_gradient(psi_n*ureg.dimensionless)
                for ispec in spec_list]
        ) / (Z_m * n_m)


# print normalized merged information
ne = kin.species_data['electron'].dens(psi_n*ureg.dimensionless)
mD = kin.species_data['deuterium'].mass
print(f'merged charge = {Z_m}')
print(f'merged density = {n_m/ne}')
print(f'merged mass = {M_m/mD}')
print(f'merged density gradient = {R_Ln_m}')


