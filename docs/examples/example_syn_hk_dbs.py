from pyrokinetics.diagnostics import SyntheticHighkDBS
import numpy as np
import matplotlib.pyplot as plt
from pyrokinetics.units import ureg
  
"""
Example file for synthetic diagnostic. Use for producing synthetic frequency/k-spectra from gyrokinetic simulations

Define here inputs characteristic of diagnostic 
"""  
  
# inputs
diag = "highk"                      # 'highk', 'dbs', 'rcdr', 'bes'
filter_type = ("gauss")             # 'bt_slab', 'bt_scotty', 'gauss' 
Rloc = 2.89678 * ureg.meter         # 
Zloc = 0.578291 * ureg.meter        # [m]       
Kn0_exp = np.asarray([0.5, 1]) / ureg.centimeter      # [cm-1]
Kb0_exp = np.asarray([0.1, 0.2]) / ureg.centimeter    # [cm-1]
wR = 0.1 * ureg.meter       #
wZ = 0.05 * ureg.meter      # 
eq_file = "/marconi/home/userexternal/jruizrui/files/jet/97090/transp/v04/jet_transp/97090V04.CDF"
kinetics_file = eq_file
simdir = "/marconi_work/FUA37_WPJET1/jruizrui/files/jet/97090/nl/t14.92_rho0.3_phia_deh/ky01_15_lowr"
savedir = simdir + "/syndiag"
if_save = 0
fsize = 22

syn_diag = SyntheticHighkDBS(
    diag=diag,
    filter_type=filter_type,
    Rloc=Rloc,
    Zloc=Zloc,
    Kn0_exp=Kn0_exp,
    Kb0_exp=Kb0_exp,
    wR=wR,
    wZ=wZ,
    eq_file=eq_file,
    kinetics_file=kinetics_file,
    simdir=simdir,
    savedir=simdir,
    fsize=fsize,
)

# map k
syn_diag.mapk()

# apply synthetic diagnostic:
[pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann] = syn_diag.get_syn_fspec( 0.7, 1, savedir, if_save )

syn_diag.plot_syn()
