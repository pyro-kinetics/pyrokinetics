from pyrokinetics.diagnostics import SyntheticHighkDBS
import numpy as np
import matplotlib.pyplot as plt

# inputs
diag = "highk"  # 'highk', 'dbs', 'rcdr', 'bes'
filter_type = (
    "gauss"  # 'bt_slab', 'bt_scotty' for beam tracing, 'gauss' for Gaussian filter
    )
Rloc = 2.89678  # [1.26] [m]      Bhavin 47107: 1.2689; David 22769: R=1.086, 1,137
Zloc = 0.578291  # [m]        Bhavin 47107: 0.1692; David 22769: 0.18
Kn0_exp = np.asarray([0.5, 1]) # np.asarray([1, 1.5, 2, 2.5, 3] ) / 8
Kb0_exp = np.asarray([0.1, 0.2]) #np.asarray( [1, 2, 3, 4, 5]) / 8  # np.asarray([2.701665 ])   # np.asarray([-2.701665])  # np.asarray([1.75, 6.903])       # [cm-1], Bhavin 47107: 6.903
wR = 0.1       #2 / 213.75751  # [m]    local sim: do sinc function
wZ = 0.05    # 2/1711.94563       # [m]    wZ 0.02 MAST-U
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

# synthetic diagnostic:
[pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann] = syn_diag.get_syn_fspec( 0.7, 1, savedir, if_save )

syn_diag.plot_syn()
