from pyrokinetics.diagnostics import Synthetic_highk_dbs
import numpy as np
import matplotlib.pyplot as plt

# inputs
diag = "highk"  # 'highk', 'dbs', 'rcdr', 'bes'
syn_filter = (
    "gauss"  # 'bt_slab', 'bt_scotty' for beam tracing, 'gauss' for Gaussian filter
)
Rloc = 2.89678  # [1.26] [m]      Bhavin 47107: 1.2689; David 22769: R=1.086, 1,137
Zloc = 0.578291  # [m]        Bhavin 47107: 0.1692; David 22769: 0.18
Kn0_exp = np.asarray([1, 1.5, 2, 2.5, 3] / 4)
Kb0_exp = np.asarray(
    [1, 2, 3, 4, 5] / 4
)  # np.asarray([2.701665 ])   # np.asarray([-2.701665])  # np.asarray([1.75, 6.903])       # [cm-1], Bhavin 47107: 6.903
wR = 2 / 213.75751  # [m]    local sim: do sinc function
wZ = 0.1  # 2/1711.94563       # [m]    wZ 0.02 MAST-U
eq_file = "/marconi/home/userexternal/jruizrui/files/jet/97090/transp/v04/jet_transp/97090V04.CDF"
kinetics_file = eq_file
simdir = "/marconi_work/FUA37_WPJET1/jruizrui/files/jet/97090/nl/t14.92_rho0.3_phia_deh/ky01_15_lowr"
savedir = simdir + "/syndiag"
if_save = 0
fsize = 22

syn_diag = Synthetic_highk_dbs(
    diag=diag,
    syn_filter=syn_filter,
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

for ik in np.arange(np.size(Kn0)):
    # synthetic diagnostic:
    [pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann] = syn_diag.get_syn_fspec(
        ik, 0.7, 1, savedir, if_save
    )

syn_diag.plot_syn()
