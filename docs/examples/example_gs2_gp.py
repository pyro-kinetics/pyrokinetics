import os
import sys
import torch
from pyrokinetics import Pyro
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np


Template_Path = "/home/Felix/Documents/Physics_Work/Project_Codes/GP_Model_Eval/GS2/Templates/r1.in"
pyro = Pyro(gk_file=Template_Path)

models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/UKAEAdigiLab_FullGyro/8d_loRes/"

models = [
            "growth_rate_log", "mode_frequency_log", "kperp2_phi_log", "kperp2_apa_log",
            "kperp2_bpar_log", "totIonFlux_log", "totElecFlux_log", "totPartFlux_log",
            "apa_phi_log", "bpar_phi_log"
        ]


my_models = gs2_gp(pyro=pyro, models_path=models_path, models=models)

gr = my_models.models.sel(model="mode_frequency_log_M32")
print(gr)
