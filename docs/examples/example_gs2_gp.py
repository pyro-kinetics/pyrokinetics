import os
import sys
import torch
from pyrokinetics import Pyro
from pyrokinetics.diagnostics.gs2_gp import GS2_GP
import numpy as np


Template_Path = "/home/Felix/Documents/Physics_Work/Project_Codes/GP_Model_Eval/GS2/Templates/r1.in"
pyro = Pyro(gk_file=Template_Path)

test = GS2_GP(pyro,"/home/Felix/Documents/Physics_Work/Project_Codes/UKAEAdigiLab_FullGyro/8d_loRes/")

print(f"model growth rate: {test.model_growth_rate}, and error: {test.model_growth_rate_error}")
print(f"model frequency: {test.model_frequency} and error: {test.model_frequency_error} ")