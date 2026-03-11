import os
import sys
from scipy.constants import sigma
import torch
from torch._prims_common import validate_no_repeating_dims
from pyrokinetics import Pyro, PyroScan, template_dir
import pyrokinetics
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np
from pathlib import Path
from scipy.special import erf

# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d_3000/"  # Change this to a location in template diretory

models = [
    "growth_rate_log",
    "mode_frequency_log",
    "kperp2_phi_log",
    "kperp2_apa_log",
    "kperp2_bpar_log",
    "totIonFlux_log",
    "totElecFlux_log",
    "totPartFlux_log",
    "apa_phi_log",
    "bpar_phi_log",
    "sigmas_log",
]
