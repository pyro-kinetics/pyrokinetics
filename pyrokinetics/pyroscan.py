import numpy as np
from .constants import *

class PyroScan(pyro,
               paramDict):

    """
    Creates a dictionary of pyro objects

    Need a base pyro object

    Dict of parameters to scan through
    { key : [values], }
    """

    params = paramDict.keys()

    

    
