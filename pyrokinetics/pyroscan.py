import numpy as np
from .constants import *
from .pyro import Pyro
import os
from itertools import product
from functools import reduce
import operator

class PyroScan():
    """
    Creates a dictionary of pyro objects

    Need a templates pyro object

    Dict of parameters to scan through
    { param : [values], }
    """

    def __init__(self,
                 pyro,
                 param_dict=None,
                 p_prime_type=0,
                 value_fmt='.2f',
                 value_separator='_',
                 parameter_separator='/',
                 file_name=None,
                 load_default_parameter_keys=True):

        # Keys showing map to parameter
        self.pyro_keys = {}

        if load_default_parameter_keys:
            self.load_default_parameter_keys()

        if isinstance(pyro, Pyro):
            self.pyro = pyro
        else:
            raise ValueError("PyroScan takes in a pyro object")

        if isinstance(param_dict, dict):
            self.param_dict = param_dict
        else:
            raise ValueError("PyroScan takes in a dict object")

        self.p_prime_type = p_prime_type

        self.value_fmt = value_fmt

        self.value_separator = value_separator

        if parameter_separator in ['/', '\\']:
            self.parameter_separator = os.path.sep
        else:
            self.parameter_separator = parameter_separator

        if file_name is not None:
            self.file_name = file_name
        else:
            self.file_name = pyro.gk_code.default_file_name

    def write(self, file_name=None, directory='.'):
        """
        Creates and writes GK input files for parameters in scan
        """

        if file_name is not None:
            self.file_name = file_name

        # Outer product of input dictionaries - could get very large
        outer_product = (dict(zip(self.param_dict, x)) for x in product(*self.param_dict.values()))

        # Check if parameters are in viable options
        for key in self.param_dict.keys():
            if key not in self.pyro_keys.keys():
                raise ValueError(f'Key {key} has not been loaded into pyro_keys')

        # Iterate through all runs and write output
        for run in outer_product:

            # Create file name for each run
            run_directory = directory + os.sep

            # Param value for each run written accordingly
            for param, value in run.items():
                single_run_name = f'{param}{self.value_separator}{value:{self.value_fmt}}'

                run_directory += single_run_name + self.parameter_separator

                # Get attribute and keys where param is stored
                attr_name, keys_to_param, = self.pyro_keys[param]

                # Get dictionary storing the parameter
                param_dict = getattr(self.pyro, attr_name)

                # Set the value given the dictionary and location of parameter
                set_in_dict(param_dict, keys_to_param, value)

            # Remove last instance of parameter_separator
            run_directory = run_directory[:-len(self.parameter_separator)]

            run_input_file = os.path.join(run_directory, self.file_name)

            self.pyro.write_gk_file(self.file_name, directory=run_directory)

    def add_parameter_key(self,
                          parameter_key=None,
                          parameter_attr=None,
                          parameter_location=None):
        """
        parameter_key: string to access variable
        parameter_attr: string of attribute storing value in pyro
        parameter_location: lis of strings showing path to value in pyro
        """

        if parameter_key is None:
            raise ValueError('Need to specify parameter key')

        if parameter_attr is None:
            raise ValueError('Need to specify parameter attr')

        if parameter_location is None:
            raise ValueError('Need to specify parameter location')

        dict_item = {parameter_key : [parameter_attr, parameter_location]}

        self.pyro_keys.update(dict_item)

    def load_default_parameter_keys(self):
        """
        Loads default parameters name into pyro_keys

        {param : ["attribute", ["key_to_location_1", "key_to_location_2" ]] }

        for example

        {'electron_temp_gradient': ["local_species", ['electron','a_lt']] }
        """

        self.pyro_keys = {}

        # ky
        parameter_key = 'ky'
        parameter_attr = 'numerics'
        parameter_location = ['ky']
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Electron temperature gradient
        parameter_key = 'electron_temp_gradient'
        parameter_attr = 'local_species'
        parameter_location = ['electron', 'a_lt']
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Electron density gradient
        parameter_key = 'electron_dens_gradient'
        parameter_attr = 'local_species'
        parameter_location = ['electron', 'a_ln']
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Deuterium temperature gradient
        parameter_key = 'deuterium_temp_gradient'
        parameter_attr = 'local_species'
        parameter_location = ['deuterium', 'a_lt']
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Deuterium density gradient
        parameter_key = 'deuterium_dens_gradient'
        parameter_attr = 'local_species'
        parameter_location = ['deuterium', 'a_ln']
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)


def get_from_dict(data_dict, map_list):
    """
    Gets item in dict given location as a list of string
    """
    return reduce(operator.getitem, map_list, data_dict)


def set_in_dict(data_dict, map_list, value):
    """
    Sets item in dict given location as a list of string
    """
    get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value
