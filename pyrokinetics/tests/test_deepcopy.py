from pyrokinetics import PyroScan
from pyrokinetics.examples import example_SCENE
import numpy as np
from copy import deepcopy
import operator
from functools import reduce


def get_from_dict(data_dict, map_list):
    """
    Gets item in dict given location as a list of string
    """
    return reduce(operator.getitem, map_list, data_dict)


def pyro_scan_check(pyro, param_dict, tmp_path):
    pyro_scan = PyroScan(pyro, param_dict, base_directory=tmp_path)
    pyro_scan.write()
    param = list(param_dict.keys())[0]

    (attr_name, keys_to_param) = pyro_scan.parameter_map[param]

    # Get attribute in Pyro storing the parameter
    pyro_attr = getattr(pyro, attr_name)

    original_value = get_from_dict(pyro_attr, keys_to_param)

    for pyro_run in pyro_scan.pyro_dict.values():
        scan_attr = getattr(pyro_run, attr_name)
        scan_value = get_from_dict(scan_attr, keys_to_param)
        assert id(scan_value) != id(original_value)


def test_deepcopy_pyro(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    copy_pyro = deepcopy(pyro)

    for name, old_component, new_component in zip(
        pyro.__dict__.keys(), pyro.__dict__.values(), copy_pyro.__dict__.values()
    ):

        if name not in ["eq", "kinetics", "gk_output"]:
            if not isinstance(old_component, (str, type(None), bool)):
                assert id(old_component) != id(new_component)

"""
def test_deepcopy_pyroscan_numerics(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    # Test value in Numerics
    param = "ky"
    values = np.arange(0.1, 0.3, 0.1)

    # Dictionary of param and values
    param_dict = {param: values}

    pyro_scan_check(pyro, param_dict, tmp_path)


def test_deepcopy_pyroscan_local_geometry(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    # Test value in Numerics
    param = "kappa"
    values = np.arange(0.1, 0.3, 0.1)

    # Dictionary of param and values
    param_dict = {param: values}

    pyro_scan_check(pyro, param_dict, tmp_path)


def test_deepcopy_pyroscan_local_species(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    # Test value in Numerics
    param = "electron_temp_gradient"
    values = np.arange(0.1, 0.3, 0.1)

    # Dictionary of param and values
    param_dict = {param: values}

    pyro_scan_check(pyro, param_dict, tmp_path)
"""