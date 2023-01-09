import pytest
import numpy as np
import operator
from copy import deepcopy
from functools import reduce

from pyrokinetics import Pyro, PyroScan, template_dir


@pytest.fixture
def default_pyro():
    pyro = Pyro(
        gk_file=template_dir / "input.cgyro",
        eq_file=template_dir / "test.geqdsk",
        kinetics_file=template_dir / "scene.cdf",
    )
    pyro.load_local(psi_n=0.5, local_geometry="Miller")
    return pyro


def get_from_dict(data_dict, map_list):
    """
    Gets item in dict given location as a list of string
    """
    return reduce(operator.getitem, map_list, data_dict)


def test_deepcopy_pyro(tmp_path, default_pyro):
    pyro = default_pyro
    copy_pyro = deepcopy(pyro)

    for name, old_component, new_component in zip(
        pyro.__dict__.keys(), pyro.__dict__.values(), copy_pyro.__dict__.values()
    ):

        if name not in ["eq", "kinetics", "gk_output"]:
            if not isinstance(old_component, (str, type(None), bool)):
                assert id(old_component) != id(new_component)


@pytest.mark.parametrize(
    "param,values",
    [
        ["ky", np.arange(0.1, 0.3, 0.1)],  # test numerics
        ["kappa", np.arange(0.1, 0.3, 0.1)],  # test local geometry
        ["electron_temp_gradient", np.arange(0.1, 0.3, 0.1)],  # test local species
    ],
)
def test_deepcopy_pyroscan(tmp_path, default_pyro, param, values):
    pyro = default_pyro

    # Dictionary of param and values
    param_dict = {param: values}

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
