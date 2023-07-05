import dataclasses
import json

from pyrokinetics import Numerics

import pytest
import time

def test_keys():
    numerics = Numerics()  # use defaults
    # Test you can set an existing key
    numerics["ntheta"] = 42
    assert numerics.ntheta == 42
    # Test you can't set a key that doesn't exist
    with pytest.raises(KeyError):
        numerics["n_theta"] = 42


def test_attrs():
    numerics = Numerics()  # use defaults
    # Test you can set an existing attr
    numerics.ntheta = 42
    assert numerics.ntheta == 42
    # Test you can't set a key that doesn't exist
    with pytest.raises(AttributeError):
        numerics.n_theta = 42


def test_write(tmp_path):
    numerics = Numerics()  # use defaults
    d = tmp_path / "numerics"
    d.mkdir(exist_ok=True)
    filename = d / "test_write.json"
    with open(filename, "w") as f:
        f.write(numerics.to_json())
    # Read as standard json and check values match
    with open(filename) as f:
        data = json.load(f)
    for key, value in data.items():
        if key == "_metadata":
            for k2, v2 in data["_metadata"].items():
                assert v2 == numerics._metadata[k2]
        else:
            assert value == numerics[key]


def test_roundtrip(tmp_path):
    numerics = Numerics()  # use defaults
    d = tmp_path / "numerics"
    d.mkdir(exist_ok=True)
    filename = d / "test_roundtrip.json"
    with open(filename, "w") as f:
        f.write(numerics.to_json())
    # Read as a new numerics
    with open(filename) as f:
        new_numerics = Numerics.from_json(f.read())
    assert numerics == new_numerics


def test_roundtrip_new_metadata(tmp_path):
    numerics = Numerics()  # use defaults
    d = tmp_path / "numerics"
    d.mkdir(exist_ok=True)
    filename = d / "test_roundtrip_new_metadata.json"
    with open(filename, "w") as f:
        f.write(numerics.to_json())
    # Ensure object_created metadata is different
    time.sleep(0.001)
    # Read as a new numerics, overwriting metadata
    with open(filename) as f:
        new_numerics = Numerics.from_json(
            f.read(),
            overwrite_metadata=True,
            overwrite_title="hello world",
        )
    data = dataclasses.asdict(new_numerics)
    for key, value in data.items():
        if key == "_metadata":
            for k2, v2 in data["_metadata"].items():
                print(k2, v2 , numerics._metadata[k2])
                if "software" in k2 or "session" in k2 or "object_type" in k2:
                    assert v2 == numerics._metadata[k2]
                else:
                    assert v2 != numerics._metadata[k2]
        else:
            assert value == numerics[key]
