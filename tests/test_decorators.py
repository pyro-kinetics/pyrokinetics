from pyrokinetics import decorators

import pytest


def test_not_implemented():
    class FakeThing:
        @decorators.not_implemented
        def this_should_raise_error(self):
            return 1

    with pytest.raises(NotImplementedError) as error:
        assert FakeThing().this_should_raise_error() != 1

    assert "FakeThing" in str(error)
