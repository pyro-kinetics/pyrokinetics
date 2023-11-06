from os import PathLike as os_PathLike
from typing import Union

from numpy import floating, integer
from numpy.typing import ArrayLike  # noqa

Scalar = Union[float, integer, floating]
PathLike = Union[os_PathLike, str]
