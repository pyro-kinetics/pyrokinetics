from typing import Union
from os import PathLike as os_PathLike
from numpy import integer, floating
from numpy.typing import ArrayLike  # noqa


Scalar = Union[float, integer, floating]
PathLike = Union[os_PathLike, str]
