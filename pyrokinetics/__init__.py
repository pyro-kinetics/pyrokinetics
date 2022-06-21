"""
Pyrokinetics
============

Python package for running and analysising GK data


.. moduleauthor:: Bhavin Patel <bhav.patel@ukaea.uk>

License
-------
Copyright 2021 Bhavin Patel and other contributors.
Email: bhav.patel@ukaea.uk
Pyrokinetics is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
Pyrokinetics is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License
along with Pyrokinetics.  If not, see <http://www.gnu.org/licenses/>.
"""

from .pyro import Pyro
from .pyroscan import PyroScan

import pathlib

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

# Location of bundled templates
from .templates import template_dir, gk_templates, eq_templates, kinetics_templates

__all__ = ["Pyro", "PyroScan", "template_dir", "__version__"]
