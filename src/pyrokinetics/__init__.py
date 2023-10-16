"""
Pyrokinetics
============

Python package for running and analysising GK data


.. moduleauthor:: Bhavin Patel <bhavin.s.patel@ukaea.uk>

License
-------
Copyright 2023 UKAEA
Email: bhavin.s.patel@ukaea.uk
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

from .metadata import __version__
from .pyro import Pyro
from .pyroscan import PyroScan

# Location of bundled templates
from .templates import template_dir, gk_templates, eq_templates, kinetics_templates

# Equilibrium classes
from .equilibrium import (
    Equilibrium,
    FluxSurface,
    read_equilibrium,
    supported_equilibrium_types,
)

# Kinetics classes
from .kinetics import Kinetics, read_kinetics, supported_kinetics_types

# Numerics
from .numerics import Numerics

__all__ = [
    "__version__",
    "Pyro",
    "PyroScan",
    "template_dir",
    "Equilibrium",
    "FluxSurface",
    "read_equilibrium",
    "supported_equilibrium_types",
    "Kinetics",
    "read_kinetics",
    "supported_kinetics_types",
    "Numerics",
    "gk_templates",
    "eq_templates",
    "kinetics_templates",
]
