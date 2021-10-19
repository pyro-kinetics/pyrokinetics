from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


project_name = "pyrokinetics"
project_version = "0.0.1"

setup(
    name=project_name,
    version=project_version,
    packages=find_packages(),
    license="LGPL",
    author="Bhavin Patel",
    author_email="bhavin.s.patel@ukaea.uk",
    url="https://github.com/pyro-kinetics/pyrokinetics",
    description="Python package for running and analysing gyrokinetic simulations",
    long_description=read("README.md"),
    install_requires=[
        "numpy>=1.8",
        "f90nml>=1.3",
        "scipy>=1.6.3",
        "netCDF4>=1.5.6",
        "path>=15.1.2",
        "wheel >= 0.36",
        "freegs >= 0.6",
        "numpy >= 1.20.3",
        "sphinx == 3.5.4",
        "myst_parser == 0.14.0",
        "cleverdict >= 1.9.1",
        "xarray >= 0.10",
    ],
    platforms="any",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.6",
)
