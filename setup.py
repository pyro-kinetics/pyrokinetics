from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

import pyrokinetics

setup(
    name="Pyrokinetics",
    version=pyrokinetics.__version__,
    packages=["pyrokinetics"],

    license="LGPL",
    author="Bhavin Patel",
    author_email='bhav.patel@ukaea.uk',
    url="https://github.com/Bhavin2107/pyrokinetics",
    description="Python package for running and analysing gyrokinetic simulations",

    long_description=read("README.md"),
    
    install_requires=['numpy>=1.8'],
    
    platforms='any',

    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Physics'
        ],
)
