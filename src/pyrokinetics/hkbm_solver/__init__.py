"""
hKBM eigenvalue solver with a GENE-compatible interface.

A fast reduced-model eigenvalue solver for the hybrid kinetic ballooning mode (local, linear,
electromagnetic, kinetic ions and electrons) that reads a GENE ``parameters`` file and writes
GENE's output files, so that pyrokinetics (or any GENE reader) can load its results:

    from pyrokinetics.hkbm_solver import run
    run("path/to/run_dir/parameters")        # writes parameters.dat, omega.dat, field.dat, ...

    from pyrokinetics import Pyro
    pyro = Pyro(gk_file="path/to/run_dir/parameters")
    pyro.load_gk_output()

Command line: ``hkbm-solve <run_dir>``.  See README.md in this folder.
"""

from .gene_io import Deck, UnsupportedDeck, run  # noqa: F401
