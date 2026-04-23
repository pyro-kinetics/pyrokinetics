.. _sec-stella-input-versions:

============================================
 Read and write different stella input versions
============================================

Stella's Fortran namelist input format was restructured in v1.0: what were
~10 large namelists in earlier releases became ~30+ smaller, purpose-specific
namelists. Pyrokinetics supports three format families:

=========  ==============================================================
Version    Marker namelists
=========  ==============================================================
LEGACY     ``knobs``, ``parameters``, ``physics_flags``
PRE_V1     ``parameters_physics``, ``parameters_numerical``
V1         ``geometry_options``, ``gyrokinetic_terms``, ``kxky_grid_*`` …
=========  ==============================================================

Reading
=======

Pyrokinetics auto-detects the format on read — no user action required:

.. code-block:: python

   from pyrokinetics.gk_code import GKInputSTELLA

   gk = GKInputSTELLA("my_input.in")
   print(gk._format_version.value)   # "legacy", "pre_v1", or "v1"

All three formats expose the same physics through
:py:meth:`get_local_geometry`, :py:meth:`get_local_species`,
and :py:meth:`get_numerics`.

Writing
=======

The format of a written file is determined by the **template** that was used
to initialise the :py:class:`Pyro` object for stella. Two templates ship with
pyrokinetics:

=======================================  =============
Template                                 Format
=======================================  =============
``template_dir / "input.stella_v1"``     V1 (default)
``template_dir / "input.stella"``        PRE_V1
=======================================  =============

Write v1 (default)
------------------

.. code-block:: python

   from pyrokinetics import Pyro, template_dir

   pyro = Pyro(gk_file=template_dir / "input.cgyro", gk_code="CGYRO")
   pyro.write_gk_file(file_name="out.in", gk_code="STELLA")   # -> v1

Write pre-v1
------------

Pass the pre-v1 template explicitly when creating the stella context:

.. code-block:: python

   pyro.write_gk_file(
       file_name="out.in",
       gk_code="STELLA",
       template_file=template_dir / "input.stella",
   )

.. note::
   ``template_file`` is honoured only when the stella context is created by
   this call (i.e. ``gk_code="STELLA"`` is passed alongside). If a stella
   context already exists on the ``Pyro`` object, the existing template
   dictates the output format.

Upgrading legacy files to v1
============================

Pyrokinetics reads legacy files directly. If you need to emit a legacy input
as v1 on disk, read it with pyrokinetics and write it back — the output
defaults to v1:

.. code-block:: python

   pyro = Pyro(gk_file="old_stella.in", gk_code="STELLA")
   pyro.write_gk_file(file_name="upgraded_v1.in")

Alternatively, stella ships an authoritative converter at
``$STELLA/AUTOMATIC_TESTS/convert_input_files/convert_inputFile.py`` that
upgrades namelists directly without going through pyrokinetics.

Full worked example
===================

See ``docs/examples/example_cgyro_to_stella.py`` — it reads a CGYRO input and
writes both v1 and pre-v1 stella files, verifying the physics is preserved.
