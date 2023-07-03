.. _sec-writing-docs:

=======================
 Writing Documentation
=======================

The documentation for Pyrokinetics is written in `reStructuredText`_ (RST) using
the `Sphinx`_ documentation generator. One of the really useful things about RST
is that it is extensible by design. Sphinx adds more `roles`_ and `directives`_
that are useful for writing technical documentation. This page will summarise
the more important ones for Pyrokinetics.

.. _sec-install-sphinx:

Getting And Running Sphinx
==========================

Pyrokinetics uses a couple of Sphinx plugins, as well as the `Sphinx Book
Theme`_. You can easily install all of these with one command:

.. code:: console

    $ pip install --editable .[docs]

Here we're using ``--editable`` to make sure our local changes are reflected in
the installed package. You might prefer to use a virtual environment for this.

Running Sphinx is as simple as:

.. code:: console

    $ cd docs
    $ make html

This is exactly equivalent to running the following from the top-level directory:

.. important::

   There is currently one expected warning when building the docs:

      WARNING: Cannot resolve forward reference in type annotations of
      "pyrokinetics.units.PyroUnitRegistry.Unit": name 'Unit' is not defined

   If you get more warnings than this, you should fix them!

.. code:: console

    $ sphinx-build -b html docs/ docs/_build/html

You can then view the built documentation by opening
``docs/_build/html/index.html`` in your browser.


reStructuredText Gotchas
========================

The `reStructuredText Primer`_ covers the basics of using the RST markup. Here
we'll cover some of the common "gotchas" and more specific syntax.

Backticks
---------

The most common gotcha is the use of backtick. In Markdown, single backticks
are used for inline ``monospace``. To achieve the same in RST, we have to use
double backticks ````monospace````: ``monospace``.

.. This is a comment: you'll notice we use four backticks in order to get two
   literal backticks in the inline monospace syntax.

Instead, single backticks in RST are used for `roles`_:
``:ref:`sec-writing-docs``` links to the top of this section, for instance:
:ref:`sec-writing-docs`. The default role if you don't include the initial
``:role:`` part is the `any`_ role. This makes it easy to link to a section
(```sec-getting-started```: `sec-getting-started`), a module (```species```:
`species`), a class (```Pyro```: `Pyro`), and so on.

.. _sec-section-links:

Linking to Sections
-------------------

When writing documentation pages, you may want to link to other sections
including those in other files. The main thing here is that you will need to
manually put in a label above the section name. See the Sphinx docs on
`Cross-referencing arbitrary locations`_. Note that the developers recommend
explicitly using the ``:ref:`` role so that typos and mistakes can be more
easily caught.

By convention, we prefix these labels with ``sec-`` to make clear we're linking
to a section.

LaTeX and maths
---------------

To use LaTeX and equations in RST files and Python docstrings, use either the
`math role`_ for inline maths (``:math:`\pi```: :math:`\pi`), or the `math
directive`_ for using the whole line:

.. code:: rst

   .. math::

      (a + b)^2 = a^2 + 2ab + b^2

.. math::

   (a + b)^2 = a^2 + 2ab + b^2

Code blocks
-----------

The default syntax highlighting for `literal/code blocks <literal-blocks>`_ is
Python. You can specify a different language, for example ``console`` or
``text``:

.. code:: rst

   .. code:: console

      $ echo "hello world"

.. code:: console

    $ echo "hello world"


Docstrings
==========

We make use of Python's built-in :term:`docstring` facility for documenting code
in-source. However, in Pyrokinetics we use the `numpydoc`_ style for
docstrings. This is probably familiar to you from Numpy's docstrings.

At a minimum, it's useful to have a one line short summary, followed by the
``Parameters`` (or equivalently, ``Arguments``) section:

.. code:: python

    def some_function(x: int, y: float) -> List[float]:
        """Calculate something really impressive

        Parameters
        ----------
        x : type
            Description of parameter `x`.
        y
            Description of parameter `y` (with type not specified).
        """
        ...

Note that if you use type hints in the function signature, then you can usually
skip the type in the docstring, unless you add more information like expected
units.

.. caution::
   :name: returns-syntax

   The ``Returns`` `section syntax <return-section>`_ is a little different to
   the Parameters section! The name of the parameter is optional and *the type
   is required*. If you use a type hint for the return value, you can skip this
   section altogether.

Getting Fancier
===============

The `Kitchen Sink <kitchen-sink>`_ example in the Sphinx Book Theme docs shows
off lots of the features of this theme. You might find it useful to have a look
through to see how different features can be used, including admonitions (like
`the note <returns-syntax_>`_ above), images, tables, and citations.

.. _reStructuredText Primer:
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Sphinx: https://www.sphinx-doc.org/en/master/index.html
.. _roles: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html
.. _directives: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _Sphinx Book Theme: https://sphinx-book-theme.readthedocs.io/en/stable/
.. _any: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-any
.. _Cross-referencing arbitrary locations:
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#ref-role
.. _math role:
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#math
.. _math directive:
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-math
.. _literal-blocks:
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#literal-blocks
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _return-section:
   https://numpydoc.readthedocs.io/en/latest/format.html#returns
.. _kitchen-sink:
   https://sphinx-book-theme.readthedocs.io/en/stable/reference/kitchen-sink/index.html
