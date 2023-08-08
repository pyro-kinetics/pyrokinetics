.. default-role:: math
.. default-domain:: py
.. _sec-file-readers:

File Readers
============

Pyrokinetics can read many different file types, and often it can do so without the user
specifying which kind of file they wish to read:

.. code-block:: python

    from pyrokinetics import read_equilibrium

    # No need to specify that we're reading a G-EQDSK file!
    eq = read_equilibrium("my_eq.geqdsk")

This tutorial provides information on the tools Pyrokinetics uses to achieve this, and
how to extend Pyrokinetics to handle new file types.

Overview
--------

There are typically two main classes involved in the process of reading data from disk:

- *Readables*: These classes contain the data that is read and processed from files on
  disk.

- *Readers*: These classes read and return 'readables', and can also validate file
  types.

For example, :class:`~pyrokinetics.equilibrium.equilibrium.Equilibrium` is a readable,
and it is instantiated and returned by reader classes such as
:class:`~pyrokinetics.equilibrium.geqdsk.EquilibriumReaderGEQDSK`. In this tutorial, we
will set up a readable class ``Foo``, along with a reader classes that reads from
``.csv`` files.

To begin, we must mark ``Foo`` as being readable. This is achieved as so:

.. code-block:: python

    # file: foo.py
    import numpy as np
    from numpy.typing import ArrayLike

    # These components are needed to make something 'readable'
    from pyrokinetics.file_utils import (
        readable_from_file,
        ReadableFromFileMixin,
    )

    # readable_from_file is used as a decorator, while
    # ReadableFromFileMixin is used as a super-class.
    @readable_from_file
    class Foo(ReadableFromFileMixin):

        # __init__ should take raw data -- not a file path
        def __init__(self, x: ArrayLike, y: ArrayLike):
            if not np.array_equal(np.shape(x), np.shape(y)):
                raise ValueError("x and y must have the same shape")
            if not np.ndim(x) == 1:
                raise ValueError("x and y must be 1D arrays")
            self._x = np.asarray(x)
            self._y = np.asarray(y)

In order to make a readable class, we have sub-classed
:class:`~pyrokinetics.file_utils.ReadableFromFileMixin` and decorated the class with
:func:`~pyrokinetics.file_utils.readable_from_file`. Both of these components are
necessary:

- :class:`~pyrokinetics.file_utils.ReadableFromFileMixin` contains the classmethods:

   - ``from_file``: Used as an alternative constructor. Creates a readable from a file.

   - ``supported_file_types``: Returns a list of file types that ``from_file`` can read.

   - ``reader``: A decorator used to tag other classes as 'readers' and associate them
     with this readable.

- :func:`~pyrokinetics.file_utils.readable_from_file` adds class-level attributes that
  are used by the aforementioned classmethods.

Having defined a 'readable' class, we can now define an associated reader:

.. code-block:: python

    # file: foo_csv_reader.py
    from pyrokinetics.file_utils import AbstractFileReader
    from .foo import Foo

    @Foo.reader("csv")
    class FooReaderCSV(AbstractFileReader):
        ...

Again, we have sub-classed a class from :mod:`~pyrokinetics.file_utils` and wrapped the
class with a decorator:

- The decorator ``Foo.reader`` was added to ``Foo`` by
  :class:`~pyrokinetics.file_utils.ReadableFromFileMixin`. This 'registers' the reader
  with an associated readable via a key. This key should be the name of the file type we
  wish to read, or the name of the software that generated the file.

- :class:`~pyrokinetics.file_utils.AbstractFileReader` defines abstract methods
  :meth:`~pyrokinetics.file_utils.AbstractFileReader.read_from_file` and
  :meth:`~pyrokinetics.file_utils.AbstractFileReader.verify_file_type`. This means that
  sub-classes must provide a definition of these methods, or else Python will throw an
  error. The former method is used to read/process data from files, while the latter is
  used to determine whether a file is of the correct type.

We'll now demonstrate how we might implement these functions:

.. code-block:: python

    from pathlib import Path
    import pandas as pd

    @Foo.reader("csv")
    class FooReaderCSV(AbstractFileReader):

        # read_from_file should take a file path as a positional argument,
        # and any number of keyword arguments. Keyword arguments can be
        # passed on to this function via the 'from_file' method of Foo.
        def read_from_file(self, path: Path, y_col: str = "y") -> Foo:
            # Use pandas to read a csv and extract two columns
            df = pd.read_csv(path)
            return Foo(df["x"], df[y_col])

        # verify_file_type should check that the file provided is of the
        # correct type. This may include making sure that the file contains
        # any essential data. If the file is of the wrong type, an Exception
        # should be raised. Otherwise, the function should end normally.
        def verify_file_type(self, path: Path) -> None:
            # Use pandas to read csv, but without loading all rows.
            # It will throw an exception if the file can't be found,
            # or if it isn't readable as a csv file.
            df = pd.read_csv(path, nrows=1)
            # Also check that any required data is present. In this
            # case, we only need to check for the presence of the
            # column 'x'
            if not "x" in df:
                raise RuntimeError("Foo csv needs an 'x' column")
            # If we get here, it's probably a Foo csv. Exit normally
            # without returning.
            pass

Real ``read_from_file`` methods are likely to be much more complicated, and will likely
require further data processing. They may also require adding units to the readable's
input data. A good ``verify_file_type`` function should be very fast to run, and should
load/process the minimum amount of data in order to ensure the file is of the correct
type.

With these functions defined, and reader classes registered, we can now use the
classmethods ``supported_file_types`` and ``from_file``:

.. code-block:: python

   >>> foo = Foo.from_file("my_foo.csv", file_type="csv")
   >>> foo = Foo.from_file("my_foo.csv") # file_type isn't needed!
   >>> print(Foo.supported_file_types())
   ["csv"]

We'll explain in the next section why the ``file_type`` argument isn't strictly needed.

.. caution::
   :name: import-readers-caution

   You *must* ``import`` the module containing any file readers you write yourself,
   even if you don't use anything inside. If the module isn't imported, the
   ``@MyClass.reader`` decorator isn't used, and therefore the reader class isn't
   registered with the readable.

   The readers built in to Pyrokinetics are all imported via various ``__init__.py``
   files, so are pre-registered when you import anything from Pyrokinetics itself. If
   you wish to implement your own reader within a Python package, it is similarly
   recommended to import it in your ``__init__.py`` file, as this will register your
   reader when anything inside your package is imported.

.. _sec-reader-internals:

Internal Details
----------------

So how do the tools discussed in the previous section work to allow us to determine
a file type automatically and read a file via a single call to ``Readable.from_file``?
Internally, this is managed using a specialised 'factory' class.

A factory is a function/class that allows users to create objects without specifying
their exact types. They provide a common interface to the constructors of a collection
of related types. The way they typically work is as follows:

- A collection of related classes are defined: ``A1``, ``A2``, and ``A3``. These may be
  related via a (possibly abstract) super class ``A``, or they may be related simply by
  'duck typing', i.e. they all have similar constructor/function signatures.
- Each class we wish the factory to produce is assigned a 'key' by which they may be
  referenced: ``"A1"``, ``"A2"``, ``"A3"``. These classes are *registered* with the
  factory, e.g. ``my_factory.register("A1", A1)``.
- The factory can then be used to create new instances of each class by providing the
  registered key. ``my_factory.create("A1", *args, **kwargs)`` may be used as an
  alternative to ``A1(*args, **kwargs)``.

Some of the benefit of using factories over using classes directly are:

- The user doesn't need to know exact class names, and doesn't need to import each
  class they might want to build independently -- they only need to import the factory.
- We avoid long ``if..elif...else`` chains such as the following:

.. code-block:: python

    if condition_for_A1:
        return A1(*args, **kwargs)
    elif condition_for_A2:
        return A2(*args, **kwargs)
    elif condition_for_A3:
        return A3(*args, **kwargs)
    else:
        ...

- The factory can create objects based on other conditions instead of simply looking up
  a registered key, so in cases where it isn't clear which type the user might want to
  return, the factory can figure this out and return a suitable class for them.

The factories used to link readers and readables don't need to be imported directly, as
they are stored as class-level attributes on each readable. The decorator
:func:`~pyrokinetics.file_utils.readable_from_file` is responsible for setting this up
for each readable. Users don't need to interact with these factories directly, as the
decorator ``Readable.reader`` handles registration of new classes with the factory, and
the ``from_file(path, file_type)`` method handles the object creation process. For
readers and readables, this is a two step process:

- Use a factory to create the correct type of reader. This is determined by the optional
  ``file_type`` argument.
- Call that reader's ``read_from_file`` function using the provided ``path``.

The additional bit of magic in Pyrokinetics is provided by the ``verify_file_type``
functions defined by each reader class. If the user doesn't pass ``file_type`` to
``from_file``, the internal factory instead searches through each registered reader
class and calls ``verify_file_type`` for each reader in turn. If, for some reader, this
function exits normally without raising an exception, that reader it is then used to
read the provided file. This can take a long time if ``verify_file_type`` functions are
slow to execute, so it is best for these functions to be very short and not to perform
any unnecessary additional processing.

.. _sec-gkinput-reader:

``GKInput``: Both Reader and Readable
-------------------------------------

:class:`~pyrokinetics.gk_code.gk_input.GKInput` fits strangely into this scheme, as
while :class:`~pyrokinetics.gk_code.gk_input.GKInput` itself is a 'readable', it's
'readers' are its own subclasses. This is because the reader classes fill in their
attributes as a side effect of calling ``read_from_file``. These readers should usually be retained
after use, as they provide further functionality besides that offered by
``read_from_file``. The way these readers are
handled within Pyrokinetics differs compared to other reader/readable pairs, as
:class:`~pyrokinetics.pyro.Pyro` makes direct use of the private factory object within
:class:`~pyrokinetics.gk_code.gk_input.GKInput` to manage them.

This implementation may change in a later release.

.. caution::
   :name: gkinput-reader-caution

   The subclasses of :class:`GKInput` do not return ``self`` from ``read_from_file``,
   but rather a dict-like object containing the raw data from the file they read.
   Remember to keep the reader class around if you want to call any other functions!
