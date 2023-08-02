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

- 'Readables': These classes contain the data that is read and processed from files on
  disk.
- 'Readers': These classes read and return 'readables', and can also validate file
  types.

For example, :class:`Equilibrium` is a readable, and it is instantiated and returned by
reader classes such as :class:`EquilibriumReaderGEQDSK`. In this tutorial, we will set
up a readable class ``Foo``, along with a reader classes that reads from ``.csv`` files.

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

In order to make a readable class, we have sub-classed :class:`ReadableFromFileMixin` and
decorated the class with :func:`readable_from_file`. Both of these components are
necessary:

- :class:`ReadableFromFileMixin` contains the classmethods:
  - ``from_file``: Used as an alternative constructor. Creates a readable from a file.
  - ``supported_file_types``: Returns a list of file types that ``from_file`` can read.
  - ``reader``: A decorator used to tag other classes as 'readers' and associate them
    with this readable.
- :func:`readable_from_file` adds class-level attributes that are used by the
  aforementioned classmethods.

Having defined a 'readable' class, we can now define an associated reader:

.. code-block:: python

    # file: foo_csv_reader.py
    from pyrokinetics.file_utils import AbstractFileReader
    from .foo import Foo

    @Foo.reader("csv")
    class FooReaderCSV(AbstractFileReader):
        ...

Again, we have sub-classed a class from :mod:`file_utils` and wrapped the class with a
decorator:

- The decorator ``Foo.reader`` was added to ``Foo`` by :class:`ReadableFromFileMixin`.
  This 'registers' the reader with an associated readable via a key. This key should be
  the name of the file type we wish to read, or the name of the software that generated
  the file.
- :class:`AbstractFileReader` defines abstract methods
  :meth:`~AbstractFileReader.read_from_file` and
  :meth:`~AbstractFileReader.verify_file_type`. This means that sub-classes must provide
  a definition of these methods, or else Python will throw an error. The former method
  is used to read/process data from files, while the latter is used to determine whether
  a file is of the correct type.

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

Real `read_from_file` methods are likely to be much more complicated, and will likely
require further data processing. They may also require adding units to the readable's
input data. A good `verify_file_type` function should be very fast to run, and should
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
   :name: import-readers

   You _must_ ``import`` the module containing any file readers you write yourself,
   even if you don't use anything inside. If the module isn't imported, the
   ``@MyClass.reader`` decorator isn't used, and therefore the reader class isn't
   registered with the readable.

.. _sec-reader-internals:

The Messy Details
-----------------

.. _sec-gkinput-exception:

``GKInput``: The Exception
--------------------------

