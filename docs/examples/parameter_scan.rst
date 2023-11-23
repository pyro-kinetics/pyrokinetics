===============
Parameter scans
===============


This example generates an input file from a JETTO output and then
writes a series of input files covering a scan in :math:`k_y\rho_s` 
and :math:`a/L_{Te}` for GS2. This is done by using a dictionary
stating the parameter name and values.

.. literalinclude:: example_JETTO_1dscan.py

:math:`k_y\rho_s` is a pre-defined parameter to scan through, but we 
can add in custom parameters using the `PyroScan.add_parameter_key` 
function in the `PyroScan` object which tells pyrokinetics how to find
the relevant parameter

More details on parameter scans can be found in `sec-nd-parameter-scans`
