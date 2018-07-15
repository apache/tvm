VTA API
=======

This document contains the python API to VTA compiler toolchain.

.. automodule:: vta

Hardware Information
--------------------

.. autofunction:: vta.Environment
.. autofunction:: vta.get_env

RPC Utilities
-------------

.. autofunction:: vta.reconfig_runtime
.. autofunction:: vta.program_fpga


Compiler API
------------
We program VTA using TVM, so the compiler API in vta package
is only a thin wrapper to provide VTA specific extensions.

.. autofunction:: vta.build_config
.. autofunction:: vta.build
.. autofunction:: vta.lower
