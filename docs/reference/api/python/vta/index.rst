..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

vta
===

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
