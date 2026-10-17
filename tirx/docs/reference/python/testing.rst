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

tvm_tirx.testing
================

.. automodule:: tvm_tirx.testing
   :no-members:

.. currentmodule:: tvm_tirx.testing

Functions
---------

.. autosummary::
   :toctree: generated/

   echo
   object_use_count
   test_raise_error
   run_with_gpu_lock
   assert_allclose
   device_enabled
   parameter
   identity_after
   terminate_self
   main
   np_dtype_from_str
   generate_random_array

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.testing.env <testing.env>`
     - Thin capability probes for test gating.
   * - :doc:`tvm_tirx.testing.locking <testing.locking>`
     - Helpers for tests that use exclusive machine resources.
   * - :doc:`tvm_tirx.testing.popen_pool <testing.popen_pool>`
     - Helper functions for popen_pool test cases.
   * - :doc:`tvm_tirx.testing.utils <testing.utils>`
     - TVM testing utilities

.. toctree::
   :hidden:
   :maxdepth: 2

   testing.env
   testing.locking
   testing.popen_pool
   testing.utils

.. raw:: html

   <span id="tvm-tirx-testing-env"></span>

.. raw:: html

   <span id="tvm-tirx-testing-locking"></span>

.. raw:: html

   <span id="tvm-tirx-testing-utils"></span>
