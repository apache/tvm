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

tvm_tirx.tirx.script.nvshmem
============================

Generated namespace exposed by :mod:`tvm_tirx.tirx.script`.

.. currentmodule:: tvm_tirx.tirx.script.nvshmem

Functions
---------

.. autosummary::
   :toctree: generated/

   barrier_all
   fence
   getmem_nbi

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``getmem_nbi_block``
     - :obj:`tvm_tirx.tirx.script.nvshmem.getmem_nbi.block`
   * - ``getmem_nbi_warp``
     - :obj:`tvm_tirx.tirx.script.nvshmem.getmem_nbi.warp`

.. autosummary::
   :toctree: generated/

   my_pe
   n_pes
   putmem_nbi

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``putmem_nbi_block``
     - :obj:`tvm_tirx.tirx.script.nvshmem.putmem_nbi.block`
   * - ``putmem_nbi_warp``
     - :obj:`tvm_tirx.tirx.script.nvshmem.putmem_nbi.warp`

.. autosummary::
   :toctree: generated/

   putmem_signal_nbi

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``putmem_signal_nbi_block``
     - :obj:`tvm_tirx.tirx.script.nvshmem.putmem_signal_nbi.block`
   * - ``putmem_signal_nbi_warp``
     - :obj:`tvm_tirx.tirx.script.nvshmem.putmem_signal_nbi.warp`

.. autosummary::
   :toctree: generated/

   quiet
   signal_op
   wait_until

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.tirx.script.nvshmem.getmem_nbi <tirx.script.nvshmem.getmem_nbi>`
     - Generated operations.
   * - :doc:`tvm_tirx.tirx.script.nvshmem.putmem_nbi <tirx.script.nvshmem.putmem_nbi>`
     - Generated operations.
   * - :doc:`tvm_tirx.tirx.script.nvshmem.putmem_signal_nbi <tirx.script.nvshmem.putmem_signal_nbi>`
     - Generated operations.

.. toctree::
   :hidden:
   :maxdepth: 2

   tirx.script.nvshmem.getmem_nbi
   tirx.script.nvshmem.putmem_nbi
   tirx.script.nvshmem.putmem_signal_nbi
