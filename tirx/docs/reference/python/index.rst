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

tvm_tirx
========

.. automodule:: tvm_tirx
   :no-members:

.. currentmodule:: tvm_tirx

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``DataTypeCode``
     - :obj:`tvm_tirx.runtime.DataTypeCode`
   * - ``IRModule``
     - :obj:`tvm_tirx.ir.IRModule`

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``cpu``
     - :obj:`tvm_tirx.runtime.cpu`
   * - ``cuda``
     - :obj:`tvm_tirx.runtime.cuda`
   * - ``opencl``
     - :obj:`tvm_tirx.runtime.opencl`
   * - ``vulkan``
     - :obj:`tvm_tirx.runtime.vulkan`
   * - ``metal``
     - :obj:`tvm_tirx.runtime.metal`
   * - ``vpi``
     - :obj:`tvm_tirx.runtime.vpi`
   * - ``rocm``
     - :obj:`tvm_tirx.runtime.rocm`
   * - ``ext_dev``
     - :obj:`tvm_tirx.runtime.ext_dev`
   * - ``device_from_target``
     - :obj:`tvm_tirx.target.device_from_target`

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.runtime <runtime>`
     - TVM runtime namespace.
   * - :doc:`tvm_tirx.ir <ir>`
     - Common data structures across all IR variants.
   * - :doc:`tvm_tirx.ir.transform <ir.transform>`
     - Utilities for IR transformations.
   * - :doc:`tvm_tirx.script <script>`
     - TVMScript public namespace and registered language variant exports.
   * - :doc:`tvm_tirx.sym <sym>`
     - Symbolic analysis: integer bounds, simplification and pattern detection.
   * - :doc:`tvm_tirx.target <target>`
     - Target configuration and registry.
   * - :doc:`tvm_tirx.tirx <tirx>`
     - Namespace for Tensor-level IR
   * - :doc:`tvm_tirx.tirx.transform <tirx.transform>`
     - Namespace of all TIR transformations
   * - :doc:`tvm_tirx.codegen <codegen>`
     - Code generation related functions.
   * - :doc:`tvm_tirx.backend <backend>`
     - Backend-owned Python modules and load hooks.
   * - :doc:`tvm_tirx.support <support>`
     - tvm_tirx.support — Python helpers that integrate TVM with external CLIs
   * - :doc:`tvm_tirx.testing <testing>`
     - Utility Python functions for TVM testing
   * - :doc:`tvm_tirx.libinfo <libinfo>`
     - Library and include path information for TVM.

.. toctree::
   :hidden:
   :maxdepth: 2

   runtime
   ir
   ir.transform
   script
   sym
   target
   tirx
   tirx.transform
   codegen
   backend
   support
   testing
   libinfo
