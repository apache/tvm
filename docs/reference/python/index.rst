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

Namespaces
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Namespace
     - Common references
   * - :doc:`ir`
     - :doc:`Primitive expressions and operators <ir.prim>`, :doc:`IR passes <ir.transform>`
   * - :doc:`script`
     - :doc:`Parser <script.parser>`, :doc:`IR builder <script.ir_builder>`
   * - :doc:`tirx`
     - :doc:`Analysis <tirx.analysis>`, :doc:`Operators <tirx.op>`, :doc:`Passes <tirx.transform>`
   * - :doc:`backend`
     - :doc:`CUDA <backend.cuda>`, :doc:`CUDA operators <backend.cuda.op>`

.. toctree::
   :maxdepth: 1
   :titlesonly:

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

Additional namespaces
---------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   libinfo
