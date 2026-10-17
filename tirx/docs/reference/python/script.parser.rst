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

tvm_tirx.script.parser
======================

.. automodule:: tvm_tirx.script.parser
   :no-members:

.. currentmodule:: tvm_tirx.script.parser

Functions
---------

.. autosummary::
   :toctree: generated/

   register_namespace
   register_namespace_initializer
   from_source
   ir_module
   make_decorator
   make_macro_decorator

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``parse``
     - :obj:`tvm_tirx.script.parser.from_source`

.. autosummary::
   :toctree: generated/

   pyfunc

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.script.parser.protocol_registry <script.parser.protocol_registry>`
     - Static syntax policies keyed by registered builder namespace paths.

.. toctree::
   :hidden:
   :maxdepth: 2

   script.parser.protocol_registry
