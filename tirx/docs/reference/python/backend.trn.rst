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

tvm_tirx.backend.trn
====================

.. automodule:: tvm_tirx.backend.trn
   :no-members:

.. currentmodule:: tvm_tirx.backend.trn

Functions
---------

.. autosummary::
   :toctree: generated/

   register_backend
   script_namespace
   script_namespaces

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.backend.trn.op <backend.trn.op>`
     - Trainium-owned NKI intrinsic Python wrappers.
   * - :doc:`tvm_tirx.backend.trn.axis <backend.trn.axis>`
     - TRN layout axes.
   * - :doc:`tvm_tirx.backend.trn.layout <backend.trn.layout>`
     - Trainium-specific TIRx layout helpers.
   * - :doc:`tvm_tirx.backend.trn.pipeline <backend.trn.pipeline>`
     - Trainium TIRX pipeline entrypoints.
   * - :doc:`tvm_tirx.backend.trn.script <backend.trn.script>`
     - Trainium TVMScript namespaces.
   * - :doc:`tvm_tirx.backend.trn.thread_scope <backend.trn.thread_scope>`
     - TRN thread-scope lookup.
   * - :doc:`tvm_tirx.backend.trn.tile <backend.trn.tile>`
     - Namespace reference.
   * - :doc:`tvm_tirx.backend.trn.transform <backend.trn.transform>`
     - Trainium-specific TIRX transformations.

.. toctree::
   :hidden:
   :maxdepth: 2

   backend.trn.op
   backend.trn.axis
   backend.trn.layout
   backend.trn.pipeline
   backend.trn.script
   backend.trn.thread_scope
   backend.trn.tile
   backend.trn.transform
