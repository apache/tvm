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

Layout IR
=========

Layouts describe how logical tensor coordinates map to storage and execution
axes.  See :doc:`../layout` for the programming model and worked examples.

The primary classes are re-exported from ``tvm.tirx``, but are defined in this
module:

.. autoclass:: tvm.tirx.layout.Layout
   :members:
   :no-index:

.. autoclass:: tvm.tirx.layout.Axis
   :members:
   :no-index:

.. autoclass:: tvm.tirx.layout.Iter
   :members:
   :no-index:

.. autoclass:: tvm.tirx.layout.TileLayout
   :members:
   :no-index:

.. autoclass:: tvm.tirx.layout.ComposeLayout
   :members:
   :no-index:

``S[...]`` and ``R[...]`` build shard and replica layout specifications,
respectively.  Named axes such as ``laneid``, ``warpid``, ``tid_in_wg``,
``TLane``, and ``TCol`` are resolved lazily by this module.

.. automodule:: tvm.tirx.layout
   :members:
   :no-index:
   :exclude-members: Axis, ComposeLayout, Iter, Layout, TileLayout
