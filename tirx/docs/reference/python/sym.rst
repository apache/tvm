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

tvm_tirx.sym
============

.. automodule:: tvm_tirx.sym
   :no-members:

.. currentmodule:: tvm_tirx.sym

Classes
-------

.. autosummary::
   :toctree: generated/

   IntSet
   IntervalSet
   ModularSet
   ConstIntBound
   Analyzer
   ProofStrength
   CompareResult
   IterMapExpr
   IterMark
   IterSplitExpr
   IterSumExpr

Functions
---------

.. autosummary::
   :toctree: generated/

   estimate_region_lower_bound
   estimate_region_strict_bound
   estimate_region_upper_bound
   allow_uint_as_index
   deduce_bound
   detect_linear_equation
   detect_clip_bound
   detect_iter_map
   iter_map_simplify
   normalize_iter_map_to_expr
   normalize_to_iter_sum
   subspace_divide
   inverse_affine_iter_map

Namespaces
----------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Purpose
   * - :doc:`tvm_tirx.sym.analyzer <sym.analyzer>`
     - Arithmetic data structure and utility
   * - :doc:`tvm_tirx.sym.bound <sym.bound>`
     - Bound deduction.
   * - :doc:`tvm_tirx.sym.int_set <sym.int_set>`
     - Integer set.
   * - :doc:`tvm_tirx.sym.iter_affine_map <sym.iter_affine_map>`
     - Iterator (quasi)affine mapping patterns.
   * - :doc:`tvm_tirx.sym.pattern <sym.pattern>`
     - Detect common patterns.
   * - :doc:`tvm_tirx.sym.utils <sym.utils>`
     - Utilities for extracting values from symbolic expressions.

.. toctree::
   :hidden:
   :maxdepth: 2

   sym.analyzer
   sym.bound
   sym.int_set
   sym.iter_affine_map
   sym.pattern
   sym.utils
