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

Language Reference
==================
This document provides references to
embedded languages and IRs in the TVM stack.

Introduction to Relay
---------------------

Relay is a functional, differentiable programming language
designed to be an expressive intermediate representation for machine
learning systems. Relay supports algebraic data types, closures,
control flow, and recursion, allowing it to directly represent more
complex models than computation graph-based IRs can.
Relay also includes a form of dependent typing using *type relations*
in order to handle shape analysis for operators with complex
requirements on argument shapes.

Relay is extensible by design and makes it easy for machine learning
researchers and practitioners to develop new large-scale program
transformations and optimizations.

The below pages describe the grammar, type system,
algebraic data types, and operators in Relay, respectively.

.. toctree::
   :maxdepth: 2

   relay_expr
   relay_type
   relay_adt
   relay_op
   relay_pattern

Hybrid Script
-------------

The below page describes the TVM hybrid script front-end,
which uses software emulation to support some constructs not
officially supported in TVM.

.. toctree::
   :maxdepth: 2

   hybrid_script
