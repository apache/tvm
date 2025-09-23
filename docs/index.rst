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

Apache TVM Documentation
========================

Welcome to the documentation for Apache TVM, a deep learning compiler that
enables access to high-performance machine learning anywhere for everyone.
TVM's diverse community of hardware vendors, compiler engineers and ML
researchers work together to build a unified, programmable software stack, that
enriches the entire ML technology ecosystem and make it accessible to the wider
ML community. TVM empowers users to leverage community-driven ML-based
optimizations to push the limits and amplify the reach of their research and
development, which in turn raises the collective performance of all ML, while
driving its costs down.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   get_started/overview
   install/index
   get_started/tutorials/quick_start
   get_started/tutorials/ir_module

.. toctree::
   :maxdepth: 1
   :caption: How To

   how_to/tutorials/e2e_opt_model
   how_to/tutorials/customize_opt
   how_to/tutorials/optimize_llm
   how_to/tutorials/cross_compilation_and_rpc
   how_to/dev/index

.. The Deep Dive content is comprehensive
.. we maintain a ``maxdepth`` of 2 to display more information on the main page.

.. toctree::
   :maxdepth: 2
   :caption: Deep Dive

   arch/index
   deep_dive/tensor_ir/index
   deep_dive/relax/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   reference/api/python/index
   reference/api/links

.. toctree::
   :maxdepth: 1
   :caption: About

   contribute/index
   reference/publications
   reference/security

.. toctree::
   :maxdepth: 1
   :caption: Index

   genindex
