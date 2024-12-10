.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

Overview
========

Apache TVM is a machine learning compilation framework, following the principle of **Python-first development**
and **universal deployment**. It takes in pre-trained machine learning models,
compiles and generates deployable modules that can be embedded and run everywhere. Apache TVM also enables customizing optimization processes to introduce new optimizations, libraries, codegen
and more.

Key Principle
-------------

- **Python-first**: the optimization process is fully customizable in Python.
  It is easy to customize the optimization pipeline without recompiling the TVM stack.
- **Composable**: the optimization process is composable. It is easy to compose
  new optimization passes, libraries and codegen to the existing pipeline.

Key Goals
---------

- **Optimize** performance of ML workloads, composing libraries and codegen.
- **Deploy** ML workloads to a diverse set of new environments, including new runtime and new hardware.
- **Continuously improve and customize** ML deployment pipeline in Python by quickly customizing library dispatching,
  bringing in customized operators and code generation.

Key Flow
--------

Here is a typical flow of using TVM to deploy a machine learning model. For a runnable example,
please refer to :ref:`quick_start`

1. **Import/construct an ML model**

    TVM supports importing models from various frameworks, such as PyTorch, TensorFlow for generic ML models. Meanwhile, we can create models directly using Relax frontend for scenarios of large language models.

2. **Perform composable optimization** transformations via ``pipelines``

    The pipeline encapsulates a collection of transformations to achieve two goals:

    - **Graph Optimizations**: such as operator fusion, and layout rewrites.
    - **Tensor Program Optimization**: Map the operators to low-level implementations (both library or codegen)

    .. note::

        The two are goals but not the stages of the pipeline. The two optimizations are performed
        **at the same level**, or separately in two stages.

3. **Build and universal deploy**

    Apache TVM aims to provide a universal deployment solution to bring machine learning everywhere with every language with minimum runtime support. TVM runtime can work in non-Python environments, so it works on mobile, edge devices or even bare metal devices. Additionally, TVM runtime comes with native data structures, and can also have zero copy exchange with the existing ecosystem (PyTorch, TensorFlow, TensorRT, etc.) using DLPack support.
