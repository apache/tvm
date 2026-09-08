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

Installation
============

There are two pieces:

- the **TIRx compiler** (``tvm.tirx``), which ships inside Apache TVM — this is
  the authoring, IR, lowering, and backend infrastructure. Compiling or running
  for a particular target also requires a TVM build with that target enabled;
- the optional **kernel library** (``tirx-kernels``), a collection of ready-made
  kernels and benchmark infrastructure built with TIRx.

Requirements
------------

- Python ≥ 3.10.
- For the CUDA programming guide and bundled kernels: a CUDA-enabled TVM build,
  an NVIDIA driver, and a compatible CUDA toolkit. The bundled kernels target
  Blackwell (``sm_100a``); TIRx itself also supports other target backends.

Install the TIRx compiler
-------------------------

Install the Apache TVM wheel (the TIRx compiler is the ``tvm.tirx`` module):

.. code-block:: bash

   pip install apache-tvm

The wheel is enough to inspect and author TIRx IR. Some CUDA workflows also
need NVIDIA's Python CUDA bindings, available through ``apache-tvm[cuda]``.
That extra does not enable CUDA in the TVM library and does not install a driver
or toolkit. To compile and run the CUDA examples, use a TVM build configured
with ``USE_CUDA=ON``; see :ref:`install TVM from source <install-from-source>`.

Verify:

.. code-block:: bash

   python -c "import tvm, tvm.tirx; print(tvm.__version__)"

Install the kernel library (optional)
-------------------------------------

Install the latest ``tirx-kernels`` release from PyPI:

.. code-block:: bash

   pip install tirx-kernels

Or install a checkout for development:

.. code-block:: bash

   git clone https://github.com/mlc-ai/tirx-kernels
   cd tirx-kernels
   pip install -e .

Apache TVM and PyTorch are externally managed runtime/compiler dependencies;
installing ``tirx-kernels`` does not install them. Put a TIRx-enabled TVM on
``PYTHONPATH`` and use a CUDA build of PyTorch matching the system. Individual
correctness tests and reference baselines need additional upstream projects.
For a source checkout, install their pinned, mutually compatible revisions with:

.. code-block:: bash

   python scripts/install_reference_dependencies.py

The kernel registry and reference dependency set evolve independently of TVM.
Use ``python -m tirx_kernels.registry --format json`` for the installed kernel
list, and consult the `tirx-kernels README
<https://github.com/mlc-ai/tirx-kernels#readme>`_ and
``reference-dependencies.json`` in that repository for current optional
requirements.
