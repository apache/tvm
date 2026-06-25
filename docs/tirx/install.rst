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
  all you need to write and compile kernels;
- the optional **kernel library** (``tirx-kernels``), a set of ready-made GEMM
  and attention kernels built with TIRx.

Requirements
------------

- Python ≥ 3.10.
- An NVIDIA GPU with a recent CUDA toolkit. The bundled kernels target Blackwell
  (``sm_100a``); the compiler itself targets GPUs and accelerators more broadly.

Install the TIRx compiler
-------------------------

Install the Apache TVM wheel (the TIRx compiler is the ``tvm.tirx`` module):

.. code-block:: bash

   pip install apache-tvm

Verify:

.. code-block:: bash

   python -c "import tvm, tvm.tirx; print(tvm.__version__)"

Install the kernel library (optional)
-------------------------------------

``tirx-kernels`` provides prebuilt kernels (``fp16_bf16_gemm``,
``fp8_blockwise_gemm``, ``nvfp4_gemm``, ``flash_attention4``). It has no PyPI
wheel — install it from source:

.. code-block:: bash

   git clone https://github.com/mlc-ai/tirx-kernels
   cd tirx-kernels
   pip install -e .

Its runtime dependencies are **not** pulled from PyPI and must be available
separately (they are imported lazily, so ``import tirx_kernels`` and kernel
discovery work without them — they are only needed to actually compile/run a
kernel):

.. list-table::
   :header-rows: 1
   :widths: 18 24 58

   * - Dependency
     - Needed by
     - Notes
   * - ``tvm.tirx``
     - all kernels
     - the TIRx compiler (installed above, or put a source checkout's
       ``python/`` on ``PYTHONPATH``)
   * - ``torch``
     - all kernels
     - a CUDA build matching your GPU
   * - ``deep_gemm``
     - ``fp8_blockwise_gemm``
     - optional — quantization helpers and the reference baseline
   * - ``flashinfer``
     - ``nvfp4_gemm``
     - optional — quantization and the baseline

Build from source
-----------------

To develop TIRx or build the docs, build TVM from source and make it importable.
See :doc:`/install/from_source` for the full instructions; in short:

.. code-block:: bash

   export TVM_HOME=/path/to/tvm
   export TVM_LIBRARY_PATH=$TVM_HOME/build
   export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
   python -c "import tvm, tvm.tirx; print(tvm.__file__)"
