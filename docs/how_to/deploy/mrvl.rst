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


Marvell Machine Learning Integration
====================================

1. Introduction
---------------
Marvell(R) supports a family of high performance Data Processing
Units (DPUs) with integrated compute, high speed I/O and workload
accelerators. These workload accelerators includes Marvell's
Machine Learning Inference Processor (MLIP), a highly optimized,
integrated inference engine.

TVM supports Marvell's MLIP using the "mrvl" library. This partitions and
compiles supported operations for accelerated execution on MLIP, or LLVM
for general compute.

For runtime, the library supports native execution on MLIP hardware
as well as Marvell's ML simulator (mrvl-mlsim).

The library supports Marvell's Octeon family of processors with ML accelarators.

This guide demonstrates building TVM with codegen and
runtime enabled. It also provides example code to compile and run
models using 'mrvl' runtime.

2. Building TVM with mrvl support
---------------------------------

2.1 Clone TVM repo
-------------------

Refer to the following TVM documentation for cloning TVM
https://tvm.apache.org/docs/install/from_source.html

2.2 Build and start the TVM - mrvl docker container
----------------------------------------------------

.. code:: bash

    ./docker/build.sh demo_mrvl bash  # Build the docker container
    ./docker/bash.sh tvm.demo_mrvl    # Load the docker image

3. Compiling a model using TVMC command line
--------------------------------------------
Models can be compiled and run for mrvl target using TVMC
which is optimized for performance.

Refer to the following TVMC documentation, for tvmc generic options.
https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html

Additional mrvl-specific options may be added as attributes if
necessary. The advanced usage is described in this document below.

3.1 TVMC Compilation Flow for a model
-------------------------------------

Refer to the following TVM documentation, for compilation flow
https://tvm.apache.org/docs/arch/index.html#example-compilation-flow


3.2. TVMC - Command line option(s): Syntax for mrvl target
----------------------------------------------------------

Compiling an ONNX model using the tvmc for mrvl target.

**Syntax:**

.. code:: python

    python3 -m tvm.driver.tvmc compile --target="mrvl, llvm"
        --target-llvm-<options>
        --target-mrvl-<options>
        --<tvm-generic-options>
        model_file.onnx

Following is an example TVMC Compile command for an ARMv9 core and
integrated MLIP cn10ka processor, using only 4 tiles in the block.

**Example:**

.. code:: python

    python3 -m tvm.driver.tvmc compile --target="mrvl, llvm" \
        --target-llvm-mtriple=aarch64-linux-gnu --target-llvm-mcpu=neoverse-n2 \
        --target-mrvl-num_tiles=4 \
        --cross-compiler aarch64-linux-gnu-gcc \
        --output model.tar \
        mnist-12.onnx

The runtime support for hardware acceleration is a WIP, it will be added in future PR.

3.3. TVMC Compiler: mrvl specific Command Line Options
------------------------------------------------------

.. code:: python

  --target-mrvl-mcpu
  --target-mrvl-num_tiles
  --target-mrvl-mattr

**Description of mrvl options**

* mcpu:
    The CPU class of Marvell(R) ML Inference Processor;
    possible values = {cn10ka, cnf10kb}; defaults to cn10ka

* num_tiles:
    Maximum number of tiles that may be used, possible values = {1,2,4,8}, defaults to 8

* mattr:
    Attributes for mrvl; possible values = {quantize, wb_pin_ocm}

    mattr specifies the data type, code generation options and optimizations.

    *List of supported attributes are:*

    **1. quantize**

    Specify the data type. Possible values = {fp16, int8}.
    Default is fp16, int8 is WIP and full support will be added in a future PR.

    **2. wb_pin_ocm**

    Optimize runtime by preloading a model's weights and bias into
    the on chip memory. Possible values = {0, 1}. Default is 0 (no preload)

4. Compile ONNX model for Simulator + LLVM / x86_64 target
----------------------------------------------------------

In the TVMC mrvl flow, the model is partitioned into Marvell and LLVM regions.
Building each partitioned Marvell subgraph generates serialized nodes.json and
const.json. Partitioned nodes.json is the representation of the model graph which is
suitable for the Marvell compiler (mrvl-tmlc). The compiler compiles the model graph to
generate the model binary with MLIP instructions.

**Model Compilation for Simulator + LLVM / x86_64 target**

.. code:: python

    python3 -m tvm.driver.tvmc compile --target="mrvl, llvm" \
        --target-mrvl-num_tiles=4 --output model.tar model.onnx

**Run TVM models on x86_64 host using MLIP Simulator**

Generated model binary is simulated using Marvell's MLIP Simulator(mrvl-mlsim).

.. code:: python

    python3 -m tvm.driver.tvmc run --inputs infer.npz --outputs predict.npz model.tar --number=0

5. Compiling a model using Python APIs
--------------------------------------

In addition to using TVMC, models can also be compiled and run using
TVM Python API. Below is an example to compile and run the MNIST model.

**Download MNIST model from the web**

.. code:: bash

    cd $HOME
    wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx

**Import the TVM and other dependent modules**

.. code:: python

    import tvm, onnx
    import numpy as np
    import tvm.relay as relay
    from tvm.contrib import graph_executor
    from tvm.relay.op.contrib.mrvl import partition_for_mrvl
    from tvm.relay.build_module import build
    from keras.datasets import mnist

**Load model onnx file**

.. code:: python

    onnx_model = onnx.load("mnist-12.onnx")

**Create a Relay graph from MNIST model**

.. code:: python

    shape_dict = {'Input3' : (1,1,28,28)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

**Define option dictionary and Partition the Model**

Annotate and partition the graph for mrvl. All operations which are supported
by the mrvl will be marked and offloaded to mrvl hardware accelerator. The rest of the
operations will go through the regular LLVM compilation and code generation for ARM.

.. code:: python

    tvm_target = "llvm"

    option_dict = {'num_tiles': 4}

    mod = partition_for_mrvl(mod, params, **option_dict)

**Build the Relay Graph**

Build the Relay graph, using the new module returned by partition_for_mrvl.

.. code:: python

    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.mrvl.options" : option_dict}):
        model_lib = relay.build(mod, tvm_target, params=params)

**Generate runtime graph of the model library**

.. code:: python

    dev = tvm.cpu()
    model_rt_graph = graph_executor.GraphModule(model_lib["default"](dev))

**Get test data and initialize model input**

.. code:: python

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    image = tvm.nd.array(test_X[0].reshape(1, 1, 28, 28).astype("float32") / 255)
    inputs_dict = {}
    inputs_dict["Input3"] = image
    model_rt_graph.set_input(**inputs_dict)

**Run Inference and print the output**

.. code:: python

    model_rt_graph.run()
    output_tensor = model_rt_graph.get_output(0).numpy()
    print (output_tensor)
