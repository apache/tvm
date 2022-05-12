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

Relay BNNS Integration
======================
**Author**: `Egor Churaev <https://github.com/echuraev>`_

Introduction
------------

Apple BNNS library is a collection of functions that can be used to construct neural networks
for inference (and train). It’s supported in macOS, iOS, tvOS, and watchOS. BNNS provides
primitives executed on all CPU supported on those platforms and optimized for high performance
and low-energy consumption. This integration will offload as many operators as possible from Relay to BNNS.

BNNS runtime is a part of platform API and available on all modern Apple operating systems.
Application using BNNS will not depends on any additional external dependencies.

BNNS functions uses Apple private hardware capabilities which are not exposed yet by Apple. Example
of such capabilities can be AMX Apple cpu extension.

This guide will demonstrate how to build TVM with BNNS codegen and runtime enabled. It will also provide example
code to compile and run models using BNNS runtime. Finally, we document the supported operators.

Building TVM with BNNS support
------------------------------

To turn on TVM BNNS codegen and TVM BNNS runtime you need to turn on the only USE_BNNS flag

* USE_BNNS=ON/OFF - This flag will enable compiling a network with offloading subgraphs to BNNS primitives
  and will link tvm library to the BNNS runtime module.

Enabling of this flag will cause to search the default Accelerate Frameworks on current target SDK.
The minimal versions of required SDK is macOS 11.0, iOS 14.0, tvOS 14.0 and watchOS 7.0.

Example setting in config.cmake file:

.. code:: cmake

    set(USE_BNNS ON)

BNNS partitioning of Relay graph
--------------------------------

Operations to be offloaded on BNNS execution must be annotated before passing of module for compilation.
All ops annotated by `partition_for_bnns` will be offloaded for BNNS execution. The rest of the ops
will go through the LLVM compilation and code generation.

Important note: BNNS support primitives only with constant weights. To satisfy this requirements we have
to map constants to related tensor abstraction in relay representation. To freeze tensors and operate
with them as constants you may need to call ONNX importer with special flag "freeze_params=True"
or performer binding manually. In general cases all relay importers don't do that by default.
For your convenience "partition_for_bnns" can do this for you if params dictionary is passed as the argument.

.. code:: python

    from tvm.relay.op.contrib.bnns import partition_for_bnns
    model = partition_for_bnns(model, params=params)


Input data layout for operations to be offloaded to BNNS execution
------------------------------------------------------------------

BNNS kernels support only planar format of input data. The partitioner will require to have NCHW input
layout for conv2d input.

To use BNNS integration for models with interleave input layout, they should be converted before
passing of module to `partition_for_bnns`. The layout conversion will happen only for explicitly
enumerated types of ops. It might happen that depending on topology there might be regular data reorder
around conv2d to interleave and planar layout. This will be reflected in performance penalties and affect
execution time. It is recommended to analyze the whole topology and extend below list to convert all
intermediate tensors to NCHW data layout.

Example of input layouts change:

.. code:: python

    # For models with NHWC input layout
    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"],
                                            "nn.bias_add": ["NCHW", "default"],
                                            "nn.relu": ["NCHW"]})(mod)


Example: Build and Deploy Mobilenet v2 1.0 with BNNS
----------------------------------------------------

Create a Relay graph from a MXNet Mobilenet v2 1.0 model.

.. code:: python

    import tvm
    from tvm import relay
    import mxnet
    from mxnet.gluon.model_zoo.vision import get_model

    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    block = get_model('mobilenetv2_1.0', pretrained=True)
    module, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)


Markup the parts of graphs to be offloaded to BNNS primitives. All ops which are supported by the BNNS
integration will be handled by BNNS invocations, the rest of the ops will go through the
regular TVM llvm compilation and code generation.

After that you need to compile new module with target corresponding to required Apple platform

.. code:: python

    from tvm.relay.op.contrib.bnns import partition_for_bnns

    # target for macOS Big Sur 11.1:
    target = "llvm -mtriple=x86_64-apple-darwin20.2.0"

    model = partition_for_bnns(model, params=params)  # to markup operations to be offloaded to BNNS
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target=target, params=params)

Export the module.

.. code:: python

    lib.export_library('compiled.dylib')


Load module and run inference on the target machine with TVM  built with ``USE_BNNS`` enabled

.. code:: python

    import tvm
    import numpy as np
    from tvm.contrib import graph_executor

    dev = tvm.cpu(0)
    loaded_lib = tvm.runtime.load_module('compiled.dylib')
    gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    input_data = np.random.uniform(0, 1, input_shape).astype(dtype)
    gen_module.run(data=input_data)



Operator support
----------------

+------------------------+------------------------------------------------------------------------------+
|       Relay Node       |              Remarks                                                         |
+========================+==============================================================================+
| nn.conv2d              |                                                                              |
+------------------------+------------------------------------------------------------------------------+
| nn.batch_norm          | Supported by BNNS integration only in nn.conv2d-batch_norm pattern           |
+------------------------+------------------------------------------------------------------------------+
| nn.dense               |                                                                              |
+------------------------+------------------------------------------------------------------------------+
| nn.batch_matmul        |                                                                              |
+------------------------+------------------------------------------------------------------------------+
| nn.bias_add            | Supported by BNNS integration only as a bias part of nn.conv2d or nn.dense   |
|                        | fusion                                                                       |
+------------------------+------------------------------------------------------------------------------+
| add                    | Supported by BNNS integration only as a bias part of nn.conv2d or nn.dense   |
|                        | fusion                                                                       |
+------------------------+------------------------------------------------------------------------------+
| nn.relu                | Supported by BNNS integration only as a part of nn.conv2d or nn.dense fusion |
+------------------------+------------------------------------------------------------------------------+
| nn.gelu                | Supported by BNNS integration only as a part of nn.conv2d or nn.dense fusion |
+------------------------+------------------------------------------------------------------------------+
