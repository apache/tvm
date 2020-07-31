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

Relay Arm|reg| Compute Library Integration
==========================================

Introduction
------------

Arm Compute Library (ACL) is an open source project that provides accelerated kernels for Arm CPU's
and GPU's. Currently the integration offloads operators to ACL to use hand-crafted assembler
routines in the library. By offloading select operators from a relay graph to ACL we can achieve
a performance boost on such devices.

Building with ACL support
-------------------------

The current implementation has two separate build options in cmake. The reason for this split is
because ACL cannot be used on an x86 machine. However, we still want to be able compile an ACL
runtime module on an x86 machine.

* USE_ARM_COMPUTE_LIB=ON/OFF - Enabling this flag will add support for compiling an ACL runtime module.
* USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME=ON/OFF/path-to-acl - Enabling this flag will allow the graph runtime to
  compute the ACL offloaded functions.

These flags can be used in different scenarios depending on your setup. For example, if you want
to compile an ACL module on an x86 machine and then run the module on a remote Arm device via RPC, you will
need to use USE_ARM_COMPUTE_LIB=ON on the x86 machine and USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME=ON on the remote
AArch64 device.

Usage
-----

.. note::

    This section may not stay up-to-date with changes to the API.

Create a relay graph. This may be a single operator or a whole graph. The intention is that any
relay graph can be input. The ACL integration will only pick supported operators to be offloaded
whilst the rest will be computed via TVM. (For this example we will use a single
max_pool2d operator).

.. code:: python

    import tvm
    from tvm import relay

    data_type = "float32"
    data_shape = (1, 14, 14, 512)
    strides = (2, 2)
    padding = (0, 0, 0, 0)
    pool_size = (2, 2)
    layout = "NHWC"
    output_shape = (1, 7, 7, 512)

    data = relay.var('data', shape=data_shape, dtype=data_type)
    out = relay.nn.max_pool2d(data, pool_size=pool_size, strides=strides, layout=layout, padding=padding)
    module = tvm.IRModule.from_expr(out)


Annotate and partition the graph for ACL.

..code:: python

    from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
    module = partition_for_arm_compute_lib(module)


Build the Relay graph.

.. code:: python

    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(module, target=target)


Export the module.

.. code:: python

    lib_path = '~/lib_acl.so'
    cross_compile = 'aarch64-linux-gnu-c++'
    lib.export_library(lib_path, cc=cross_compile)


Run Inference. This must be on an Arm device. If compiling on x86 device and running on AArch64,
consider using the RPC mechanism. Tutorials for using the RPC mechanism:
https://tvm.apache.org/docs/tutorials/cross_compilation_and_rpc.html#sphx-glr-tutorials-cross-compilation-and-rpc-py

.. code:: python

    ctx = tvm.cpu(0)
    loaded_lib = tvm.runtime.load_module('lib_acl.so')
    gen_module = tvm.contrib.graph_runtime.GraphModule(loaded_lib['default'](ctx))
    d_data = np.random.uniform(0, 1, data_shape).astype(data_type)
    map_inputs = {'data': d_data}
    gen_module.set_input(**map_inputs)
    gen_module.run()


More examples
-------------
The example above only shows a basic example of how ACL can be used for offloading a single
Maxpool2D. If you would like to see more examples for each implemented operator and for
networks refer to the tests: `tests/python/contrib/test_arm_compute_lib`. Here you can modify
`infrastructure.py` to use the remote device you have setup.


Operator support
----------------
+--------------+-------------------------------------------------------------------------+
| Relay Node   | Remarks                                                                 |
+==============+=========================================================================+
| nn.conv2d    | fp32:                                                                   |
|              |   Simple: nn.conv2d                                                     |
|              |   Composite: nn.pad?, nn.conv2d, nn.bias_add?, nn.relu?                 |
|              |                                                                         |
|              | (only groups = 1 supported)                                             |
+--------------+-------------------------------------------------------------------------+
| qnn.conv2d   | uint8:                                                                  |
|              |   Composite: nn.pad?, nn.conv2d, nn.bias_add?, nn.relu?, qnn.requantize |
|              |                                                                         |
|              | (only groups = 1 supported)                                             |
+--------------+-------------------------------------------------------------------------+
| nn.maxpool2d | fp32, uint8                                                             |
+--------------+-------------------------------------------------------------------------+
| reshape      | fp32, uint8                                                             |
+--------------+-------------------------------------------------------------------------+

.. note::
    A composite operator is a series of operators that map to a single Arm Compute Library operator. You can view this
    as being a single fused operator from the view point of Arm Compute Library. '?' denotes an optional operator in
    the series of operators that make up a composite operator.


Adding a new operator
---------------------
Adding a new operator requires changes to a series of places. This section will give a hint on
what needs to be changed and where, it will not however dive into the complexities for an
individual operator. This is left to the developer.

There are a series of files we need to make changes to:
* `python/relay/op/contrib/arm_compute_lib.py` In this file we define the operators we wish to offload using the
`op.register` decorator. This will mean the annotation pass recognizes this operator as ACL
offloadable.
* `src/relay/backend/contrib/arm_compute_lib/codegen.cc` Implement `Create[OpName]JSONNode` method. This is where we
declare how the operator should be represented by JSON. This will be used to create the ACL module.
* `src/runtime/contrib/arm_compute_lib/acl_kernel.h` Implement `Create[OpName]Layer` method. This is where we
define how the JSON representation can be used to create an ACL function. We simply define how to
translate from the JSON representation to ACL API.
* `tests/python/contrib/test_arm_compute_lib` Add unit tests for the given operator.
