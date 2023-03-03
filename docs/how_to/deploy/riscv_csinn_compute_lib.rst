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

Relay CSI-NN2 Compute Library Integration
=========================================

Introduction
------------

`CSI-NN2 Compute Library <https://github.com/T-head-Semi/csi-nn2>`_ (CSINN2) is an open-source project
that provides hand-crafted assembler routines for RISC-V CPUs with vector extension. It is compatible with
RISC-V v0.7.1 and v1.0 vector extension instruction standards. This integration will look at how we can
accelerate CPU performance for RISC-V devices like XuanTie C906 in TVM using CSINN2. The idea is that by
converting operators from a relay graph to CSINN2 we can achieve faster inference times due to these routines.
The initial intention is that this will improve performance for FP32 models. Although, with further improvements
to the integration this will extend to quantized models and support for a wider range of operators.

Installing CSI-NN2 Compute Library
----------------------------------

TVM only supports latest version of CSINN2, there is a recommended way to build and install the required
libraries:

* Use the script located at `docker/install/ubuntu_download_csinn2_compute_lib.sh`. You can use this
  script for downloading CSINN2 source code, these will be installed to the location denoted by `install_path`.

Building with CSI-NN2 support
-----------------------------

The current implementation has two separate build options in CMake. The reason for this split is because
the optimized code for RISC-V cannot be used on an x86 machine. We can set the flag to decide to generate
code running on X86 or RISC-V.

* USE_CSINN=OFF/ON/path-to-CSINN2
   * OFF - disable CSINN2 support. (default)
   * ON - add support for compiling CSINN2 codegen.
   * path-to-CSINN2 - use a specific version of the CSI-NN2 compute library.

* USE_CSINN_DEVICE_RUNTIME=OFF/X86/C906
   * OFF - disable CSINN2 runtime support. (default)
   * X86 - compiling CSINN2 runtime for x86 device.
   * C906 - cross-compiling CSINN2 runtime for C906 device.

If you want to compile an CSINN2 module on an x86 machine and then run the module on a remote RISC-V device
via RPC, you will need to do as below.

Build TVM with flags
>>>>>>>>>>>>>>>>>>>>

**Set in your config.cmake file**

.. code:: cmake

    set(USE_OPENMP gnu)
    set(USE_CSINN /path/to/csi-nn2)
    set(USE_CSINN_DEVICE_RUNTIME X86)

**Execute on the command line**

.. code:: bash

    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make -j4

Build runtime and rpc for device
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Set in your config.cmake file**

.. code:: cmake

    set(USE_CPP_RPC ON)
    set(USE_LIBBACKTRACE OFF)
    set(USE_CSINN /path/to/csi-nn2)
    set(USE_CSINN_DEVICE_RUNTIME C906)

**Execute on the command line**

.. code:: bash

    mkdir build-rv
    cp cmake/config.cmake build-rv
    cd build-rv
    cmake ..
    make runtime tvm_rpc -j4

After building successfully, we need to copy tvm_rpc and libs which used to device

To start an RPC server, run the following command on your remote device
**Execute on the command line**

.. code:: bash

    ./tvm_rpc server --host=172.16.202.11(your device ip) --port=9090

or using QEMU

    qemu-riscv64 -cpu c906fdv -L /path/to/csi-nn2/tools/gcc-toolchain/sysroot/ ./tvm_rpc server --host=127.0.0.1 --port=9090


Usage
-----

.. note::

    This section may not stay up-to-date with changes to the API.

Create a relay graph. This may be a single operator or a whole graph. The intention is that any
relay graph can be input. The CSINN2 integration will only pick supported operators to be offloaded
whilst the rest will be computed via TVM. (For this example we will use a single conv2d operator).

.. code:: python

    import tvm
    from tvm import relay
    import numpy as np

    data_type = "float32"
    data_shape = (1, 3, 24, 24)
    weight_shape = (10, 3, 3, 3)
    strides = (2, 2)
    padding = (1, 1, 1, 1)
    layout = "NCHW"
    output_shape = (1, 10, 12, 12)

    data_exp = relay.var('data', shape=data_shape, dtype=data_type)
    weight = tvm.nd.array((np.random.uniform(size=weight_shape)).astype(data_type))
    weight_exp = weight_exp = relay.const(weight, dtype=data_type)
    out = relay.nn.conv2d(data_exp, weight_exp, strides, padding, data_layout=layout)
    mod = tvm.IRModule.from_expr(out)


Annotate and partition the graph for CSINN2.

.. code:: python

    from tvm.relay.op.contrib.csinn import partition_for_csinn
    mod = partition_for_csinn(mod)


Build the Relay graph.

.. code:: python

    target = tvm.target.Target(
        "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
    )
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target=target)


Export the module.

.. code:: python

    lib_path = "~/lib_csinn2.so"
    cross_compile = 'riscv64-unknown-linux-gnu-g++'
    lib.export_library(lib_path, cc=cross_compile)


Run Inference. This must be on an RISC-V device or QEMU. If compiling on x86 device and
running on RISC-V, consider using the RPC mechanism.

.. code:: python

    # change this to the IP address of your target device
    host = "127.0.0.1"
    port = 9090
    remote = tvm.rpc.connect(host, port)
    # upload the library to remote device and load it
    remote.upload(lib_path)
    rlib = remote.load_module("lib_csinn2.so")
    # create the remote runtime module
    dev = remote.cpu(0)
    module = graph_executor.graph_executor.GraphModule(rlib["default"](dev))
    # set input data
    data = tvm.nd.array((np.random.uniform(size=data_shape)).astype(data_type))
    input_dict = {"data":data}
    module.set_input(**input_dict)
    # run
    module.run()


More examples
-------------
The example above only shows a basic example of how CSINN2 can be used for offloading a single
Maxpool2D. If you would like to see more examples for each implemented operator and for
networks refer to the tests: `tests/python/contrib/test_csinn`. Here you can modify
`test_config.json` to configure how a remote device is created in `infrastructure.py` and,
as a result, how runtime tests will be run.

An example configuration for `test_config.json`:

* connection_type - The type of RPC connection. Options: local, tracker, remote.
* host - The host device to connect to.
* port - The port to use when connecting.
* target - The target to use for compilation.
* device_key - The device key when connecting via a tracker.
* cross_compile - Path to cross compiler e.g. riscv64-unknown-linux-gnu-g++.

.. code:: json

    {
      "connection_type": "remote",
      "host": "127.0.0.1",
      "port": 9090,
      "target": "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c",
      "device_key": "",
      "cross_compile": "riscv64-unknown-linux-gnu-g++"
    }


Operator support
----------------
+---------------+-------------------------------------+
| Relay Node    | Remarks                             |
+===============+=====================================+
|| nn.conv2d    || fp32:                              |
||              || Simple: nn.conv2d                  |
||              || Composite: nn.conv2d, nn.bias_add? |
+---------------+-------------------------------------+
|| nn.dense     || fp32:                              |
||              || Simple: nn.dense                   |
||              || Composite: nn.dense, nn.bias_add?  |
+---------------+-------------------------------------+
| nn.relu       | fp32                                |
+---------------+-------------------------------------+
| nn.max_pool2d | fp32                                |
+---------------+-------------------------------------+
| nn.avg_pool2d | fp32                                |
+---------------+-------------------------------------+
| nn.softmax    | fp32                                |
+---------------+-------------------------------------+


.. note::
    A composite operator is a series of operators that map to a single CSI-NN2 Compute Library operator.
    You can view this as being a single fused operator from the view point of CSI-NN2 Compute Library.
    '?' denotes an optional operator in the series of operators that make up a composite operator.


Adding a new operator
---------------------
Adding a new operator requires changes to a series of places. This section will give a hint on
what needs to be changed and where, it will not however dive into the complexities for an
individual operator. This is left to the developer.

There are a series of files we need to make changes to:

* `python/relay/op/contrib/csinn.py` In this file we define the operators we wish to offload using the
  `op.register` decorator. This will mean the annotation pass recognizes this operator as CSINN2 offloadable.
* `src/relay/backend/contrib/csinn/codegen.cc` Implement `Create[OpName]JSONNode` method. This is where we
  declare how the operator should be represented by JSON. This will be used to create the CSINN2 module.
* `src/runtime/contrib/csinn/csinn_json_runtime.cc` Implement `[OpName]` method. This is where we
  define how the JSON representation can be used to create an CSINN2 function. We simply define how to
  translate from the JSON representation to CSINN2 API.
* `tests/python/contrib/test_csinn` Add unit tests for the given operator.
