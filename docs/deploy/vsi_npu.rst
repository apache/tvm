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

Relay VSI NPU Integration
==========================
**Author**: `Tang Jing <https://github.com/antkillerfarm>`_

Introduction
------------

Verisilicon’s Neural-network Processing Units(also called VSI NPU) applied on the edge broadly
with many SoC vendors. This integration will offload as many operators as possible from Relay
to VSI NPU.

Installing TIM-VX
-------------------

Tensor Interface Module for OpenVX(TIM-VX) is a library for optimized deep learning inference
on VSI NPU device. This integration is based on TIM-VX.

TIM-VX website:
https://github.com/VeriSilicon/TIM-VX

You can find install instructions in:
https://github.com/VeriSilicon/TIM-VX/blob/main/README.md


Building TVM with VSI NPU support
----------------------------------

Build TVM with TIM-VX(host build):

.. code:: bash

    mkdir host_compiler_build
    cd host_compiler_build
    cp ../cmake/config.cmake ./
    # NOTE: 
    # 1.Config llvm by set USE_LLVM to the llvm-config; (For example: llvm-config-10 on Ubuntu 20.04)
    # 2.Add set(USE_VSI_NPU ON) to config.cmake;
    # 3.Disable other backend to speed up build, if you wish.
    cmake -DTIM_VX_INSTALL_DIR=<full_path_to_tim_vx_install> ..
    make tvm -j12


Build TVM with TIM-VX(target build):

.. code:: bash

    mkdir target_runtime_build
    cd target_runtime_build
    cp ../cmake/config.cmake ./
    # add set(USE_VSI_NPU ON) to config.cmake, you can do it with cmake command option too
    cmake -DTIM_VX_INSTALL_DIR=<full_path_to_tim_vx_target_build_install_dir> \
         -DCMAKE_TOOLCHAIN_FILE=<path_to_cross_compile_toolchain.make> ..
    make runtime -j12


Build and Deploy Mobilenet_v2 with VSI NPU
----------------------------------------

Create a Relay graph from a Torchvision Mobilenet_v2 model.

.. code:: python

    import tvm
    from tvm import relay
    import torch
    import torchvision

    dummy_input = torch.rand(1, 3, 224, 224)
    model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    scripted_model = torch.jit.trace(model, dummy_input).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [('input', dummy_input.shape)])


Annotate and partition the graph for VSI NPU. All ops which are supported by the VSI NPU
integration will be marked and offloaded to VSI NPU. The rest of the ops will go through the
regular TVM compilation and code generation.

.. code:: python

    from tvm.relay.op.contrib.vsi_npu import partition_for_vsi_npu
    target_string = "llvm"
    kwargs = {"cc": "aarch64-linux-gnu-gcc", 'fcompile': False}
    disabled_passes = ["AlterOpLayout"]
    with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled_passes):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        lib = relay.build(mod, target_string, params=params)
        lib.export_library('NBG.so',  **kwargs)


NBG(network binary graph) is the executeble format for the VSI NPU, we can compile it from
host server and deployment it to a target.

Load module and run inference on the target machine.

.. code:: bash

    # make sure NPU driver installed and can work without error (check dmesg after you insmod galcore)
    # 0.Append tvm/python 
    export PYTHONPATH=<path/to/tvm/ptyon>:$PYTHONPATH
    # 1.Setup libraries
    export LD_LIBRARY_PATH=<path/to/versilicon/driver/sdk>:<path/to/tim-vx/target/install>:<path/to/tvm/target_runtime_build/>:$LD_LIBRARY_PATH
    # 2. start service on given TCP port
    python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090

Execute test from host.

.. code:: bash

    # 0. Set correct NPU target name for your device, you can learned this from your soc vendor
    export VSIMULATOR_CONFIG=PID_0x99
    # 1. Set up testcase, please refer model list from tests/python/contrib/test_vsi_npu/test_vsi_tflite_model_all.py
    export TFLITE_MODEL="<full/path/to/mobilenet_v1_1.0_224_quant.tflite>"
    # 2. Setup cross compile toolchain configuration 
    export PATH=<cross-compiler-path>:$PATH
    export CROSS_CC=<cross-compiler-binary-name>
    export ROOTFS=<rootfs-for-cross-compile>
    # 3. Remote service configuration
    export RPC_HOST=<target device ip address>
    export RPC_PORT=<TCP port exposed by the service>
    # debug purpose
    export MOD_PATH="<any/folder/can/write>"
    export MOD_NAME="NBG.so" # could be any name, for debug purpose
    # 4. Add TVM to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=<path/to/host_compiler_build/>
    # 5. Execute test
    python3 tests/python/contrib/test_vsi_npu/test_vsi_tflite_model_all.py


TVMC Guide
----------------------------------------

TVMC is also supported.

.. code:: bash

    export TVM_HOME=<tvm path>
    export PYTHONPATH=$TVM_HOME/python
    export TVM_LIBRARY_PATH=$TVM_HOME/host_compiler_build
    export VIVANTE_SDK_DIR=<VSI NPU driver sdk path>
    export LD_LIBRARY_PATH=<TIM-VX build path>/install/lib:$TVM_LIBRARY_PATH:$VIVANTE_SDK_DIR/drivers:$LD_LIBRARY_PATH
    export VSIMULATOR_CONFIG=VIP8000NANOSI_PLUS_PID0X9F
    export TARGET="vsi_npu, llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    export CC=<cross compile toolchain>
    export CC_OPTIONS=<cc option>
    python3 -m tvm.driver.tvmc compile ./mobilenet_v1_0.25_224_quant.tflite --target "$TARGET" -o tvmc.tar \
      --cross-compiler "$CC" --cross-compiler-options "$CC_OPTIONS"


TIM-VX Settings
----------------

There are some additional options which can be configured using environment variables.

* TIM-VX Log - Environment variable ``VSI_NN_LOG_LEVEL=5`` can be set to print TIM-VX Log.
* VSI NPU Driver Log - Environment variable ``VIV_VX_DEBUG_LEVEL=1`` can be set to print
  VSI NPU Driver Log.


Operator support
----------------
+------------------------+------------------------------------+
|       Relay Node       |              Remarks               |
+========================+====================================+
| nn.relu                |                                    |
+------------------------+------------------------------------+
| nn.leaky_relu          |                                    |
+------------------------+------------------------------------+
| sigmoid                |                                    |
+------------------------+------------------------------------+
| logical_and            |                                    |
+------------------------+------------------------------------+
| logical_or             |                                    |
+------------------------+------------------------------------+
| nn.batch_norm          |                                    |
+------------------------+------------------------------------+
| clip                   |                                    |
+------------------------+------------------------------------+
| nn.softmax             |                                    |
+------------------------+------------------------------------+
| nn.conv2d              |                                    |
+------------------------+------------------------------------+
| add                    |                                    |
+------------------------+------------------------------------+
| maximum                |                                    |
+------------------------+------------------------------------+
| minimum                |                                    |
+------------------------+------------------------------------+
| nn.max_pool2d          |                                    |
+------------------------+------------------------------------+
| nn.avg_pool2d          |                                    |
+------------------------+------------------------------------+
| squeeze                |                                    |
+------------------------+------------------------------------+
| nn.conv2d_transpose    |                                    |
+------------------------+------------------------------------+
| transpose              |                                    |
+------------------------+------------------------------------+
| reshape                |                                    |
+------------------------+------------------------------------+
| nn.pad                 |                                    |
+------------------------+------------------------------------+
| mean                   |                                    |
+------------------------+------------------------------------+
| nn.adaptive_avg_pool2d |                                    |
+------------------------+------------------------------------+
| qnn.add                |                                    |
+------------------------+------------------------------------+
| qnn.subtract           |                                    |
+------------------------+------------------------------------+
| qnn.mul                |                                    |
+------------------------+------------------------------------+
| qnn.quantize           |                                    |
+------------------------+------------------------------------+
| qnn.dequantize         |                                    |
+------------------------+------------------------------------+
| qnn.requantize         |                                    |
+------------------------+------------------------------------+
| qnn.concatenate        |                                    |
+------------------------+------------------------------------+
| image.resize2d         |                                    |
+------------------------+------------------------------------+
