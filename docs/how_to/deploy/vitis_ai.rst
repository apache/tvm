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


Vitis-AI Integration
====================

`Vitis-AI <https://github.com/Xilinx/Vitis-AI>`__ is Xilinx's development stack
for hardware-accelerated AI inference on Xilinx platforms, including both edge
devices and Alveo cards. It consists of optimized IP, tools, libraries, models,
and example designs. It is designed with high efficiency and ease of use in
mind, unleashing the full potential of AI acceleration on Xilinx FPGA and ACAP.

The current Vitis-AI Byoc flow inside TVM enables acceleration of Neural
Network model inference on edge and cloud. The identifiers for the supported
edge and cloud Deep Learning Processor Units (DPU's) are DPUCZDX8G respectively
DPUCADX8G. DPUCZDX8G and DPUCADX8G are hardware accelerators for convolutional
neural networks (CNN's) on top of the Xilinx `Zynq Ultrascale+ MPSoc
<https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html>`__
respectively `Alveo
<https://www.xilinx.com/products/boards-and-kits/alveo.html>`__ (U200/U250)
platforms. For more information about the DPU identifiers see the section on
`DPU naming information <#dpu-naming-information>`__.

On this page you will find information on how to `build
<#build-instructions>`__ TVM with Vitis-AI and on how to `get started
<#getting-started>`__ with an example.

DPU naming information
----------------------

+---------------------------------+-----------------+-------------------------------------------------------------------------+------------------------------------------------------------+---------------------------------------------------+--------------------------------------------------------------------------+
| DPU                             | Application     | HW Platform                                                             | Quantization Method                                        | Quantization Bitwidth                             | Design Target                                                            |
+=================================+=================+=========================================================================+============================================================+===================================================+==========================================================================+
| Deep Learning Processing Unit   | C: CNN R: RNN   | AD: Alveo DDR AH: Alveo HBM VD: Versal DDR with AIE & PL ZD: Zynq DDR   | X: DECENT I: Integer threshold F: Float threshold R: RNN   | 4: 4-bit 8: 8-bit 16: 16-bit M: Mixed Precision   | G: General purpose H: High throughput L: Low latency C: Cost optimized   |
+---------------------------------+-----------------+-------------------------------------------------------------------------+------------------------------------------------------------+---------------------------------------------------+--------------------------------------------------------------------------+

Build instructions
------------------

This section lists the instructions for building TVM with Vitis-AI for both
`cloud <#cloud-dpucadx8g>`__ and `edge <#edge-dpuczdx8g>`__.

Cloud (DPUCADX8G)
~~~~~~~~~~~~~~~~~

For Vitis-AI acceleration in the cloud TVM has to be built on top of the Xilinx
Alveo platform.

System requirements
^^^^^^^^^^^^^^^^^^^

The following table lists system requirements for running docker containers as
well as Alveo cards.

+-----------------------------------------------------+----------------------------------------------------------+
| **Component**                                       | **Requirement**                                          |
+=====================================================+==========================================================+
| Motherboard                                         | PCI Express 3.0-compliant with one dual-width x16 slot   |
+-----------------------------------------------------+----------------------------------------------------------+
| System Power Supply                                 | 225W                                                     |
+-----------------------------------------------------+----------------------------------------------------------+
| Operating System                                    | Ubuntu 16.04, 18.04                                      |
+-----------------------------------------------------+----------------------------------------------------------+
|                                                     | CentOS 7.4, 7.5                                          |
+-----------------------------------------------------+----------------------------------------------------------+
|                                                     | RHEL 7.4, 7.5                                            |
+-----------------------------------------------------+----------------------------------------------------------+
| CPU                                                 | Intel i3/i5/i7/i9/Xeon 64-bit CPU                        |
+-----------------------------------------------------+----------------------------------------------------------+
| GPU (Optional to accelerate quantization)           | NVIDIA GPU with a compute capability > 3.0               |
+-----------------------------------------------------+----------------------------------------------------------+
| CUDA Driver (Optional to accelerate quantization)   | nvidia-410                                               |
+-----------------------------------------------------+----------------------------------------------------------+
| FPGA                                                | Xilinx Alveo U200 or U250                                |
+-----------------------------------------------------+----------------------------------------------------------+
| Docker Version                                      | 19.03.1                                                  |
+-----------------------------------------------------+----------------------------------------------------------+

Hardware setup and docker build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the Vitis AI repository:

   .. code:: bash

      git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI

2. Install Docker, and add the user to the docker group. Link the user to
   docker installation instructions from the following docker's website:

   -  https://docs.docker.com/install/linux/docker-ce/ubuntu/
   -  https://docs.docker.com/install/linux/docker-ce/centos/
   -  https://docs.docker.com/install/linux/linux-postinstall/

3. Download the latest Vitis AI Docker with the following command. This
   container runs on CPU.

   .. code:: bash

      docker pull xilinx/vitis-ai:latest

   To accelerate the quantization, you can optionally use the Vitis-AI GPU
   docker image. Use the below commands to build the Vitis-AI GPU docker
   container:

   .. code:: bash

      cd Vitis-AI/docker
      ./docker_build_gpu.sh

4. Set up Vitis AI to target Alveo cards. To target Alveo cards with Vitis AI
   for machine learning workloads, you must install the following software
   components:

   -  Xilinx Runtime (XRT)
   -  Alveo Deployment Shells (DSAs)
   -  Xilinx Resource Manager (XRM) (xbutler)
   -  Xilinx Overlaybins (Accelerators to Dynamically Load - binary programming
     files)

   While it is possible to install all of these software components
   individually, a script has been provided to automatically install them at
   once. To do so:

   -  Run the following commands:

      .. code:: bash

         cd Vitis-AI/alveo/packages
         sudo su
         ./install.sh

   -  Power cycle the system.

5. Clone tvm repo and pyxir repo

   .. code:: bash

      git clone --recursive https://github.com/apache/tvm.git
      git clone --recursive https://github.com/Xilinx/pyxir.git

6. Build and start the tvm runtime Vitis-AI Docker Container.

   .. code:: bash

      ./tvm/docker/build.sh demo_vitis_ai bash
      ./tvm/docker/bash.sh tvm.demo_vitis_ai

      #Setup inside container
      source /opt/xilinx/xrt/setup.sh
      . $VAI_ROOT/conda/etc/profile.d/conda.sh
      conda activate vitis-ai-tensorflow

7. Install PyXIR

   .. code:: bash

     cd pyxir
     python3 setup.py install --use_vai_rt_dpucadx8g --user


8. Build TVM inside the container with Vitis-AI

   .. code:: bash

      cd tvm
      mkdir build
      cp cmake/config.cmake build
      cd build
      echo set\(USE_LLVM ON\) >> config.cmake
      echo set\(USE_VITIS_AI ON\) >> config.cmake
      cmake ..
      make -j$(nproc)

9.  Install TVM

    .. code:: bash

      cd tvm/python
      pip3 install -e . --user

Edge (DPUCZDX8G)
~~~~~~~~~~~~~~~~

For edge deployment we make use of two systems referred to as host and edge.
The `host <#host-requirements>`__ system is responsible for quantization and
compilation of the neural network model in a first offline step. Afterwards,
the model will de deployed on the `edge <#edge-requirements>`__ system.

Host requirements
^^^^^^^^^^^^^^^^^

The following table lists system requirements for running the TVM - Vitis-AI
docker container.

+-----------------------------------------------------+----------------------------------------------+
| **Component**                                       | **Requirement**                              |
+=====================================================+==============================================+
| Operating System                                    | Ubuntu 16.04, 18.04                          |
+-----------------------------------------------------+----------------------------------------------+
|                                                     | CentOS 7.4, 7.5                              |
+-----------------------------------------------------+----------------------------------------------+
|                                                     | RHEL 7.4, 7.5                                |
+-----------------------------------------------------+----------------------------------------------+
| CPU                                                 | Intel i3/i5/i7/i9/Xeon 64-bit CPU            |
+-----------------------------------------------------+----------------------------------------------+
| GPU (Optional to accelerate quantization)           | NVIDIA GPU with a compute capability > 3.0   |
+-----------------------------------------------------+----------------------------------------------+
| CUDA Driver (Optional to accelerate quantization)   | nvidia-410                                   |
+-----------------------------------------------------+----------------------------------------------+
| FPGA                                                | Not necessary on host                        |
+-----------------------------------------------------+----------------------------------------------+
| Docker Version                                      | 19.03.1                                      |
+-----------------------------------------------------+----------------------------------------------+

Host setup and docker build
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone tvm repo

   .. code:: bash

      git clone --recursive https://github.com/apache/tvm.git

2. Build and start the tvm runtime Vitis-AI Docker Container.

   .. code:: bash

      cd tvm
      ./tvm/docker/build.sh demo_vitis_ai bash
      ./tvm/docker/bash.sh tvm.demo_vitis_ai

      #Setup inside container
      . $VAI_ROOT/conda/etc/profile.d/conda.sh
      conda activate vitis-ai-tensorflow

3. Install PyXIR

   .. code:: bash

      git clone --recursive https://github.com/Xilinx/pyxir.git
      cd pyxir
      python3 setup.py install --user


4. Build TVM inside the container with Vitis-AI.

   .. code:: bash

      cd tvm
      mkdir build
      cp cmake/config.cmake build
      cd build
      echo set\(USE_LLVM ON\) >> config.cmake
      echo set\(USE_VITIS_AI ON\) >> config.cmake
      cmake ..
      make -j$(nproc)

5. Install TVM

   .. code:: bash

      cd tvm/python
      pip3 install -e . --user

Edge requirements
^^^^^^^^^^^^^^^^^

The DPUCZDX8G can be deployed on the `Zynq Ultrascale+ MPSoc
<https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html>`__
platform. The following development boards can be used out-of-the-box:

+--------------------+----------------------+-----------------------------------------------------------------------+
| **Target board**   | **TVM identifier**   | **Info**                                                              |
+====================+======================+=======================================================================+
| Ultra96            | DPUCZDX8G-ultra96    | https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html         |
+--------------------+----------------------+-----------------------------------------------------------------------+
| ZCU104             | DPUCZDX8G-zcu104     | https://www.xilinx.com/products/boards-and-kits/zcu104.html           |
+--------------------+----------------------+-----------------------------------------------------------------------+
| ZCU102             | DPUCZDX8G-zcu102     | https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html   |
+--------------------+----------------------+-----------------------------------------------------------------------+

Edge hardware setup
^^^^^^^^^^^^^^^^^^^

.. note::

  This section provides instructions for setting up with the `Pynq
  <http://www.pynq.io/>`__ platform but Petalinux based flows are also
  supported.

1. Download the Pynq v2.6 image for your target (use Z1 or Z2 for Ultra96
   target depending on board version) Link to image:
   https://github.com/Xilinx/PYNQ/releases/tag/v2.6.0
2. Follow Pynq instructions for setting up the board: `pynq setup
   <https://pynq.readthedocs.io/en/latest/getting_started.html>`__
3. After connecting to the board, make sure to run as root. **Execute** ``su``
4. Set up DPU on Pynq:

    .. code:: bash

     git clone --branch v1.2.0 --recursive --shallow-submodules https://github.com/Xilinx/DPU-PYNQ.git
     cd DPU-PYNQ/upgrade
     make
     pip3 install pynq-dpu==1.2.0

5. Run the following command to download the DPU bitstream:

   .. code:: bash

     python3 -c 'from pynq_dpu import DpuOverlay ; overlay = DpuOverlay("dpu.bit")'

6. Check whether the DPU kernel is alive:

   .. code:: bash

     dexplorer -w

Edge TVM setup
^^^^^^^^^^^^^^

.. note::

  When working on Petalinux instead of Pynq, the following steps might take
  more manual work (e.g building hdf5 from source). Also, TVM has a scipy
  dependency which you then might have to build from source or circumvent. We
  don't depend on scipy in our flow.

Building TVM depends on the Xilinx `PyXIR <https://github.com/Xilinx/pyxir>`__
package. PyXIR acts as an interface between TVM and Vitis-AI tools.

1. First install the PyXIR h5py and pydot dependencies:

   .. code:: bash

      apt-get install libhdf5-dev
      pip3 install pydot==1.4.1 h5py==2.8.0

2. Install PyXIR

   .. code:: bash

      git clone --recursive https://github.com/Xilinx/pyxir.git
      cd pyxir
      sudo python3 setup.py install --use_vai_rt_dpuczdx8g

3. Build TVM with Vitis-AI

   .. code:: bash

      git clone --recursive https://github.com/apache/tvm
      cd tvm
      mkdir build
      cp cmake/config.cmake build
      cd build
      echo set\(USE_LLVM OFF\) >> config.cmake
      echo set\(USE_VITIS_AI ON\) >> config.cmake
      cmake ..
      make tvm_runtime -j$(nproc)

4. Install TVM

   .. code:: bash

      cd tvm/python
      pip3 install -e .

5. Check whether the setup was successful in the Python shell:

   .. code:: bash

      python3 -c 'import pyxir; import tvm'


Getting started
---------------

This section shows how to use TVM with Vitis-AI. For this it's important to
understand that neural network models are quantized for Vitis-AI execution in
fixed point arithmetic. The approach we take here is to quantize on-the-fly
using the first N inputs as explained in the next section.

On-the-fly quantization
~~~~~~~~~~~~~~~~~~~~~~~

Usually, to be able to accelerate inference of Neural Network models with
Vitis-AI DPU accelerators, those models need to quantized upfront.  In TVM -
Vitis-AI flow, we make use of on-the-fly quantization to remove this additional
preprocessing step. In this flow, one doesn't need to quantize his/her model
upfront but can make use of the typical inference execution calls (module.run)
to quantize the model on-the-fly using the first N inputs that are provided
(see more information below). This will set up and calibrate the Vitis-AI DPU
and from that point onwards inference will be accelerated for all next inputs.
Note that the edge flow deviates slightly from the explained flow in that
inference won't be accelerated after the first N inputs but the model will have
been quantized and compiled and can be moved to the edge device for deployment.
Please check out the `edge <#Edge%20usage>`__ usage instructions below for more
information.

Config/Settings
~~~~~~~~~~~~~~~

A couple of environment variables can be used to customize the Vitis-AI Byoc
flow.

+----------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Environment Variable**   | **Default if unset**                   | **Explanation**                                                                                                                                                                                                                                                                                                                            |
+============================+========================================+============================================================================================================================================================================================================================================================================================================================================+
| PX\_QUANT\_SIZE            | 128                                    | The number of inputs that will be used for quantization (necessary for Vitis-AI acceleration)                                                                                                                                                                                                                                              |
+----------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PX\_BUILD\_DIR             | Use the on-the-fly quantization flow   | Loads the quantization and compilation information from the provided build directory and immediately starts Vitis-AI hardware acceleration. This configuration can be used if the model has been executed before using on-the-fly quantization during which the quantization and comilation information was cached in a build directory.   |
+----------------------------+----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Cloud usage
~~~~~~~~~~~

This section shows how to accelerate a convolutional neural network model in
TVM with Vitis-AI on the cloud.

To be able to target the Vitis-AI cloud DPUCADX8G we first have to import the
DPU target in PyXIR. This PyXIR package is the interface being used by TVM to
integrate with the Vitis-AI stack. Additionaly, import the typical TVM and
Relay modules and the Vitis-AI contrib module inside TVM.

.. code:: python

   import pyxir
   import pyxir.contrib.target.DPUCADX8G

   import tvm
   import tvm.relay as relay
   from tvm.contrib.target import vitis_ai
   from tvm.contrib import utils, graph_executor
   from tvm.relay.build_module import bind_params_by_name
   from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai

After importing a convolutional neural network model using the usual Relay
API's, annotate the Relay expression for the given Vitis-AI DPU target and
partition the graph.

.. code:: python
   
   dpu = 'DPUCADX8G'
   mod = partition_for_vitis_ai(mod, params, dpu)

Now, we can build the TVM runtime library for executing the model. The TVM
target is 'llvm' as the operations that can't be handled by the DPU are
executed on the CPU. The Vitis-AI DPU is DPUCADX8G as we are targeting the
cloud DPU and this DPU identifier is passed as a config to the TVM build call.

.. code:: python

   target = 'llvm'

   with tvm.transform.PassContext(opt_level=3, config= {'relay.ext.vitis_ai.options': {'dpu': dpu}}):
      lib = relay.build(mod, target, params=params)

As one more step before we can accelerate a model with Vitis-AI in TVM we have
to quantize and compile the model for execution on the DPU. We make use of
on-the-fly quantization for this. Using this method one doesn’t need to
quantize their model upfront and can make use of the typical inference
execution calls (module.run) to calibrate the model on-the-fly using the first
N inputs that are provided. After the first N iterations, computations will be
accelerated on the DPU. So now we will feed N inputs to the TVM runtime module.
Note that these first N inputs will take a substantial amount of time.

.. code:: python

   module = graph_executor.GraphModule(lib["default"](tvm.cpu()))

   # First N (default = 128) inputs are used for quantization calibration and will
   # be executed on the CPU
   # This config can be changed by setting the 'PX_QUANT_SIZE' (e.g. export PX_QUANT_SIZE=64)
   for i in range(128):
      module.set_input(input_name, inputs[i])
      module.run()

Afterwards, inference will be accelerated on the DPU.

.. code:: python

   module.set_input(name, data)
   module.run()

To save and load the built module, one can use the typical TVM API's:

.. code:: python

   lib_path = "deploy_lib.so"
   lib.export_library(lib_path)

Load the module from compiled files and run inference

.. code:: python

   # load the module into memory
   loaded_lib = tvm.runtime.load_module(lib_path)

   module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
   module.set_input(name, data)
   module.run()

Edge usage
~~~~~~~~~~

This section shows how to accelerate a convolutional neural network model in
TVM with Vitis-AI at the edge. The first couple of steps will have to be run on
the host machine and take care of quantization and compilation for deployment
at the edge.

A complete ResNet 18 example can be found `here
<https://github.com/Xilinx/pyxir/tree/master/examples/tvm>`__.

Host steps
^^^^^^^^^^

To be able to target the Vitis-AI cloud DPUCZDX8G we first have to import the
DPU target in PyXIR. This PyXIR package is the interface being used by TVM to
integrate with the Vitis-AI stack. Additionaly, import the typical TVM and
Relay modules and the Vitis-AI contrib module inside TVM.

.. code:: python

   import pyxir
   import pyxir.contrib.target.DPUCZDX8G

   import tvm
   import tvm.relay as relay
   from tvm.contrib.target import vitis_ai
   from tvm.contrib import utils, graph_executor
   from tvm.relay.build_module import bind_params_by_name
   from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai

After importing a convolutional neural network model using the usual Relay
API's, annotate the Relay expression for the given Vitis-AI DPU and partition
the graph.

.. note::

    We recommend converting DPU convolutions' data layouts to NHWC and CPU convolutions'
    data layouts to NCHW for best DPU and out of the box CPU performance. You can use the
    ConvertLayout transformation pass two times to achieve this as demonstrated in the code
    block underneath. You can also leave the CPU convolution layouts in NHWC and tune ARM CPU
    performance for this data layout to avoid the layout transformation overheads introduced by
    executing DPU convolutions in NHWC and CPU convolutions in NCHW
    (check out the `AutoScheduling <https://tvm.apache.org/docs/tutorials/index.html#autoscheduler-template-free-auto-scheduling>`__
    and `AutoTuning <https://tvm.apache.org/docs/tutorials/autotvm/tune_relay_arm.html>`__
    tutorials for this).

.. code:: python

   mod["main"] = bind_params_by_name(mod["main"], params)
   
   # For edge DPU we recommend converting the convolutions' data layout
   #    to NHWC for best performance. Therefore, we first convert the layouts
   #    of all convolutions to NHWC before partitioning. Afterwards, we can
   #    convert any remaining convolutions (to be executed on CPU) back to NCHW.
   desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
   seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                   relay.transform.ConvertLayout(desired_layouts),
                                   relay.transform.FoldConstant()])
   with tvm.transform.PassContext(opt_level=3):
       mod = seq(mod)
   
   dpu = 'DPUCZDX8G-zcu104'
   # Annotate and partition the Relay expression for the given DPU
   mod = partition_for_vitis_ai(mod, params, dpu)
   
   # After partitioning we recommend transforming the remaining convolutions
   #    (that will be executed on CPU, if any) back to NCHW data layout
   #    for best CPU performance
   desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
   seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                   relay.transform.ConvertLayout(desired_layouts),
                                   relay.transform.FoldConstant()])
   with tvm.transform.PassContext(opt_level=3):
       mod = seq(mod)

Now, we can build the TVM runtime library for executing the model. The TVM
target is 'llvm' as the operations that can't be handled by the DPU are
executed on the CPU. At this point that means the CPU on the host machine.  The
Vitis-AI DPU identifier is DPUCZDX8G-zcu104 as we are targeting the edge DPU on
the ZCU104 board and this identifier is passed as a config to the TVM build
call. Note that different identifiers can be passed for different DPU's, see
`edge DPU's info <#edge-requirements>`__. Additionally, we provide the
'export_runtime_module' config that points to a file to which we can export the
Vitis-AI runtime module. We have to do this because we will first be compiling
and quantizing the model on the host machine before building the model for edge
deployment. As you will see later on, the exported runtime module will be
passed to the edge build so that the Vitis-AI runtime module can be included.

.. code:: python

   target = 'llvm'
   export_rt_mod_file = "vitis_ai.rtmod"
   
   build_options = {
      'dpu': dpu,
      'export_runtime_module': export_rt_mod_file
   }
   with tvm.transform.PassContext(opt_level=3, config= {'relay.ext.vitis_ai.options': build_options}):
      lib = relay.build(mod, target, params=params)

We will quantize and compile the model for execution on the DPU using
on-the-fly quantization on the host machine. This makes use of TVM inference
calls (module.run) to quantize the model on the host with the first N inputs.

.. code:: python

   module = graph_executor.GraphModule(lib["default"](tvm.cpu()))

   # First N (default = 128) inputs are used for quantization calibration and will
   # be executed on the CPU
   # This config can be changed by setting the 'PX_QUANT_SIZE' (e.g. export PX_QUANT_SIZE=64)
   for i in range(128):
      module.set_input(input_name, inputs[i])
      module.run()

Save the TVM lib module so that the Vitis-AI runtime module will also be
exported (to the 'export_runtime_module' path we previously passed as a
config).

.. code:: python

   from tvm.contrib import utils

   temp = utils.tempdir()
   lib.export_library(temp.relpath("tvm_lib.so"))

After quantizing and compiling the model for Vitis-AI acceleration using the
first N inputs we can build the model for execution on the ARM edge device.
Here we pass the previously exported Vitis-AI runtime module so it can be
included in the TVM build.

.. code:: python

   # Export lib for aarch64 target
   target = tvm.target.arm_cpu('ultra96')
   lib_kwargs = {
        'fcompile': contrib.cc.create_shared,
        'cc': "/usr/aarch64-linux-gnu/bin/ld"
   }
   
   build_options = {
        'load_runtime_module': export_rt_mod_file
   }
   with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
        lib_arm = relay.build(mod, target, params=params)

   lib_dpuv2.export_library('tvm_dpu_arm.so', **lib_kwargs)

Now, move the TVM build files (tvm\_dpu\_arm.json, tvm\_dpu\_arm.so,
tvm\_dpu\_arm.params) to the edge device. For information on setting up the
edge device check out the `edge setup <#edge-dpuczdx8g>`__ section.

Edge steps
^^^^^^^^^^

After setting up TVM with Vitis-AI on the edge device, you can now load the TVM
runtime module into memory and feed inputs for inference. A nearly complete
runtiem script can be found underneath. Make sure to run the script as root
(execute ``su`` in terminal to log into root).

.. note::

    You will see a warning about the 'cpu-tf' runtime not being found. This
    warning is expected on the board and can be ignored. Note also that you
    **shouldn't** import the PyXIR DPU targets in the run script (``import
    pyxir.contrib.target.DPUCZDX8G``).

.. code:: python

   import pyxir
   import tvm
   from tvm.contrib import graph_executor

   dev = tvm.cpu()
   
   # input_name = ...
   # input_data = ...

   # load the module into memory
   lib = tvm.runtime.load_module("tvm_dpu_arm.so")

   module = graph_executor.GraphModule(lib["default"](dev))
   module.set_input(input_name, input_data)
   module.run()
