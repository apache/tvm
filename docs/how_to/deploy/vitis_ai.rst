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


Vitis AI Integration
====================

`Vitis AI <https://github.com/Xilinx/Vitis-AI>`__ is Xilinx's
development stack for hardware-accelerated AI inference on Xilinx
platforms, including both edge devices and Alveo cards. It consists of
optimized IP, tools, libraries, models, and example designs. It is
designed with high efficiency and ease of use in mind, unleashing the
full potential of AI acceleration on Xilinx FPGA and ACAP.

The current Vitis AI flow inside TVM enables acceleration of Neural
Network model inference on edge and cloud with the `Zynq Ultrascale+
MPSoc <https://www.xilinx.com/products/silicon-devices/soc/zynq-ultrascale-mpsoc.html>`__,
`Alveo <https://www.xilinx.com/products/boards-and-kits/alveo.html>`__
and `Versal <https://www.xilinx.com/products/silicon-devices/acap/versal.html>`__ platforms.
The identifiers for the supported edge and cloud Deep Learning Processor Units (DPU's) are:

+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| **Target Board**                                                                        | **DPU ID**            | **TVM Target ID**          |
+=========================================================================================+=======================+============================+
| `ZCU104 <https://www.xilinx.com/products/boards-and-kits/zcu104.html>`__                | DPUCZDX8G             | DPUCZDX8G-zcu104           |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `ZCU102 <https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html>`__        | DPUCZDX8G             | DPUCZDX8G-zcu102           |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `Kria KV260 <https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html>`__ | DPUCZDX8G             | DPUCZDX8G-kv260            |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `VCK190 <https://www.xilinx.com/products/boards-and-kits/vck190.html>`__                | DPUCVDX8G             | DPUCVDX8G                  |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `VCK5000 <https://www.xilinx.com/products/boards-and-kits/vck5000.html>`__              | DPUCVDX8H             | DPUCVDX8H                  |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `U200 <https://www.xilinx.com/products/boards-and-kits/alveo/u200.html>`__              | DPUCADF8H             | DPUCADF8H                  |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `U250 <https://www.xilinx.com/products/boards-and-kits/alveo/u250.html>`__              | DPUCADF8H             | DPUCADF8H                  |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `U50 <https://www.xilinx.com/products/boards-and-kits/alveo/u50.html>`__                | DPUCAHX8H / DPUCAHX8L | DPUCAHX8H-u50 / DPUCAHX8L  |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+
| `U280 <https://www.xilinx.com/products/boards-and-kits/alveo/u280.html>`__              | DPUCAHX8H / DPUCAHX8L | DPUCAHX8H-u280 / DPUCAHX8L |
+-----------------------------------------------------------------------------------------+-----------------------+----------------------------+

For more information about the DPU identifiers see following table:

+-------------------+-------------+--------------------------------+------------------------+------------------------+------------------------+
| DPU               | Application | HW Platform                    | Quantization Method    | Quantization Bitwidth  | Design Target          |
+===================+=============+================================+========================+========================+========================+
| | Deep Learning   | | C: CNN    | | AD: Alveo DDR                | | X: DECENT            | | 4: 4-bit             | | G: General purpose   |
| | Processing Unit | | R: RNN    | | AH: Alveo HBM                | | I: Integer threshold | | 8: 8-bit             | | H: High throughput   |
|                   |             | | VD: Versal DDR with AIE & PL | | F: Float threshold   | | 16: 16-bit           | | L: Low latency       |
|                   |             | | ZD: Zynq DDR                 | | R: RNN               | | M: Mixed Precision   | | C: Cost optimized    |
+-------------------+-------------+--------------------------------+------------------------+------------------------+------------------------+

On this page you will find information on how to `setup <#setup-instructions>`__ TVM with Vitis AI
on different platforms (Zynq, Alveo, Versal) and on how to get started with `Compiling a Model <#compiling-a-model>`__
and executing on different platforms: `Inference <#inference>`__.

System Requirements
-------------------

The `Vitis AI System Requirements page <https://github.com/Xilinx/Vitis-AI/blob/master/docs/learn/system_requirements.md>`__
lists the system requirements for running docker containers as well as doing executing on Alveo cards.
For edge devices (e.g. Zynq), deploying models requires a host machine for compiling models using the TVM with Vitis AI flow,
and an edge device for running the compiled models. The host system requirements are the same as specified in the link above.

Setup instructions
------------------

This section provide the instructions for setting up the TVM with Vitis AI flow for both cloud and edge.
TVM with Vitis AI support is provided through a docker container. The provided scripts and Dockerfile
compiles TVM and Vitis AI into a single image.

1. Clone TVM repo

   .. code:: bash

      git clone --recursive https://github.com/apache/tvm.git
      cd tvm

2. Build and start the TVM - Vitis AI docker container.

   .. code:: bash

      ./docker/build.sh demo_vitis_ai bash
      ./docker/bash.sh tvm.demo_vitis_ai

      # Setup inside container
      conda activate vitis-ai-tensorflow

3. Build TVM inside the container with Vitis AI (inside tvm directory)

   .. code:: bash

      mkdir build
      cp cmake/config.cmake build
      cd build
      echo set\(USE_LLVM ON\) >> config.cmake
      echo set\(USE_VITIS_AI ON\) >> config.cmake
      cmake ..
      make -j$(nproc)

4.  Install TVM

    .. code:: bash

      cd ../python
      pip3 install -e . --user

Inside this docker container you can now compile models for both cloud and edge targets.
To run on cloud Alveo or Versal VCK5000 cards inside the docker container, please follow the
`Alveo <#alveo-setup>`__ respectively  `Versal VCK5000 <#versal-vck5000-setup>`__ setup instructions.
To setup your Zynq or Versal VCK190 evaluation board for inference, please follow
the `Zynq <#zynq-setup>`__ respectively `Versal VCK190 <#versal-vck190-setup>`__ instructions.

Alveo Setup
~~~~~~~~~~~

Check out following page for setup information: `Alveo Setup <https://github.com/Xilinx/Vitis-AI/blob/v1.4/setup/alveo/README.md>`__.

After setup, you can select the right DPU inside the docker container in the following way:

.. code:: bash

      cd /workspace
      git clone --branch v1.4 --single-branch --recursive https://github.com/Xilinx/Vitis-AI.git
      cd Vitis-AI/setup/alveo
      source setup.sh [DPU-IDENTIFIER]

The DPU identifier for this can be found in the second column of the DPU Targets table at the top of this page.

Versal VCK5000 Setup
~~~~~~~~~~~~~~~~~~~~

Check out following page for setup information: `VCK5000 Setup <https://github.com/Xilinx/Vitis-AI/blob/v1.4/setup/vck5000/README.md>`__.

After setup, you can select the right DPU inside the docker container in the following way:

.. code:: bash

      cd /workspace
      git clone --branch v1.4 --single-branch --recursive https://github.com/Xilinx/Vitis-AI.git
      cd Vitis-AI/setup/vck5000
      source setup.sh

Zynq Setup
~~~~~~~~~~

For the Zynq target (DPUCZDX8G) the compilation stage will run inside the docker on a host machine.
This doesn't require any specific setup except for building the TVM - Vitis AI docker. For executing the model,
the Zynq board will first have to be set up and more information on that can be found here.

1. Download the Petalinux image for your target:
    - `ZCU104 <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu104-dpu-v2021.1-v1.4.0.img.gz>`__
    - `ZCU102 <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2021.1-v1.4.0.img.gz>`__
    - `Kria KV260 <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2020.2-v1.4.0.img.gz>`__
2. Use Etcher software to burn the image file onto the SD card.
3. Insert the SD card with the image into the destination board.
4. Plug in the power and boot the board using the serial port to operate on the system.
5. Set up the IP information of the board using the serial port. For more details on step 1 to 5, please refer to `Setting Up The Evaluation Board <https://www.xilinx.com/html_docs/vitis_ai/1_4/installation.html#ariaid-title8>`__.
6. Create 4GB of swap space on the board

.. code:: bash

      fallocate -l 4G /swapfile
      chmod 600 /swapfile
      mkswap /swapfile
      swapon /swapfile
      echo "/swapfile swap swap defaults 0 0" > /etc/fstab

7. Install hdf5 dependency (will take between 30 min and 1 hour to finish)

.. code:: bash

      cd /tmp && \
        wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz && \
        tar -zxvf hdf5-1.10.7.tar.gz && \
        cd hdf5-1.10.7 && \
        ./configure --prefix=/usr && \
        make -j$(nproc) && \
        make install && \
        cd /tmp && rm -rf hdf5-1.10.7*

8. Install Python dependencies

.. code:: bash

      pip3 install Cython==0.29.23 h5py==2.10.0 pillow

9. Install PyXIR

.. code:: bash

      git clone --recursive --branch rel-v0.3.1 --single-branch https://github.com/Xilinx/pyxir.git
      cd pyxir
      sudo python3 setup.py install --use_vart_edge_dpu

10. Build and install TVM with Vitis AI

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
      cd ../python
      pip3 install --no-deps  -e .

11. Check whether the setup was successful in the Python shell:

.. code:: bash

      python3 -c 'import pyxir; import tvm'

.. note::

    You might see a warning about the 'cpu-tf' runtime not being found. This warning is
    expected on the board and can be ignored.


Versal VCK190 Setup
~~~~~~~~~~~~~~~~~~~

For the Versal VCK190 setup, please follow the instructions for `Zynq Setup <#zynq-setup>`__,
but now use the `VCK190 image <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2020.2-v1.4.0.img.gz>`__
in step 1. The other steps are the same.


Compiling a Model
-----------------

The TVM with Vitis AI flow contains two stages: Compilation and Inference.
During the compilation a user can choose a model to compile for the cloud or
edge target devices that are currently supported. Once a model is compiled,
the generated files can be used to run the model on a the specified target
device during the `Inference <#inference>`__ stage. Currently, the TVM with
Vitis AI flow supported a selected number of Xilinx data center and edge devices.

In this section we walk through the typical flow for compiling models with Vitis AI
inside TVM.

**Imports**

Make sure to import PyXIR and the DPU target (``import pyxir.contrib.target.DPUCADF8H`` for DPUCADF8H):

.. code:: python

   import pyxir
   import pyxir.contrib.target.DPUCADF8H

   import tvm
   import tvm.relay as relay
   from tvm.contrib.target import vitis_ai
   from tvm.contrib import utils, graph_executor
   from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai

**Declare the Target**

.. code:: python

   tvm_target = 'llvm'
   dpu_target = 'DPUCADF8H' # options: 'DPUCADF8H', 'DPUCAHX8H-u50', 'DPUCAHX8H-u280', 'DPUCAHX8L', 'DPUCVDX8H', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102', 'DPUCZDX8G-kv260'

The TVM with Vitis AI flow currently supports the DPU targets listed in
the table at the top of this page. Once the appropriate targets are defined,
we invoke the TVM compiler to build the graph for the specified target.

**Import the Model**

Example code to import an MXNet model:

.. code:: python

   mod, params = relay.frontend.from_mxnet(block, input_shape)


**Partition the Model**

After importing the model, we utilize the Relay API to annotate the Relay expression for the provided DPU target and partition the graph.

.. code:: python

    mod = partition_for_vitis_ai(mod, params, dpu=dpu_target)


**Build the Model**

The partitioned model is passed to the TVM compiler to generate the runtime libraries for the TVM Runtime.

.. code:: python

    export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
    build_options = {
        'dpu': dpu_target,
        'export_runtime_module': export_rt_mod_file
    }
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
        lib = relay.build(mod, tvm_target, params=params)

**Quantize the Model**

Usually, to be able to accelerate inference of Neural Network models
with Vitis AI DPU accelerators, those models need to quantized upfront.
In TVM - Vitis AI flow, we make use of on-the-fly quantization to remove
this additional preprocessing step. In this flow, one doesn't need to
quantize his/her model upfront but can make use of the typical inference
execution calls (module.run) to quantize the model on-the-fly using the
first N inputs that are provided (see more information below). This will
set up and calibrate the Vitis-AI DPU and from that point onwards
inference will be accelerated for all next inputs. Note that the edge
flow deviates slightly from the explained flow in that inference won't
be accelerated after the first N inputs but the model will have been
quantized and compiled and can be moved to the edge device for
deployment. Please check out the `Running on Zynq <#running-on-zynq>`__
section below for more information.

.. code:: python

   module = graph_executor.GraphModule(lib["default"](tvm.cpu()))

   # First N (default = 128) inputs are used for quantization calibration and will
   # be executed on the CPU
   # This config can be changed by setting the 'PX_QUANT_SIZE' (e.g. export PX_QUANT_SIZE=64)
   for i in range(128):
      module.set_input(input_name, inputs[i])
      module.run()

By default, the number of images used for quantization is set to 128.
You could change the number of images used for On-The-Fly Quantization
with the PX_QUANT_SIZE environment variable. For example, execute the
following line in the terminal before calling the compilation script
to reduce the quantization calibration dataset to eight images.
This can be used for quick testing.

.. code:: bash

    export PX_QUANT_SIZE=8

Lastly, we store the compiled output from the TVM compiler on disk for
running the model on the target device. This happens as follows for
cloud DPU's (Alveo, VCK5000):

.. code:: python

   lib_path = "deploy_lib.so"
   lib.export_library(lib_path)


For edge targets (Zynq, VCK190) we have to rebuild for aarch64. To do this
we first have to normally export the module to also serialize the Vitis AI
runtime module (vitis_ai.rtmod). We will load this runtime module again
afterwards to rebuild and export for aarch64.

.. code:: python

    temp = utils.tempdir()
    lib.export_library(temp.relpath("tvm_lib.so"))

    # Build and export lib for aarch64 target
    tvm_target = tvm.target.arm_cpu('ultra96')
    lib_kwargs = {
       'fcompile': contrib.cc.create_shared,
       'cc': "/usr/aarch64-linux-gnu/bin/ld"
    }

    build_options = {
        'load_runtime_module': export_rt_mod_file
    }
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
         lib_edge = relay.build(mod, tvm_target, params=params)

    lib_edge.export_library('deploy_lib_edge.so', **lib_kwargs)


This concludes the tutorial to compile a model using TVM with Vitis AI.
For instructions on how to run a compiled model please refer to the next section.

Inference
---------

The TVM with Vitis AI flow contains two stages: Compilation and Inference.
During the compilation a user can choose to compile a model for any of the
target devices that are currently supported. Once a model is compiled, the
generated files can be used to run the model on a target device during the
Inference stage.

Check out the `Running on Alveo and VCK5000 <#running-on-alveo-and-vck5000>`__
and `Running on Zynq and VCK190 <#running-on-zynq-and-vck190>`__ sections for
doing inference on cloud accelerator cards respectively edge boards.

Running on Alveo and VCK5000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After having followed the steps in the `Compiling a Model <#compiling-a-model>`__
section, you can continue running on new inputs inside the docker for accelerated
inference:

.. code:: python

    module.set_input(input_name, inputs[i])
    module.run()

Alternatively, you can load the exported runtime module (the deploy_lib.so
exported in  `Compiling a Model <#compiling-a-model>`__):

.. code:: python

   import pyxir
   import tvm
   from tvm.contrib import graph_executor

   dev = tvm.cpu()

   # input_name = ...
   # input_data = ...

   # load the module into memory
   lib = tvm.runtime.load_module("deploy_lib.so")

   module = graph_executor.GraphModule(lib["default"](dev))
   module.set_input(input_name, input_data)
   module.run()

Running on Zynq and VCK190
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before proceeding, please follow the  `Zynq <#zynq-setup>`__ or
`Versal VCK190 <#versal-vck190-setup>`__ setup instructions.

Prior to running a model on the board, you need to compile the model for
your target evaluation board and transfer the compiled model on to the board.
Please refer to the `Compiling a Model <#compiling-a-model>`__ section for
information on how to compile a model.

Afterwards, you will have to transfer the compiled model (deploy_lib_edge.so)
to the evaluation board. Then, on the board you can use the typical
"load_module" and "module.run" APIs to execute. For this, please make sure to
run the script as root (execute ``su`` in terminal to log into root).

.. note::

    Note also that you **shouldn't** import the
    PyXIR DPU targets in the run script (``import pyxir.contrib.target.DPUCZDX8G``).

.. code:: python

   import pyxir
   import tvm
   from tvm.contrib import graph_executor

   dev = tvm.cpu()

   # input_name = ...
   # input_data = ...

   # load the module into memory
   lib = tvm.runtime.load_module("deploy_lib_edge.so")

   module = graph_executor.GraphModule(lib["default"](dev))
   module.set_input(input_name, input_data)
   module.run()
