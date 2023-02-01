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

Deploy to Adreno™ GPU
=====================

**Authors**: Daniil Barinov, Egor Churaev, Andrey Malyshev, Siva Rama Krishna

Introduction
------------

Adreno™ is a series of graphics processing unit (GPU) semiconductor
intellectual property cores developed by Qualcomm and used in many of
their SoCs.

The Adreno™ GPU accelerates the rendering of complex geometries to
deliver high-performance graphics and a rich user experience with low
power consumption.

TVM supports deep learning acceleration on Adreno™ GPU by native OpenCL backend of TVM and
also through OpenCLML backend. Native OpenCL backend of TVM is enhanced to make it
Adreno™ friendly by incorporating texture memory usage and Adreno™ friendly layouts.
OpenCLML is an SDK release by Qualcomm that provides kernel acceleration library
for most of the deep learning operators.

This guide is organized to demonstrate various design aspects of

- :ref:`OpenCL Backend Ehnahcements<opencl_enhancements>`
- :ref:`About OpenCLML<about_openclml>`
- :ref:`Build and Deploy<build_deploy>`



.. how to :ref:`build TVM with OpenCL<building_tvm_for_adreno>` (needed by Adreno™ devices) and TVM RPC
.. enabled. It will also provide :ref:`example code<build_and_deploy_model_for_adreno>` to better understand the differences in compiling and deploying models
.. for Adreno™ devices.


.. _opencl_enhancements:

OpenCL Backend Enhancements
---------------------------

OpenCL backend of TVM is enhanced to take advantage of Adreno™ specific features like
- Texture memory usage.
- Adreno™ friendly activation layouts.
- Brand new schedules to accelerate with above features.

One of the Adreno™'s advantages is the clever handling of textures. At
the moment, TVM is able to benefit from this by having texture support
for Adreno™. The graph below shows the Adreno™ A5x architecture.

|High-level overview of the Adreno™ A5x architecture for OpenCL|

*Fig. 1 High-level overview of the Adreno™ A5x architecture for OpenCL*

*source:* `OpenCL Optimization and Best Practices for Qualcomm Adreno™ GPUs <https://dl.acm.org/doi/10.1145/3204919.3204935>`_

Reasons of using textures:

-  Texture processor (TP) has a dedicated L1 cache, which is read-only cache and stores data
   fetched from level-2 (L2) cache for texture operations (primary
   reason)

-  The handling of image boundaries is built-in.

-  Supports numerous image format and data type combinations with
   support for automatic format conversions

Overall, with textures, it is possible to achieve a significant performance boost
compared to OpenCL buffer based solutions.

.. _about_openclml:

About OpenCLML
--------------

OpenCLML is a SDK released by Qualcomm that provides accelerated deep learning operators.
These operators are exposed as an extension "cl_qcom_ml_ops" to standard OpenCL specification.
Please refer `Accelerate your models with our OpenCL ML SDK <https://developer.qualcomm.com/blog/accelerate-your-models-our-opencl-ml-sdk>`_ for more details.

OpenCLML is integrated into TVM as a `BYOC <https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html?highlight=bring%20your%20own>`_ solution.
OpenCLML operators can use same context and the operatrors can be enqueued on same command queue if native OpenCL.
We took advantage of this to avoid any context switching over heads while fallback to native OpenCL.


.. _build_deploy:

TVM for Adreno™
---------------

This section gives instructions about various ways of building and deploying model
to Adreno™ target. Adreno™ is a remote target which is connected to the host via ADB connection.
Deploying the compiled model here require use some tools on host as well as on target.

TVM has simplified user friendly command line based tools as well as
developer centric python API interface for various steps like auto tuning, building and deploying.

TVM compilation process for remote devices has multiple stages listed below.

**Model import:**
At this stage we import a model from well known frameworks like Tensorflow, PyTorch, ONNX ...etc.
This stage converts the given model into TVM's relay module format. Alternatively one can build a relay module manually
by using TVM's operator inventory too. TVM module generated here is a target independent representation of the graph.

**Auto Tuning:**
At this stage we tune the TVM generated kernels specific to a target. Auto tuning process requires
target device availability and in case of a remote target like Adreno™ on Android device we use RPC Setup for communication.
Later sections in this guide will detail about RPC Setup for Android device. Auto tuning is not a necessary step for
compilation of a model. It is necessary for acheiving best performance out of TVM generated kernels.

**Compilation:**
At this stage we compile the model for specific target. Given we auto tuned the module in previous stage,
TVM compilation make use of the tuning log for genetrating best performing kernels. TVM compilation process produces artifacts
containing kernel shared lib, graph definition in json format and parameters binary file in TVM specific format.

**Deploy (or test run) on Target:**
At this stage we run the TVM compilation output on the target. Deployment is possible from python
environment using RPC Setup and also using TVM's native tool which is native binary cross compiled for Android.
At this stage we can run the compiled model on Android target and unit test output correctness and performance aspects.

**Aplication Integration:**
This stage is all about integrating TVM compiled model in applications. Here we discuss about
interfacing tvm runtime from Android (cpp native environment or from JNI) for setting input and getting output.

**Advanced Usage:**
This section advanced user interests like viewing generated source code, altering precision of the module ...etc.


This tutorial covers all the above aspects as part of below sections.

- :ref:`Development environment<development_environment>`
- :ref:`RPC Setup<rpc_setup>`
- :ref:`Commandline tools<commandline_interface>`
- :ref:`Python interface<python_interface>`
- :ref:`Application Integration<application_integration>`
- :ref:`Advanced Usage<advanced_usage>`

.. _development_environment:


Development Environment Setup : Automatic
-----------------------------------------
TVM ships a predefined docker container environment with all prerequisites to get started quickly.
You may also refer to :ref:`Manual Environment Setup<manual_setup>` for more control on the dependencies.

For docker setup the pre requisite is just docker tool availabilty on host.

Below commands can build a docker image for adreno.

::

   ./docker/build.sh ci_adreno
   docker tag tvm.ci_adreno ci_adreno


Now we can build both host and target utils with below command.

::

   ./tests/scripts/ci.py adreno -i

To build TVM with OpenCLML SDK we need export the OpenCLML SDK as shown below while building

::

   export ADRENO_OPENCL=<Path to OpenCLML SDK>
   ./tests/scripts/ci.py adreno -i

On successful compilation this leaves us into a docker shell. The build leaves two folders

* build-adreno:  The host side TVM compiler build.
* build-adreno-target : Contains the android target components

    * libtvm_runtime.so : TVM runtime library
    * tvm_rpc : The rpc runtime environment tool
    * rtvm : A native stand alone tool

While using docker environment the android device is shared with host. Hence, it is required
to have adb version "1.0.41" on the host as the docker used the same version.

We can check adb devices availability inside docker environment too.

::

   user@ci-adreno-fpeqs:~$ adb devices
   List of devices attached
   aaaabbbb	device
   ccccdddd	device

.. _manual_setup:

Development Environment Setup : Manual
--------------------------------------

Manual build process require building of host and target components.

Below command will configure the build the host compiler

::

   mkdir -p build
   cd build
   cp ../cmake/config.cmake .

   echo set\(USE_OPENCL ON\) >> config.cmake
   echo set\(USE_RPC ON\) >> config.cmake
   echo set\(USE_GRAPH_EXECUTOR ON\) >> config.cmake
   echo set\(USE_LIBBACKTRACE AUTO\) >> config.cmake
   echo set\(USE_LLVM ON\) >> config.cmake

Additionally we can push below config entry to compile with OpenCLML support.

::

   export ADRENO_OPENCL=<Path to OpenCLML SDK>
   echo set\(USE_CLML ${ADRENO_OPENCL}\) >> config.cmake

now we can build as shown below

::

   cmake ..
   make

Finally we can export python path as

::

   export PYTHONPATH=$PWD:/python
   python3 -c "import tvm" # Verify tvm python package


Now, we can configure and build the target components with below configuration
Target build require Android NDK to be installed.

- Read documentation about *Android NDK installation* here: https://developer.android.com/ndk
- To get access to adb tools you can see *Android Debug Bridge installation* here: https://developer.android.com/studio/command-line/adb


::

   mkdir -p build-adreno
   cd build-adreno
   cp ../cmake/config.cmake .
   echo set\(USE_MICRO OFF\) >> config.cmake
   echo set\(USE_OPENCL ON\) >> config.cmake
   echo set\(USE_RPC ON\) >> config.cmake
   echo set\(USE_CPP_RPC ON\) >> config.cmake
   echo set\(USE_CPP_RTVM ON\) >> config.cmake
   echo set\(USE_GRAPH_EXECUTOR ON\) >> config.cmake
   echo set\(USE_LIBBACKTRACE AUTO\) >> config.cmake
   echo set\(USE_KALLOC_ALIGNMENT 32\) >> config.cmake

   echo set\(ANDROID_ABI arm64-v8a\) >> config.cmake
   echo set\(ANDROID_PLATFORM android-28\) >> config.cmake
   echo set\(MACHINE_NAME aarch64-linux-gnu\) >> config.cmake

Additionally we can push below config to compile with OpenCLML support.

::

   export ADRENO_OPENCL=<Path to OpenCLML SDK>
   echo set\(USE_CLML "${ADRENO_OPENCL}"\) >> config.cmake
   echo set\(USE_CLML_GRAPH_EXECUTOR "${ADRENO_OPENCL}"\) >> config.cmake

For Android target build ANDROID_NDK_HOME is a dependency and we should have the same in the enviromnet variable.
Below commands will build Adreno™ target components

::

   cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DCMAKE_SYSTEM_VERSION=1 \
      -DCMAKE_FIND_ROOT_PATH="${ADRENO_OPENCL}" \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_CXX_COMPILER="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++" \
      -DCMAKE_C_COMPILER="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang" \
      -DMACHINE_NAME="aarch64-linux-gnu" ..

   make tvm_runtime tvm_rpc rtvm


.. _rpc_setup:

RPC Setup
---------

RPC Setup allows remote target access over TCP/IP networking interface. RPC Setup is essential for auto tuning stage as tuning
involves running of auto generated kernels on real device and optimize the same by using machine learning approach. Please refer
`Auto-Tune with Templates and AutoTVM <https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html>`_ got more details about AutoTVM.

RPC Setup is also useful to deply the compiled model to a remote device from python interface or ```tvmc``` tool from host device.

RPC Setup has multiple components as listed below.

**TVM Tracker:**
TVM tracker is a host side daemon that manages remote devices and serve them to host side applications. Applications
can connect to this tracker and acquire a remote device handle to communicate.

**TVM RPC:**
TVM RPC is a native application that runs on the remote device (Android in our case) and registers itself to the TVM Tracker
running on the host.


Hence, for RPC based setup we will have above components running on host and target device. Below sections explain how to setup the same
manually and also inside docker using automated tools.

**Automated RPC Setup:**
Here we will explain how to setup RPC in docker environment.

Below command launches tracker in docker environment, where docker listens on port 9120.

::

   ./tests/scripts/ci.py adreno -i # Launch a new shell on the anreno docker
   source  tests/scripts/setup-adreno-env.sh -e tracker -p 9120

Now, the below comand can run TVM RPC on remote android device with id "abcdefgh".


::

   ./tests/scripts/ci.py adreno -i # Launch a new shell on adreno docker.
   source  tests/scripts/setup-adreno-env.sh -e device -p 9120 -d abcdefgh


**Manual RPC Setup:**

Below command in manual setup starts the tracker on port 9120

::

   python3 -m tvm.exec.rpc_tracker --host "0.0.0.0" --port "9120"

TVM RPC launch on Android device require some environment setup due to Android device is connected via ADB interface and we need to re-route
TCP/IP communication over ADB interface. Below commands will do necessary setup and run tvm_rpc on remote device.

::

    # Set android device to use
    export ANDROID_SERIAL=abcdefgh
    # Create a temporary folder on remote device.
    adb shell "mkdir -p /data/local/tmp/tvm_ci"
    # Copy tvm_rpc and it's dependency to remote device
    adb push build-adreno-target/tvm_rpc /data/local/tmp/tvm_test/tvm_rpc
    adb push build-adreno-target/libtvm_runtime.so /data/local/tmp/tvm_test
    # Forward port 9120 from target to host
    adb reverse tcp:9210 tcp:9120
    # tvm_rpc by default listens on ports starting from 5000 for incoming connections.
    # Hence, reroute connections to these ports on host to remore device.
    adb forward tcp:5000 tcp:5000
    adb forward tcp:5001 tcp:5001
    adb forward tcp:5002 tcp:5002
    # Finally launch rpc_daemon on remote device with identity key as "android"
    adb shell "cd /data/local/tmp/tvm_test; killall -9 tvm_rpc; sleep 2; LD_LIBRARY_PATH=/data/local/tmp/tvm_test/ ./tvm_rpc server --host=0.0.0.0 --port=5000 --port-end=5010 --tracker=127.0.0.1:9120 --key=android"

Upon successfull running this remote device will be available on tracker which can be queried as below.

::

   python3 -m tvm.exec.query_rpc_tracker --port 9120
   Tracker address 127.0.0.1:9120
   Server List
   ------------------------------
   server-address           key
   ------------------------------
       127.0.0.1:5000    server:android
   ------------------------------

   Queue Status
   -------------------------------
   key       total  free  pending
   -------------------------------
   android   1      1     0
   -------------------------------

This concludes RPC Setup and we have rpc-tracker available on host 127.0.0.1 (rpc-tracker) and port 9120 (rpc-port).


.. _commandline_interface:

Commandline Tools
-----------------

Here we describe entire compilation process using command line tools. TVM has command line utility "tvmc" to perform
model import, auto tuning, compilation and deply over rpc. "tvmc" has many options to explore and try.

**Model Import & Tuning:**
Use the below command to import a model from any framework and auto tune the same.
Here we use a model from Keras and it uses RPC setup for tuning and finally generates tuning log file
"keras-resnet50.log".

::

   python3 -m tvm.driver.tvmc tune --target="opencl -device=adreno" \
   --target-host="llvm -mtriple=aarch64-linux-gnu" \
   resnet50.h5 -o \
   keras-resnet50.log \
   --early-stopping 0 --repeat 30 --rpc-key android \
   --rpc-tracker 127.0.0.1:9120 --trials 1024 \
   --tuning-records keras-resnet50-records.log --tuner xgb

**Model Compilation:**

Use below command for compiling the model and produce TVM compiler outputs.

::

   python3 -m tvm.driver.tvmc compile \
   --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
   --target="opencl, llvm" --target-llvm-mtriple aarch64-linux-gnu --target-opencl-device adreno \
   --tuning-records keras-resnet50.log -o keras-resnet50.tar resnet50.h5

While enabled OpenCLML offloading we nee dto add target "clml" as shown below. Tuning log is valid for OpenCLML offloading also
as the OpenCL path is fallback option for any operator didn't go through OpenCLML path. The tuning log will be used for such operators.

::

   python3 -m tvm.driver.tvmc compile \
   --cross-compiler ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
   --target="opencl, clml, llvm" --target-llvm-mtriple aarch64-linux-gnu --target-opencl-device adreno \
   --tuning-records keras-resnet50.log -o keras-resnet50.tar resnet50.h5

On success ful compilation above commands produce "keras-resnet50.tar". It is a compressed archive with kernel shared lib, graph json and params binary.

**Deploy & Run on Target:**

Running the compiled model on Android target is possible in RPC way as well as native deployment.

We can use below tvmc command to deploy on remore target via RPC based setup.

::

   python3 -m tvm.driver.tvmc run --device="cl" keras-resnet50.tar \
   --rpc-key android --rpc-tracker 127.0.0.1:9120 --print-time

tvmc based run has more option to initialize the input in various modes line fill, random ..etc.


TVM also supports "rtvm" tool to run the model narivelu on ADB shell. The build process produced this tool under build-adreno-target.
Please refer to `rtvm <https://github.com/apache/tvm/tree/main/apps/cpp_rtvm>`_ for more details about this tool.


.. _python_interface:

This section explains importing, auto tuning, compiling and running a model using python interface.\
TVM has a high level interface through tvmc abstraction as well as relay api. We will discuss about both of these in details.

Unlike command line interface python interface starts with model importing. Model importing converts the models from any framework
to a relay module. Relay module will be used across the auto tuning, compilation stages.

**TVMC Interface:**

TVMC interface can be accessed as shown below to import, compile and run a model.

.. code:: python

   from tvm.driver import tvmc
   from tvm.driver.tvmc.model import TVMCPackage

   # Convert a model from any framework to a tvm relay module.
   # tvmc.load supports models from any framework (like tensorflow saves_model, onnx, tflite ..etc) and auto detects the filetype.
   tvmc_model = tvmc.load("resnet50.h5")

   # tvmc_model consists of tvmc_mode.mod which is relay module and tvmc_model.params which parms of the module.

   # Now, the below api can be used for autotuning the model for any target. Tuning required RPC setup and please refer to
   # :ref:`RPC Setup<rpc_setup>` for the same.

   tvmc.tune(
     tvmc_model,
     target="opencl -device=adreno",
     output="keras-resnet50.log",
     tuning_records="keras-resnet50-records.log",
     target_host="llvm -mtriple=aarch64-linux-gnu"
     rpc_tracker="127.0.0.1:9120",
     rpc_key=android,
     repeat=30,
     trials=1024,
     early_stopping=0,
   )

   # Compilation to produce tvm artifacts

   tvmc_package = tvmc.compile(
      tvmc_model,
      target="opencl -device=adreno",
      target_host="llvm -mtriple=aarch64-linux-gnu",
      cross="/android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang",
      tuning_records="keras-resnet50.log",
   )

   # tvmc_package consists of tvmc_package.lib_path, tvmc_package.graph, tvmc_package.params

   # Altrernatively, we can ave the cmpilation output and save it as a TVMCPackage.
   # This way avoids loading of compiled module without compiling again.

   tvmc.compile(
      tvmc_model,
      target="opencl -device=adreno",
      target_host="llvm -mtriple=aarch64-linux-gnu",
      cross="/android_ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang",
      tuning_records="keras-resnet50.log",
      package_path="keras-resnet50.tar"
   )
   # Load the compiled package
   tvmc_package = TVMCPackage(package_path=module_file)

   # Saved TVMPackage is nothing but tar archive with mod.so, mod.json and mod.params.

   # Deploy and run the compiled model on RPC
   # Prepare input data dict
   input_data = tvm.nd.array((np.random.uniform(size=(1, 229, 229, 3))).astype("float32"))
   input_dict = {"input": input_data}

   # Run on RPC setup
   result = tvmc.run(
      tvmc_package,
      device="cl",
      rpc_key="android",
      hostname="127.0.0.1",
      port=9120,
      inputs=input_dict
   )

   # result is a dictionary of outputs.


tvmc compiled package can be used for native deploy also using "rtvm" utility.
Please refer to `rtvm <https://github.com/apache/tvm/tree/main/apps/cpp_rtvm#readme>`_ for more details about this tool.

Also, please refer to tvmc documentation for more details about the api interface.

**Relay Interface:**

Relay api interface gives lower level api access to the tvm compiler interface.
Relay interface follows tvmc kind os a flow where we produce TVM module first followed by auto tuning, compilation and deployment.

Below example explains about relay interface usage

.. code:: python

   import tvm
   from tvm import relay
   from tvm.relay.op.contrib import clml
   import numpy as np

   from tensorflow.keras.applications import InceptionV3
   import tensorflow as tf

   target = "opencl -device=adreno"
   target_host = "llvm -mtriple=arm64-linux-android"

   # We first need to get a handle for a model from any framework.
   # In this example we will prepare a keras InceptionV3 model
   tf.keras.backend.clear_session()
   keras_net = InceptionV3(
       include_top=True, weights=None, input_shape=(299, 299, 3), classes=1000
   )
   input_info = {inceptionV3.input_names[0]: (1, 3, 299, 299)}
   input_data = {inceptionV3.input_names[0], np.random.uniform(-1, -1, (1, 3, 299, 299)).astype("float32")}
   from tensorflow.keras.layers import Input
   from tensorflow.keras.models import Model
   def get_bottom_top_model(model, layer_name):
       layer = model.get_layer(layer_name)
       bottom_input = model.layers[0].input
       bottom_output = layer.output
       bottom_model = Model(bottom_input, bottom_output)
       return bottom_model
   keras_model = get_bottom_top_model(keras_net, "predictions")
   ref_output = keras_model.predict(data["input_1"].transpose(0, 2, 3, 1))

   # Now we have a keras_model with input "input_1" with shape (1, 3, 299,299), output "predictions" and a reference output ref_output.

   # Lets import the model and get a relay module. TVM has frontend api for various frameworks under relay.frontend and now for keras
   # model import we have relay.frontend.from_keras api.
   mod, params = relay.frontend.from_keras(keras_model, input_info, layout="NCHW")

   # With relay module mod and parameters params we can not fo for tuning followed by compilation.
   # The below few instructions can auto tune the relay module with xgboost being the tuner algorithm.

   # Auto Tuning process involces stages of extracting the tasks, defining tuning congiguration and
   # tuning each task for best performing kernel configuration.

   # Auto Tuning Stage 1: Extract tunable tasks
   tasks = autotvm.task.extract_from_program(
       net, target=target, target_host=target_host, params=params
   )

   # Auto Tuning Stage 2: Define tuning configuration
   tune_log = "adreno-resnet50.log"
   tmp_log_file = tune_log + ".tmp"
   measure_option = autotvm.measure_option(
       builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15), # Build the test kernel locally
       runner=autotvm.RPCRunner( # The runner would be on a remote device.
           "android",            # RPC Key
           host="127.0.0.1",     # Tracker host
           port=9120,            # Tracker port
           number=3,             # Number of runs before averaging
           timeout=600,          # RPC Timeout
       ),
   ),
   n_trail = 1024                # Number of iteration of training before choosing the best kernel config
   early_stopping=False,         # Do we apply early stopping when the loss is not minimizing

   # Iterate through each task and call the tuner
   from tvm.autotvm.tuner import XGBTuner
   for i, tsk in enumerate(reversed(tasks)):
       tuner_obj = XGBTuner(tsk, loss_type="rank")

       tsk_trial = min(n_trial, len(tsk.config_space))
       tuner_obj.tune(
           n_trial=tsk_trial,
           early_stopping=early_stopping,
           measure_option=measure_option,
           callbacks=[
               autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
               autotvm.callback.log_to_file(tmp_log_file),
           ],
       )
   # Pick the best performing kerl configurations from the overall log.
   autotvm.record.pick_best(tmp_log_file, log_filename)


   # Given we have relay module and it's best performing kernel configurations
   # We can now go for compilation with tuned log or without tuning log if auto tuning is not enabled.

   if os.path.exists(tune_log):
       with autotvm.apply_history_best(tune_log):
           with tvm.transform.PassContext(opt_level=3):
               # Enable CLML partitioning if required.
               net = clml.partition_for_clml(net, params)

               lib = relay.build(
                   net, target=tvm.target.Target(target, host=target_host), params=params
               )
   else:
       with tvm.transform.PassContext(opt_level=3):
           # Enable CLML partitioning if required.
           net = clml.partition_for_clml(net, params)
           lib = relay.build(
               net, target=tvm.target.Target(target, host=target_host), params=params
           )

   # Compilation results a lib module and it has everything required to deploy on target.
   # We can save the compiler artifacts as shoun below and reload them later without entire compilation.
   lib.export_library("mod.so", ndk.create_shared)
   with open("mod.json", "w") as fo:
       fo.write(graph.json())
   with open("mod.params", "wb") as fo:
       fo.write(runtime.save_param_dict(params))

   # We can prepare TVMPackage from above files by art archiveing the same.
   # The tar archive can be used with tvmc tool or tvmc api interfae to deploy and run.
   # The tar archive can be used with "rtvm" tool also for native deploy on target device.

   # Now, lets look at deploying the compiled tvm artifact on remote target and run
   tmp = tempdir()
   filename = "%s.so" % network
   lib.export_library(tmp.relpath(filename), ndk.create_shared)

   # connect to remote device
   tracker = tvm.rpc.connect_tracker("127.0.0.1", 9120)
   remote = tracker.request("android")
   dev = remote.device(str(target), 0)
   remote.upload(tmp.relpath(filename))
   rlib = remote.load_module(filename)

   # Create Graph runtime module on remote device
   module = runtime.GraphModule(rlib["default"](dev))
   # Set input
   module.set_input("input_1", input_data["input_1"])
   # Get output
   output = module.get_output(0)


.. _application_integration:

Aplication Integration:
----------------------

TVM compilation output is represented as module shared lib (mod.so), graph json(mod.json) and params (mod.params).
Archived representation of TVMPackage is also contains the same.

In general a CPP/C based interface will be sufficient for any Android application integration.

TVM natively expose c_runtime_api for loading a TVM compiled module and run the same.

Alternatively one may refer to `cpp_rtvm <https://github.com/apache/tvm/blob/main/apps/cpp_rtvm/tvm_runner.h>`_
tvm_runner interface too for further simplified version of the same.



.. _advanced_usage:

Advanced Usage:
---------------

This section details some of the advanced usage and additional information whihc using Adreno™ target on TVM.

Generated Source Inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Apart from standard tvm compilation artifacts kernel library (mod.so), graph (mod.json) and params (mod.params)
we can also generate opencl kernel source, clml offloaded graph ...etc from lib handle as shown below.
TVM compilation output is organized as a TVM module and many other TVM modules imported into it.

Below snippet can dump CLML sub graphs in json format.

.. code:: python

   # Look for "clml" typed module impoted.
   clml_modules = list(filter(lambda mod: mod.type_key == "clml", lib.get_lib().imported_modules))
   # Loop throught all clml sub graphs and dump the json formatted CLML sub graphs.
   for cmod in clml_modules:
       print("CLML Src:", cmod.get_source())


Similarly, below snippet can extract opencl kernel source from the compiled TVM module.

.. code:: python

   # Similarly we can dump open kernel source too as shown below
   # Look for "opencl" typed module impoted.
   opencl_modules = list(filter(lambda mod: mod.type_key == "opencl", lib.get_lib().imported_modules))
   # Now dump open cource for each opencl targetted sub graph.
   for omod in opencl_modules:
       print("OpenCL Src:", omod.get_source())


Inspecting above code for target device "opencl --device=adreno" shows texture usage (image2d_t) as shown below.

.. code:: c

   __kernel void tvmgen_default_fused_nn_conv2d_kernel0(__write_only image2d_t pad_temp_global_texture, __read_only image2d_t p0) {
   // body..

*image2d_t* is a built-in OpenCL types that represents two-dimensional image object and provides several additional functions.
When we use *image2d_t* we read *4 elements at one time*, and it helps to utilize hardware in a more efficient way.

Precisions
~~~~~~~~~~
The right choice of precision for a specific workload can greatly increase the efficiency of the solution,
shifting the initial balance of precision and speed to the side that is a priority for the problem.

We can choose from *float16*, *float16_acc32* (Mixed Precision), *float32* (standard).

**Float16**

To leverage the GPU hardware capabilities and utilize the benefits of half precision computation and memory management,
we can convert an original model having floating points operation to a model operating with half precision.
Choosing lower precision will positively affect the performance of the model, but it may also have a decrease in the accuracy of the model.

To do the conversion you need to call adreno specific transformation API as soon relay module is generated through any frontend:

.. code:: python

   from tvm.relay.op.contrib import adreno
   adreno.convert_to_dtype(mod["main"], "float16")


We then can compile our model in any convinient way

.. code:: python

   with  tvm.transform.PassContext(opt_level=3):
       lib = relay.build(
           mod, target_host=target_host, target=target, params=params
       )


**float16_acc32 (Mixed Precision)**

ToMixedPrecision pass traverse over the network and split network to clusters of ops dealing with float or float16 data types.
The clusters are defined by three types of operations:
- Operations always be converted into float16 data type
- Operations which can be converted if they follow by converted cluster
- Operations never be converted to the float16 data type
This list is defined in the ToMixedPrecision implementation here
`relay/transform/mixed_precision.py <https://github.com/apache/tvm/blob/main/python/tvm/relay/transform/mixed_precision.py#L34>`_
and can be overridden by user.

The ``ToMixedPrecision`` method is a pass to convert an FP32 relay graph into an FP16 version (with
FP16 or FP32 accumulation dtypes). Doing this transformation is useful for reducing model size
as it halves the expected size of the weights (FP16_acc16 case).

ToMixedPrecision pass usage is simplified into a simple call as shown below for usage.

.. code:: python

   from tvm.relay.op.contrib import adreno
   adreno.convert_to_dtype(mod["main"], "float16_acc32")


We then can compile our model in any convinient way

.. code:: python

   with  tvm.transform.PassContext(opt_level=3):
       lib = relay.build(
           mod, target_host=target_host, target=target, params=params
       )

.. |High-level overview of the Adreno™ A5x architecture for OpenCL| image:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/how-to/adreno_architecture.png
.. |Android deployment pipeline| image:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/how-to/android_deployment_pipeline.jpg
