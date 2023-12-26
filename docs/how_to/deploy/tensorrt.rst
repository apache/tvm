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

Relay TensorRT Integration
==========================
**Author**: `Trevor Morris <https://github.com/trevor-m>`_

Introduction
------------

NVIDIA TensorRT is a library for optimized deep learning inference. This integration will offload as
many operators as possible from Relay to TensorRT, providing a performance boost on NVIDIA GPUs
without the need to tune schedules.

This guide will demonstrate how to install TensorRT and build TVM with TensorRT BYOC and runtime
enabled. It will also provide example code to compile and run a ResNet-18 model using TensorRT and
how to configure the compilation and runtime settings. Finally, we document the supported operators
and how to extend the integration to support other operators.

Installing TensorRT
-------------------

In order to download TensorRT, you will need to create an NVIDIA Developer program account. Please
see NVIDIA's documentation for more info:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html. If you have a Jetson device
such as a TX1, TX2, Xavier, or Nano, TensorRT will already be installed on the device via the
JetPack SDK.

There are two methods to install TensorRT:

* System install via deb or rpm package.
* Tar file installation.

With the tar file installation method, you must provide the path of the extracted tar archive to
USE_TENSORRT_RUNTIME=/path/to/TensorRT. With the system install method,
USE_TENSORRT_RUNTIME=ON will automatically locate your installation.

Building TVM with TensorRT support
----------------------------------

There are two separate build flags for TensorRT integration in TVM. These flags also enable
cross-compilation: USE_TENSORRT_CODEGEN=ON will also you to build a module with TensorRT support on
a host machine, while USE_TENSORRT_RUNTIME=ON will enable the TVM runtime on an edge device to
execute the TensorRT module. You should enable both if you want to compile and also execute models
with the same TVM build.

* USE_TENSORRT_CODEGEN=ON/OFF - This flag will enable compiling a TensorRT module, which does not require any
  TensorRT library.
* USE_TENSORRT_RUNTIME=ON/OFF/path-to-TensorRT - This flag will enable the TensorRT runtime module.
  This will build TVM against the installed TensorRT library.

Example setting in config.cmake file:

.. code:: cmake

    set(USE_TENSORRT_CODEGEN ON)
    set(USE_TENSORRT_RUNTIME /home/ubuntu/TensorRT-7.0.0.11)


Build and Deploy ResNet-18 with TensorRT
----------------------------------------

Create a Relay graph from a MXNet ResNet-18 model.

.. code:: python

    import tvm
    from tvm import relay
    import mxnet
    from mxnet.gluon.model_zoo.vision import get_model

    dtype = "float32"
    input_shape = (1, 3, 224, 224)
    block = get_model('resnet18_v1', pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)


Annotate and partition the graph for TensorRT. All ops which are supported by the TensorRT
integration will be marked and offloaded to TensorRT. The rest of the ops will go through the
regular TVM CUDA compilation and code generation.

.. code:: python

    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod = partition_for_tensorrt(mod, params)


Build the Relay graph, using the new module and config returned by partition_for_tensorrt. The
target must always be a cuda target. ``partition_for_tensorrt`` will automatically fill out the
required values in the config, so there is no need to modify it - just pass it along to the
PassContext so the values can be read during compilation.

.. code:: python

    target = "cuda"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)


Export the module.

.. code:: python

    lib.export_library('compiled.so')


Load module and run inference on the target machine, which must be built with
``USE_TENSORRT_RUNTIME`` enabled. The first run will take longer because the TensorRT engine will
have to be built.

.. code:: python

    dev = tvm.cuda(0)
    loaded_lib = tvm.runtime.load_module('compiled.so')
    gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))
    input_data = np.random.uniform(0, 1, input_shape).astype(dtype)
    gen_module.run(data=input_data)


Partitioning and Compilation Settings
-------------------------------------

There are some options which can be configured in ``partition_for_tensorrt``.

* ``version`` - TensorRT version to target as tuple of (major, minor, patch). If TVM is compiled
  with USE_TENSORRT_RUNTIME=ON, the linked TensorRT version will be used instead. The version
  will affect which ops can be partitioned to TensorRT.
* ``use_implicit_batch`` - Use TensorRT implicit batch mode (default true). Setting to false will
  enable explicit batch mode which will widen supported operators to include those which modify the
  batch dimension, but may reduce performance for some models.
* ``remove_no_mac_subgraphs`` - A heuristic to improve performance. Removes subgraphs which have
  been partitioned for TensorRT if they do not have any multiply-accumulate operations. The removed
  subgraphs will go through TVM's standard compilation instead.
* ``max_workspace_size`` - How many bytes of workspace size to allow each subgraph to use for
  TensorRT engine creation. See TensorRT documentation for more info. Can be overriden at runtime.


Runtime Settings
----------------

There are some additional options which can be configured at runtime using environment variables.

* Automatic FP16 Conversion - Environment variable ``TVM_TENSORRT_USE_FP16=1`` can be set to
  automatically convert the TensorRT components of your model to 16-bit floating point precision.
  This can greatly increase performance, but may cause some slight loss in the model accuracy.
* Caching TensorRT Engines - During the first inference, the runtime will invoke the TensorRT API
  to build an engine. This can be time consuming, so you can set ``TVM_TENSORRT_CACHE_DIR`` to
  point to a directory to save these built engines to on the disk. The next time you load the model
  and give it the same directory, the runtime will load the already built engines to avoid the long
  warmup time. A unique directory is required for each model.
* TensorRT has a paramter to configure the maximum amount of scratch space that each layer in the
  model can use. It is generally best to use the highest value which does not cause you to run out
  of memory. You can use ``TVM_TENSORRT_MAX_WORKSPACE_SIZE`` to override this by specifying the
  workspace size in bytes you would like to use.
* For models which contain a dynamic batch dimension, the varaible ``TVM_TENSORRT_MULTI_ENGINE``
  can be used to determine how TensorRT engines will be created at runtime. The default mode,
  ``TVM_TENSORRT_MULTI_ENGINE=0``, will maintain only one engine in memory at a time. If an input
  is encountered with a higher batch size, the engine will be rebuilt with the new max_batch_size
  setting. That engine will be compatible with all batch sizes from 1 to max_batch_size. This mode
  reduces the amount of memory used at runtime. The second mode, ``TVM_TENSORRT_MULTI_ENGINE=1``
  will build a unique TensorRT engine which is optimized for each batch size that is encountered.
  This will give greater performance, but will consume more memory.


Operator support
----------------
+------------------------+------------------------------------+
|       Relay Node       |              Remarks               |
+========================+====================================+
| nn.relu                |                                    |
+------------------------+------------------------------------+
| sigmoid                |                                    |
+------------------------+------------------------------------+
| tanh                   |                                    |
+------------------------+------------------------------------+
| nn.batch_norm          |                                    |
+------------------------+------------------------------------+
| nn.layer_norm          |                                    |
+------------------------+------------------------------------+
| nn.softmax             |                                    |
+------------------------+------------------------------------+
| nn.conv1d              |                                    |
+------------------------+------------------------------------+
| nn.conv2d              |                                    |
+------------------------+------------------------------------+
| nn.dense               |                                    |
+------------------------+------------------------------------+
| nn.bias_add            |                                    |
+------------------------+------------------------------------+
| add                    |                                    |
+------------------------+------------------------------------+
| subtract               |                                    |
+------------------------+------------------------------------+
| multiply               |                                    |
+------------------------+------------------------------------+
| divide                 |                                    |
+------------------------+------------------------------------+
| power                  |                                    |
+------------------------+------------------------------------+
| maximum                |                                    |
+------------------------+------------------------------------+
| minimum                |                                    |
+------------------------+------------------------------------+
| nn.max_pool2d          |                                    |
+------------------------+------------------------------------+
| nn.avg_pool2d          |                                    |
+------------------------+------------------------------------+
| nn.global_max_pool2d   |                                    |
+------------------------+------------------------------------+
| nn.global_avg_pool2d   |                                    |
+------------------------+------------------------------------+
| exp                    |                                    |
+------------------------+------------------------------------+
| log                    |                                    |
+------------------------+------------------------------------+
| sqrt                   |                                    |
+------------------------+------------------------------------+
| abs                    |                                    |
+------------------------+------------------------------------+
| negative               |                                    |
+------------------------+------------------------------------+
| nn.batch_flatten       |                                    |
+------------------------+------------------------------------+
| expand_dims            |                                    |
+------------------------+------------------------------------+
| squeeze                |                                    |
+------------------------+------------------------------------+
| concatenate            |                                    |
+------------------------+------------------------------------+
| nn.conv2d_transpose    |                                    |
+------------------------+------------------------------------+
| transpose              |                                    |
+------------------------+------------------------------------+
| layout_transform       |                                    |
+------------------------+------------------------------------+
| reshape                |                                    |
+------------------------+------------------------------------+
| nn.pad                 |                                    |
+------------------------+------------------------------------+
| sum                    |                                    |
+------------------------+------------------------------------+
| prod                   |                                    |
+------------------------+------------------------------------+
| max                    |                                    |
+------------------------+------------------------------------+
| min                    |                                    |
+------------------------+------------------------------------+
| mean                   |                                    |
+------------------------+------------------------------------+
| nn.adaptive_max_pool2d |                                    |
+------------------------+------------------------------------+
| nn.adaptive_avg_pool2d |                                    |
+------------------------+------------------------------------+
| nn.batch_matmul        |                                    |
+------------------------+------------------------------------+
| clip                   | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| nn.leaky_relu          | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| sin                    | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| cos                    | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| atan                   | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| ceil                   | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| floor                  | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| split                  | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| strided_slice          | Requires TensorRT 5.1.5 or greater |
+------------------------+------------------------------------+
| nn.conv3d              | Requires TensorRT 6.0.1 or greater |
+------------------------+------------------------------------+
| nn.max_pool3d          | Requires TensorRT 6.0.1 or greater |
+------------------------+------------------------------------+
| nn.avg_pool3d          | Requires TensorRT 6.0.1 or greater |
+------------------------+------------------------------------+
| nn.conv3d_transpose    | Requires TensorRT 6.0.1 or greater |
+------------------------+------------------------------------+
| erf                    | Requires TensorRT 7.0.0 or greater |
+------------------------+------------------------------------+


Adding a new operator
---------------------
To add support for a new operator, there are a series of files we need to make changes to:

* `src/runtime/contrib/tensorrt/tensorrt_ops.cc` Create a new op converter class which
  implements the ``TensorRTOpConverter`` interface. You must implement the constructor to specify how
  many inputs there are and whether they are tensors or weights. You must also implement the
  ``Convert`` method to perform the conversion. This is done by using the inputs, attributes, and
  network from params to add the new TensorRT layers and push the layer outputs. You can use the
  existing converters as an example. Finally, register your new op conventer in the
  ``GetOpConverters()`` map.
* `python/relay/op/contrib/tensorrt.py` This file contains the annotation rules for TensorRT. These
  determine which operators and their attributes that are supported. You must register an annotation
  function for the relay operator and specify which attributes are supported by your converter, by
  checking the attributes are returning true or false.
* `tests/python/contrib/test_tensorrt.py` Add unit tests for the given operator.
