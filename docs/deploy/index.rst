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

.. _deploy-and-integration:

Deploy and Integration
======================

This page contains guidelines on how to deploy TVM to various platforms
as well as how to integrate it with your project.

.. image::  https://tvm.apache.org/images/release/tvm_flexible.png

Unlike traditional deep learning frameworks. TVM stack is divided into two major components:

- TVM compiler, which does all the compilation and optimizations
- TVM runtime, which runs on the target devices.

In order to integrate the compiled module, we **do not** need to build entire TVM on the target device. You only need to build the TVM compiler stack on your desktop and use that to cross-compile modules that are deployed on the target device.
We only need to use a light-weight runtime API that can be integrated into various platforms.

For example, you can run the following commands to build the runtime API
on a Linux based embedded system such as Raspberry Pi:

.. code:: bash

    git clone --recursive https://github.com/apache/incubator-tvm tvm
    cd tvm
    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make runtime

Note that we type `make runtime` to only build the runtime library.
If you want to include additional runtime such as OpenCL,
you can modify `config.cmake` to enable these options.
After you get the TVM runtime library, you can link the compiled library

The easiest and recommended way to test, tune and benchmark TVM kernels on
embedded devices is through TVM's RPC API.
Here are the links to the related tutorials.

- :ref:`tutorial-cross-compilation-and-rpc`
- :ref:`tutorial-deploy-model-on-rasp`

After you finished tuning and benchmarking, you might need to deploy the model on the
target device without relying on RPC. see the following resources on how to do so.

.. toctree::
   :maxdepth: 2

   cpp_deploy
   android
   integrate
   hls
