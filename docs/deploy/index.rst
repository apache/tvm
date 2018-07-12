.. _deploy-and-integration:

Deploy and Integration
======================

This page contains guidelines on how to deploy TVM to various platforms
as well as how to integrate it with your project.

.. image::  http://www.tvm.ai/images/release/tvm_flexible.png

Unlike traditional deep learning frameworks. TVM stack is divided into two major components:

- TVM compiler, which does all the compilation and optimizations
- TVM runtime, which runs on the target devices.

In order to integrate the compiled module, we **do not** need to build entire TVM on the target device. You only need to build the TVM compiler stack on your desktop and use that to cross-compile modules that are deployed on the target device.
We only need to use a light-weight runtime API that can be integrated into various platforms.

For example, you can run the following commands to build the runtime API
on a Linux based embedded system such as Raspberry Pi:

.. code:: bash

    git clone --recursive https://github.com/dmlc/tvm
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
- :ref:`tutorial-deploy-model-on-mali-gpu`
- :ref:`tutorial-deploy-model-on-rasp`

After you finished tuning and benchmarking, you might need to deploy the model on the
target device without relying on RPC. see the following resources on how to do so.

.. toctree::
   :maxdepth: 2

   cpp_deploy
   android
   nnvm
   integrate
