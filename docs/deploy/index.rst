Deploy and Integration
======================

This page contains guidelines on how to deploy TVM to various platforms
as well as how to integrate it with your project.

.. image::  http://www.tvm.ai/images/release/tvm_flexible.png

In order to integrate the compiled module, we do not have to ship the compiler stack. We only need to use a lightweight runtime API that can be integrated into various platforms.

.. toctree::
   :maxdepth: 2

   cpp_deploy
   android
   nnvm
   integrate
