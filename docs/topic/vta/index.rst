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

.. _vta-index:

VTA: Versatile Tensor Accelerator
=================================

The Versatile Tensor Accelerator (VTA) is an open, generic, and customizable deep learning accelerator with a complete TVM-based compiler stack. We designed VTA to expose the most salient and common characteristics of mainstream deep learning accelerators. Together TVM and VTA form an end-to-end hardware-software deep learning system stack that includes hardware design, drivers, a JIT runtime, and an optimizing compiler stack based on TVM.

.. image:: https://raw.githubusercontent.com/uwsampl/web-data/main/vta/blogpost/vta_overview.png
   :align: center
   :width: 60%

VTA has the following key features:

- Generic, modular, open-source hardware.
- Streamlined workflow to deploy to FPGAs.
- Simulator support to prototype compilation passes on regular workstations.
- Pynq-based driver and JIT runtime for both simulated and FPGA hardware back-end.
- End to end TVM stack integration.

This page contains links to all the resources related to VTA:


.. toctree::
   :maxdepth: 1

   install
   dev/index
   tutorials/index


Literature
----------

- Read the VTA `release blog post`_.
- Read the VTA tech report: `An Open Hardware Software Stack for Deep Learning`_.

.. _release blog post: https://tvm.apache.org/2018/07/12/vta-release-announcement
.. _An Open Hardware Software Stack for Deep Learning: https://arxiv.org/abs/1807.04188
