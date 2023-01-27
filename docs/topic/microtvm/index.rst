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

.. _microtvm-index:

microTVM: TVM on bare-metal
===========================

microTVM runs TVM models on bare-metal (i.e. IoT) devices. microTVM depends only on the C standard
library, and doesn't require an operating system to execute. microTVM is currently under heavy
development.

.. figure:: https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_workflow.svg
   :align: center
   :width: 85%

microTVM is:

* an extension to TVM's compiler to allow it to target microcontrollers
* a way to run the TVM RPC server on-device, to allow autotuning
* a minimal C runtime that supports standalone model inference on bare metal devices.

Supported Hardware
~~~~~~~~~~~~~~~~~~

microTVM currently tests against Cortex-M microcontrollers with the Zephyr RTOS; however, it is
flexible and portable to other processors such as RISC-V and does not require Zephyr. The current
demos run against QEMU and the following hardware:

* `STM Nucleo-F746ZG <https://www.st.com/en/evaluation-tools/nucleo-f746zg.html>`_
* `STM STM32F746 Discovery <https://www.st.com/en/evaluation-tools/32f746gdiscovery.html>`_
* `nRF 5340 Development Kit <https://www.nordicsemi.com/Software-and-tools/Development-Kits/nRF5340-DK>`_


Getting Started with microTVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before working with microTVM, we recommend you have a supported development board. Then, follow these
tutorials to get started with microTVM. Tutorials are in the order that could help developers to learn
more as they follow through them. Here is a list of tutorials that you can start with:

1. Try :ref:`microTVM CLI Tool <tutorial-micro-cli-tool>`.
2. Try the :ref:`microTVM TFLite Tutorial <tutorial_micro_tflite>`.
3. Try running a more complex tutorial: :ref:`Creating Your MLPerfTiny Submission with microTVM <tutorial-micro-mlperftiny>`.


How microTVM Works
~~~~~~~~~~~~~~~~~~


You can read more about the design of these pieces at the :ref:`microTVM Design Document <microTVM-design>`.


Help and Discussion
~~~~~~~~~~~~~~~~~~~

The `TVM Discuss Forum <https://discuss.tvm.ai>`_ is a great place to collaborate on microTVM tasks,
and maintains a searchable history of past problems.
