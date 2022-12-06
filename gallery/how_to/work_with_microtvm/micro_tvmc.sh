# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

: '
.. _tutorial-micro-tvmc:

Executing a Tiny Model with TVMC Micro
======================================
**Author**: `Mehrdad Hessar <https://github.com/mehrdadh>`_

This tutorial explains how to compile a tiny model for a micro device,
build a program on Zephyr platform to execute this model, flash the program
and run the model all using `tvmc micro` command.
'

######################################################################
# .. note::
#     This tutorial is explaining using TVMC Mirco on Zephyr platform. You need
#     to install Zephyr dependencies before processing with this tutorial. Alternatively,
#     you can run this tutorial in one of the following ways which has Zephyr depencencies already installed.
#
#     * Use `microTVM Reference Virtual Machines <https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_reference_vm.html#sphx-glr-how-to-work-with-microtvm-micro-reference-vm-py>`_.
#     * Use QEMU docker image provided by TVM. Following these you will download and login to the docker image:
#
#     .. code-block:: bash
#
#       cd tvm
#       ./docker/bash.sh tlcpack/ci-qemu
#

# bash-ignore
shopt -s expand_aliases
alias tvmc="python3 -m tvm.driver.tvmc"
# bash-ignore

############################################################
# Using TVMC Micro
############################################################
#
# TVMC is a command-line tool which is installed as a part of TVM Python packages. Accessing this
# package varies based on your machine setup. In many cases, you can use the ``tvmc`` command directly.
# Alternatively, if you have TVM as a Python module on your ``$PYTHONPATH``, you can access this
# driver with ``python -m tvm.driver.tvmc`` command. This tutorial will use TVMC command as
# ``tvmc`` for simplicity.
#
# To check if you have TVMC command installed on your machine, you can run:
#
# bash
tvmc --help
# bash
# To compile a model for microtvm we use ``tvmc compile`` subcommand. The output of this command
# is used in next steps with ``tvmc micro`` subcommands. You can check the availability of TVMC Micro using:
#
# bash
tvmc micro --help
# bash
#
# The main tasks that you can perform using ``tvmc micro`` are ``create``, ``build`` and ``flash``.
# To read about specific options under a givern subcommand, use
# ``tvmc micro <subcommand> --help``. We will use each subcommand in this tutorial.
#

############################################################
# Obtain a Tiny Model
############################################################
#
# For this tutorial, we will use Magic Wand model from tflite micro. Magic Wand is a
# Depthwise Convolution Layer model which recognizes gestures with an accelerometer.
#
# For this tutorial we will be using the model in tflite format.
#
# bash
wget https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/magic_wand/magic_wand.tflite
# bash

############################################################
# Compiling a TFLite model to a Model Library Format
############################################################
#
# Model Library Format (MLF) is an output format that TVM provides for micro targets. MLF is a tarball
# containing a file for each piece of the TVM compiler output which can be used on micro targets outside
# TVM environment. Read more about `Model Library Format <https://tvm.apache.org/docs//arch/model_library_format.html>`_.
#
# Here, we generate a MLF file for ``qemu_x86`` Zephyr board. To generate MLF output for the ``magic_wand`` tflite model:
#
# bash
tvmc compile magic_wand.tflite \
    --target='c -keys=cpu -model=host' \
    --runtime=crt \
    --runtime-crt-system-lib 1 \
    --executor='graph' \
    --executor-graph-link-params 0 \
    --output model.tar \
    --output-format mlf \
    --pass-config tir.disable_vectorize=1 \
    --disabled-pass=AlterOpLayout
# bash
# This will generate a ``model.tar`` file which contains TVM compiler output files. To run this command for
# a different Zephyr device, you need to update ``target``. For instance, for ``nrf5340dk_nrf5340_cpuapp`` board
# the target is ``--target='c -keys=cpu -model=nrf5340dk'``.
#


############################################################
# Create a Zephyr Project Using Model Library Format
############################################################
#
# To generate a Zephyr project we use TVM Micro subcommand ``create``. We pass the MLF format and the path
# for the project to ``create`` subcommand along with project options. Project options for each
# platform (Zephyr/Arduino) are defined in their Project API server file. To build
# Zephyr project for a different Zephyr board, change ``zephyr_board`` project option.
# To generate Zephyr project, run:
#
# bash
tvmc micro create \
    project \
    model.tar \
    zephyr \
    --project-option project_type=host_driven zephyr_board=qemu_x86
# bash
# This will generate a ``Host-Driven`` Zephyr project for ``qemu_x86`` Zephyr board. In Host-Driven template project,
# the Graph Executor will run on host and perform the model execution on Zephyr device by issuing commands to the
# device using an RPC mechanism. Read more about `Host-Driven Execution <https://tvm.apache.org/docs/arch/microtvm_design.html#host-driven-execution>`_.
#
# To get more information about TVMC Micro ``create`` subcommand:
#
# .. code-block:: bash
#
#     tvmc micro create --help
#

############################################################
# Build and Flash Zephyr Project Using TVMC Micro
############################################################
#
# Next step is to build the Zephyr project which includes TVM generated code for running the tiny model, Zephyr
# template code to run a model in Host-Driven mode and TVM runtime source/header files. To build the project:
#
# bash
tvmc micro build \
    project \
    zephyr
# bash
# This will build the project in ``project`` directory and generates binary files under ``project/build``.
#
# Next, we flash the Zephyr binary file to Zephyr device. For ``qemu_x86`` Zephyr board this step does not
# actually perform any action since QEMU will be used, however you need this step for physical hardware.
#
# bash
tvmc micro flash \
    project \
    zephyr
# bash

############################################################
# Run Tiny Model on Micro Target
############################################################
#
# After flashing the device, the compiled model and TVM RPC server are programmed on the device.
# The Zephyr board is waiting for host to open a communication channel. MicroTVM devices typicall communicate
# using a serial communication (UART). To run the flashed model on the device using TVMC, we use ``tvmc run`` subcommand
# and pass ``--device micro`` to specify the device type. This command will open a communication channel, set input
# values using ``Graph Executor`` on host and run full model on the device. Then it gets output from the device.
#
# bash
tvmc run \
    --device micro \
    project \
    --fill-mode ones \
    --print-top 4
# bash
#     # Output:
#     #
#     # INFO:__main__:b'[100%] [QEMU] CPU: qemu32,+nx,+pae\n'
#     # remote: microTVM Zephyr runtime - running
#     # INFO:__main__:b'[100%] Built target run\n'
#     # [[3.         1.         2.         0.        ]
#     # [0.47213247 0.41364592 0.07525456 0.03896701]]
#
# Specifically, this command sets the input of the model to all ones and shows the four values of the output with their indices.
