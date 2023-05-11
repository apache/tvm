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
"""
Compiling and Optimizing a Model with TVMC
==========================================
**Authors**:
`Leandro Nunes <https://github.com/leandron>`_,
`Matthew Barrett <https://github.com/mbaret>`_,
`Chris Hoge <https://github.com/hogepodge>`_

In this section, we will work with TVMC, the TVM command line driver. TVMC is a
tool that exposes TVM features such as auto-tuning, compiling, profiling and
execution of models through a command line interface.

Upon completion of this section, we will have used TVMC to accomplish the
following tasks:

* Compile a pre-trained ResNet-50 v2 model for the TVM runtime.
* Run a real image through the compiled model, and interpret the output and
  model performance.
* Tune the model on a CPU using TVM.
* Re-compile an optimized model using the tuning data collected by TVM.
* Run the image through the optimized model, and compare the output and model
  performance.

The goal of this section is to give you an overview of TVM and TVMC's
capabilities, and set the stage for understanding how TVM works.
"""


################################################################################
# Using TVMC
# ----------
#
# TVMC is a Python application, part of the TVM Python package.
# When you install TVM using a Python package, you will get TVMC as
# as a command line application called ``tvmc``. The location of this command
# will vary depending on your platform and installation method.
#
# Alternatively, if you have TVM as a Python module on your
# ``$PYTHONPATH``, you can access the command line driver functionality
# via the executable python module, ``python -m tvm.driver.tvmc``.
#
# For simplicity, this tutorial will mention TVMC command line using
# ``tvmc <options>``, but the same results can be obtained with
# ``python -m tvm.driver.tvmc <options>``.
#
# You can check the help page using:
#
# .. code-block:: bash
#
#   tvmc --help
#
# The main features of TVM available to ``tvmc`` are from subcommands
# ``compile``, and ``run``, and ``tune``.  To read about specific options under
# a given subcommand, use ``tvmc <subcommand> --help``. We will cover each of
# these commands in this tutorial, but first we need to download a pre-trained
# model to work with.
#


################################################################################
# Obtaining the Model
# -------------------
#
# For this tutorial, we will be working with ResNet-50 v2. ResNet-50 is a
# convolutional neural network that is 50 layers deep and designed to classify
# images. The model we will be using has been pre-trained on more than a
# million images with 1000 different classifications. The network has an input
# image size of 224x224. If you are interested exploring more of how the
# ResNet-50 model is structured, we recommend downloading `Netron
# <https://netron.app>`_, a freely available ML model viewer.
#
# For this tutorial we will be using the model in ONNX format.
#
# .. code-block:: bash
#
#   wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
#

################################################################################
# .. admonition:: Supported model formats
#
#   TVMC supports models created with Keras, ONNX, TensorFlow, TFLite
#   and Torch. Use the option ``--model-format`` if you need to
#   explicitly provide the model format you are using. See ``tvmc
#   compile --help`` for more information.
#

################################################################################
# .. admonition:: Adding ONNX Support to TVM
#
#    TVM relies on the ONNX python library being available on your system. You can
#    install ONNX using the command ``pip3 install --user onnx onnxoptimizer``. You
#    may remove the ``--user`` option if you have root access and want to install
#    ONNX globally.  The ``onnxoptimizer`` dependency is optional, and is only used
#    for ``onnx>=1.9``.
#

################################################################################
# Compiling an ONNX Model to the TVM Runtime
# ------------------------------------------
#
# Once we've downloaded the ResNet-50 model, the next step is to compile it. To
# accomplish that, we are going to use ``tvmc compile``. The output we get from
# the compilation process is a TAR package of the model compiled to a dynamic
# library for our target platform. We can run that model on our target device
# using the TVM runtime.
#
# .. code-block:: bash
#
#   # This may take several minutes depending on your machine
#   tvmc compile \
#   --target "llvm" \
#   --input-shapes "data:[1,3,224,224]" \
#   --output resnet50-v2-7-tvm.tar \
#   resnet50-v2-7.onnx
#
# Let's take a look at the files that ``tvmc compile`` creates in the module:
#
# .. code-block:: bash
#
# 	mkdir model
# 	tar -xvf resnet50-v2-7-tvm.tar -C model
# 	ls model
#
# You will see three files listed.
#
# * ``mod.so`` is the model, represented as a C++ library, that can be loaded
#   by the TVM runtime.
# * ``mod.json`` is a text representation of the TVM Relay computation graph.
# * ``mod.params`` is a file containing the parameters for the pre-trained
#   model.
#
# This module can be directly loaded by your application, and the model can be
# run via the TVM runtime APIs.


################################################################################
# .. admonition:: Defining the Correct Target
#
#   Specifying the correct target (option ``--target``) can have a huge
#   impact on the performance of the compiled module, as it can take
#   advantage of hardware features available on the target. For more
#   information, please refer to :ref:`Auto-tuning a convolutional network for
#   x86 CPU <tune_relay_x86>`. We recommend identifying which CPU you are
#   running, along with optional features, and set the target appropriately.

################################################################################
# Running the Model from The Compiled Module with TVMC
# ----------------------------------------------------
#
# Now that we've compiled the model to this module, we can use the TVM runtime
# to make predictions with it. TVMC has the TVM runtime built in to it,
# allowing you to run compiled TVM models. To use TVMC to run the model and
# make predictions, we need two things:
#
# - The compiled module, which we just produced.
# - Valid input to the model to make predictions on.
#
# Each model is particular when it comes to expected tensor shapes, formats and
# data types. For this reason, most models require some pre and
# post-processing, to ensure the input is valid and to interpret the output.
# TVMC has adopted NumPy's ``.npz`` format for both input and output data. This
# is a well-supported NumPy format to serialize multiple arrays into a file.
#
# As input for this tutorial, we will use the image of a cat, but you can feel
# free to substitute this image for any of your choosing.
#
# .. image:: https://s3.amazonaws.com/model-server/inputs/kitten.jpg
#    :height: 224px
#    :width: 224px
#    :align: center


################################################################################
# Input pre-processing
# ~~~~~~~~~~~~~~~~~~~~
#
# For our ResNet-50 v2 model, the input is expected to be in ImageNet format.
# Here is an example of a script to pre-process an image for ResNet-50 v2.
#
# You will need to have a supported version of the Python Image Library
# installed. You can use ``pip3 install --user pillow`` to satisfy this
# requirement for the script.
#
# .. code-block:: python
#     :caption: preprocess.py
#     :name: preprocess.py
#
#     #!python ./preprocess.py
#     from tvm.contrib.download import download_testdata
#     from PIL import Image
#     import numpy as np
#
#     img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
#     img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
#
#     # Resize it to 224x224
#     resized_image = Image.open(img_path).resize((224, 224))
#     img_data = np.asarray(resized_image).astype("float32")
#
#     # ONNX expects NCHW input, so convert the array
#     img_data = np.transpose(img_data, (2, 0, 1))
#
#     # Normalize according to ImageNet
#     imagenet_mean = np.array([0.485, 0.456, 0.406])
#     imagenet_stddev = np.array([0.229, 0.224, 0.225])
#     norm_img_data = np.zeros(img_data.shape).astype("float32")
#     for i in range(img_data.shape[0]):
#    	  norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]
#
#     # Add batch dimension
#     img_data = np.expand_dims(norm_img_data, axis=0)
#
#     # Save to .npz (outputs imagenet_cat.npz)
#     np.savez("imagenet_cat", data=img_data)
#

################################################################################
# Running the Compiled Module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With both the model and input data in hand, we can now run TVMC to make a
# prediction:
#
# .. code-block:: bash
#
#     tvmc run \
#     --inputs imagenet_cat.npz \
#     --output predictions.npz \
#     resnet50-v2-7-tvm.tar
#
# Recall that the ``.tar`` model file includes a C++ library, a description of
# the Relay model, and the parameters for the model. TVMC includes the TVM
# runtime, which can load the model and make predictions against input. When
# running the above command, TVMC outputs a new file, ``predictions.npz``, that
# contains the model output tensors in NumPy format.
#
# In this example, we are running the model on the same machine that we used
# for compilation. In some cases we might want to run it remotely via an RPC
# Tracker. To read more about these options please check ``tvmc run --help``.

################################################################################
# Output Post-Processing
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As previously mentioned, each model will have its own particular way of
# providing output tensors.
#
# In our case, we need to run some post-processing to render the outputs from
# ResNet-50 v2 into a more human-readable form, using the lookup-table provided
# for the model.
#
# The script below shows an example of the post-processing to extract labels
# from the output of our compiled module.
#
# .. code-block:: python
#     :caption: postprocess.py
#     :name: postprocess.py
#
#     #!python ./postprocess.py
#     import os.path
#     import numpy as np
#
#     from scipy.special import softmax
#
#     from tvm.contrib.download import download_testdata
#
#     # Download a list of labels
#     labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
#     labels_path = download_testdata(labels_url, "synset.txt", module="data")
#
#     with open(labels_path, "r") as f:
#         labels = [l.rstrip() for l in f]
#
#     output_file = "predictions.npz"
#
#     # Open the output and read the output tensor
#     if os.path.exists(output_file):
#         with np.load(output_file) as data:
#             scores = softmax(data["output_0"])
#             scores = np.squeeze(scores)
#             ranks = np.argsort(scores)[::-1]
#
#             for rank in ranks[0:5]:
#                 print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
#
# Running this script should produce the following output:
#
# .. code-block:: bash
#
#     python postprocess.py
#     # class='n02123045 tabby, tabby cat' with probability=0.610553
#     # class='n02123159 tiger cat' with probability=0.367179
#     # class='n02124075 Egyptian cat' with probability=0.019365
#     # class='n02129604 tiger, Panthera tigris' with probability=0.001273
#     # class='n04040759 radiator' with probability=0.000261
#
# Try replacing the cat image with other images, and see what sort of
# predictions the ResNet model makes.

################################################################################
# Automatically Tuning the ResNet Model
# -------------------------------------
#
# The previous model was compiled to work on the TVM runtime, but did not
# include any platform specific optimization. In this section, we will show you
# how to build an optimized model using TVMC to target your working platform.
#
# In some cases, we might not get the expected performance when running
# inferences using our compiled module.  In cases like this, we can make use of
# the auto-tuner, to find a better configuration for our model and get a boost
# in performance. Tuning in TVM refers to the process by which a model is
# optimized to run faster on a given target. This differs from training or
# fine-tuning in that it does not affect the accuracy of the model, but only
# the runtime performance. As part of the tuning process, TVM will try running
# many different operator implementation variants to see which perform best.
# The results of these runs are stored in a tuning records file, which is
# ultimately the output of the ``tune`` subcommand.
#
# In the simplest form, tuning requires you to provide three things:
#
# - the target specification of the device you intend to run this model on
# - the path to an output file in which the tuning records will be stored, and
#   finally
# - a path to the model to be tuned.
#
# The example below demonstrates how that works in practice:
#
# .. code-block:: bash
#
#     # The default search algorithm requires xgboost, see below for further
#     # details on tuning search algorithms
#     pip install xgboost
#
#     tvmc tune \
#     --target "llvm" \
#     --output resnet50-v2-7-autotuner_records.json \
#     resnet50-v2-7.onnx
#
# In this example, you will see better results if you indicate a more specific
# target for the ``--target`` flag.  For example, on an Intel i7 processor you
# could use ``--target llvm -mcpu=skylake``. For this tuning example, we are
# tuning locally on the CPU using LLVM as the compiler for the specified
# achitecture.
#
# TVMC will perform a search against the parameter space for the model, trying
# out different configurations for operators and choosing the one that runs
# fastest on your platform. Although this is a guided search based on the CPU
# and model operations, it can still take several hours to complete the search.
# The output of this search will be saved to the
# ``resnet50-v2-7-autotuner_records.json`` file, which will later be used to
# compile an optimized model.
#
# .. admonition:: Defining the Tuning Search Algorithm
#
#   By default this search is guided using an ``XGBoost Grid`` algorithm.
#   Depending on your model complexity and amount of time avilable, you might
#   want to choose a different algorithm. A full list is available by
#   consulting ``tvmc tune --help``.
#
# The output will look something like this for a consumer-level Skylake CPU:
#
# .. code-block:: bash
#
#   tvmc tune \
#   --target "llvm -mcpu=broadwell" \
#   --output resnet50-v2-7-autotuner_records.json \
#   resnet50-v2-7.onnx
#   # [Task  1/24]  Current/Best:    9.65/  23.16 GFLOPS | Progress: (60/1000) | 130.74 s Done.
#   # [Task  1/24]  Current/Best:    3.56/  23.16 GFLOPS | Progress: (192/1000) | 381.32 s Done.
#   # [Task  2/24]  Current/Best:   13.13/  58.61 GFLOPS | Progress: (960/1000) | 1190.59 s Done.
#   # [Task  3/24]  Current/Best:   31.93/  59.52 GFLOPS | Progress: (800/1000) | 727.85 s Done.
#   # [Task  4/24]  Current/Best:   16.42/  57.80 GFLOPS | Progress: (960/1000) | 559.74 s Done.
#   # [Task  5/24]  Current/Best:   12.42/  57.92 GFLOPS | Progress: (800/1000) | 766.63 s Done.
#   # [Task  6/24]  Current/Best:   20.66/  59.25 GFLOPS | Progress: (1000/1000) | 673.61 s Done.
#   # [Task  7/24]  Current/Best:   15.48/  59.60 GFLOPS | Progress: (1000/1000) | 953.04 s Done.
#   # [Task  8/24]  Current/Best:   31.97/  59.33 GFLOPS | Progress: (972/1000) | 559.57 s Done.
#   # [Task  9/24]  Current/Best:   34.14/  60.09 GFLOPS | Progress: (1000/1000) | 479.32 s Done.
#   # [Task 10/24]  Current/Best:   12.53/  58.97 GFLOPS | Progress: (972/1000) | 642.34 s Done.
#   # [Task 11/24]  Current/Best:   30.94/  58.47 GFLOPS | Progress: (1000/1000) | 648.26 s Done.
#   # [Task 12/24]  Current/Best:   23.66/  58.63 GFLOPS | Progress: (1000/1000) | 851.59 s Done.
#   # [Task 13/24]  Current/Best:   25.44/  59.76 GFLOPS | Progress: (1000/1000) | 534.58 s Done.
#   # [Task 14/24]  Current/Best:   26.83/  58.51 GFLOPS | Progress: (1000/1000) | 491.67 s Done.
#   # [Task 15/24]  Current/Best:   33.64/  58.55 GFLOPS | Progress: (1000/1000) | 529.85 s Done.
#   # [Task 16/24]  Current/Best:   14.93/  57.94 GFLOPS | Progress: (1000/1000) | 645.55 s Done.
#   # [Task 17/24]  Current/Best:   28.70/  58.19 GFLOPS | Progress: (1000/1000) | 756.88 s Done.
#   # [Task 18/24]  Current/Best:   19.01/  60.43 GFLOPS | Progress: (980/1000) | 514.69 s Done.
#   # [Task 19/24]  Current/Best:   14.61/  57.30 GFLOPS | Progress: (1000/1000) | 614.44 s Done.
#   # [Task 20/24]  Current/Best:   10.47/  57.68 GFLOPS | Progress: (980/1000) | 479.80 s Done.
#   # [Task 21/24]  Current/Best:   34.37/  58.28 GFLOPS | Progress: (308/1000) | 225.37 s Done.
#   # [Task 22/24]  Current/Best:   15.75/  57.71 GFLOPS | Progress: (1000/1000) | 1024.05 s Done.
#   # [Task 23/24]  Current/Best:   23.23/  58.92 GFLOPS | Progress: (1000/1000) | 999.34 s Done.
#   # [Task 24/24]  Current/Best:   17.27/  55.25 GFLOPS | Progress: (1000/1000) | 1428.74 s Done.
#
# Tuning sessions can take a long time, so ``tvmc tune`` offers many options to customize your tuning
# process, in terms of number of repetitions (``--repeat`` and ``--number``, for example), the tuning
# algorithm to be used, and so on. Check ``tvmc tune --help`` for more information.
#
# In some situations it might be a good idea, to only tune specific tasks (i.e. the most relevant ones)
# to waste less time tuning simpler workworloads. The flag `--task` offers versatile options to limt
# the tasks used for tuning, e.g. `--task 20,22` or `--task 16-`. All available tasks can be printed
# using `--task list`.
#

################################################################################
# Compiling an Optimized Model with Tuning Data
# ----------------------------------------------
#
# As an output of the tuning process above, we obtained the tuning records
# stored in ``resnet50-v2-7-autotuner_records.json``. This file can be used in
# two ways:
#
# - As input to further tuning (via ``tvmc tune --tuning-records``).
# - As input to the compiler
#
# The compiler will use the results to generate high performance code for the
# model on your specified target. To do that we can use ``tvmc compile
# --tuning-records``. Check ``tvmc compile --help`` for more information.
#
# Now that tuning data for the model has been collected, we can re-compile the
# model using optimized operators to speed up our computations.
#
# .. code-block:: bash
#
#   tvmc compile \
#   --target "llvm" \
#   --tuning-records resnet50-v2-7-autotuner_records.json  \
#   --output resnet50-v2-7-tvm_autotuned.tar \
#   resnet50-v2-7.onnx
#
# Verify that the optimized model runs and produces the same results:
#
# .. code-block:: bash
#
#   tvmc run \
#   --inputs imagenet_cat.npz \
#   --output predictions.npz \
#   resnet50-v2-7-tvm_autotuned.tar
#
#   python postprocess.py
#
# Verifying that the predictions are the same:
#
# .. code-block:: bash
#
#   # class='n02123045 tabby, tabby cat' with probability=0.610550
#   # class='n02123159 tiger cat' with probability=0.367181
#   # class='n02124075 Egyptian cat' with probability=0.019365
#   # class='n02129604 tiger, Panthera tigris' with probability=0.001273
#   # class='n04040759 radiator' with probability=0.000261

################################################################################
# Comparing the Tuned and Untuned Models
# --------------------------------------
#
# TVMC gives you tools for basic performance benchmarking between the models.
# You can specify a number of repetitions and that TVMC report on the model run
# time (independent of runtime startup). We can get a rough idea of how much
# tuning has improved the model performance. For example, on a test Intel i7
# system, we see that the tuned model runs 47% faster than the untuned model:
#
# .. code-block:: bash
#
#   tvmc run \
#   --inputs imagenet_cat.npz \
#   --output predictions.npz  \
#   --print-time \
#   --repeat 100 \
#   resnet50-v2-7-tvm_autotuned.tar
#
#   # Execution time summary:
#   # mean (ms)   max (ms)    min (ms)    std (ms)
#   #     92.19     115.73       89.85        3.15
#
#   tvmc run \
#   --inputs imagenet_cat.npz \
#   --output predictions.npz  \
#   --print-time \
#   --repeat 100 \
#   resnet50-v2-7-tvm.tar
#
#   # Execution time summary:
#   # mean (ms)   max (ms)    min (ms)    std (ms)
#   #    193.32     219.97      185.04        7.11
#


################################################################################
# Final Remarks
# -------------
#
# In this tutorial, we presented TVMC, a command line driver for TVM. We
# demonstrated how to compile, run, and tune a model. We also discussed the
# need for pre and post-processing of inputs and outputs. After the tuning
# process, we demonstrated how to compare the performance of the unoptimized
# and optimize models.
#
# Here we presented a simple example using ResNet-50 v2 locally. However, TVMC
# supports many more features including cross-compilation, remote execution and
# profiling/benchmarking.
#
# To see what other options are available, please have a look at ``tvmc
# --help``.
#
# In the `next tutorial <tvmc_python>`, we introduce the Python interface to TVM,
# and in the tutorial after that,
# `Compiling and Optimizing a Model with the Python Interface <autotvm_relay_x86>`,
# we will cover the same compilation and optimization steps using the Python
# interface.
