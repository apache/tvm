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
Compiling and Optimizing a Model with the Python Interface (AutoTVM)
====================================================================
**Author**:
`Chris Hoge <https://github.com/hogepodge>`_

In the `TVMC Tutorial <tvmc_command_line_driver>`_, we covered how to compile, run, and tune a
pre-trained vision model, ResNet-50 v2 using the command line interface for
TVM, TVMC. TVM is more that just a command-line tool though, it is an
optimizing framework with APIs available for a number of different languages
that gives you tremendous flexibility in working with machine learning models.

In this tutorial we will cover the same ground we did with TVMC, but show how
it is done with the Python API. Upon completion of this section, we will have
used the Python API for TVM to accomplish the following tasks:

* Compile a pre-trained ResNet-50 v2 model for the TVM runtime.
* Run a real image through the compiled model, and interpret the output and model
  performance.
* Tune the model that model on a CPU using TVM.
* Re-compile an optimized model using the tuning data collected by TVM.
* Run the image through the optimized model, and compare the output and model
  performance.

The goal of this section is to give you an overview of TVM's capabilites and
how to use them through the Python API.
"""


################################################################################
# TVM is a deep learning compiler framework, with a number of different modules
# available for working with deep learning models and operators. In this
# tutorial we will work through how to load, compile, and optimize a model
# using the Python API.
#
# We begin by importing a number of dependencies, including ``onnx`` for
# loading and converting the model, helper utilities for downloading test data,
# the Python Image Library for working with the image data, ``numpy`` for pre
# and post-processing of the image data, the TVM Relay framework, and the TVM
# Graph Executor.

import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

################################################################################
# Downloading and Loading the ONNX Model
# --------------------------------------
#
# For this tutorial, we will be working with ResNet-50 v2. ResNet-50 is a
# convolutional neural network that is 50 layers deep and designed to classify
# images. The model we will be using has been pre-trained on more than a
# million images with 1000 different classifications. The network has an input
# image size of 224x224. If you are interested exploring more of how the
# ResNet-50 model is structured, we recommend downloading
# `Netron <https://netron.app>`_, a freely available ML model viewer.
#
# TVM provides a helper library to download pre-trained models. By providing a
# model URL, file name, and model type through the module, TVM will download
# the model and save it to disk. For the instance of an ONNX model, you can
# then load it into memory using the ONNX runtime.
#
# .. admonition:: Working with Other Model Formats
#
#   TVM supports many popular model formats. A list can be found in the
#   :ref:`Compile Deep Learning Models <tutorial-frontend>` section of the TVM
#   Documentation.

model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# Seed numpy's RNG to get consistent results
np.random.seed(0)

################################################################################
# Downloading, Preprocessing, and Loading the Test Image
# ------------------------------------------------------
#
# Each model is particular when it comes to expected tensor shapes, formats and
# data types. For this reason, most models require some pre and
# post-processing, to ensure the input is valid and to interpret the output.
# TVMC has adopted NumPy's ``.npz`` format for both input and output data.
#
# As input for this tutorial, we will use the image of a cat, but you can feel
# free to substitute this image for any of your choosing.
#
# .. image:: https://s3.amazonaws.com/model-server/inputs/kitten.jpg
#    :height: 224px
#    :width: 224px
#    :align: center
#
# Download the image data, then convert it to a numpy array to use as an input to the model.

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

###############################################################################
# Compile the Model With Relay
# ----------------------------
#
# The next step is to compile the ResNet model. We begin by importing the model
# to relay using the `from_onnx` importer. We then build the model, with
# standard optimizations, into a TVM library.  Finally, we create a TVM graph
# runtime module from the library.

target = "llvm"

######################################################################
# .. admonition:: Defining the Correct Target
#
#   Specifying the correct target can have a huge impact on the performance of
#   the compiled module, as it can take advantage of hardware features
#   available on the target. For more information, please refer to
#   :ref:`Auto-tuning a convolutional network for x86 CPU <tune_relay_x86>`.
#   We recommend identifying which CPU you are running, along with optional
#   features, and set the target appropriately. For example, for some
#   processors ``target = "llvm -mcpu=skylake"``, or ``target = "llvm
#   -mcpu=skylake-avx512"`` for processors with the AVX-512 vector instruction
#   set.
#

# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

######################################################################
# Execute on the TVM Runtime
# --------------------------
# Now that we've compiled the model, we can use the TVM runtime to make
# predictions with it. To use TVM to run the model and make predictions, we
# need two things:
#
# - The compiled model, which we just produced.
# - Valid input to the model to make predictions on.

dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

################################################################################
# Collect Basic Performance Data
# ------------------------------
# We want to collect some basic performance data associated with this
# unoptimized model and compare it to a tuned model later. To help account for
# CPU noise, we run the computation in multiple batches in multiple
# repetitions, then gather some basis statistics on the mean, median, and
# standard deviation.
import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)

################################################################################
# Postprocess the output
# ----------------------
#
# As previously mentioned, each model will have its own particular way of
# providing output tensors.
#
# In our case, we need to run some post-processing to render the outputs from
# ResNet-50 v2 into a more human-readable form, using the lookup-table provided
# for the model.

from scipy.special import softmax

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

################################################################################
# This should produce the following output:
#
# .. code-block:: bash
#
#     # class='n02123045 tabby, tabby cat' with probability=0.610553
#     # class='n02123159 tiger cat' with probability=0.367179
#     # class='n02124075 Egyptian cat' with probability=0.019365
#     # class='n02129604 tiger, Panthera tigris' with probability=0.001273
#     # class='n04040759 radiator' with probability=0.000261

################################################################################
# Tune the model
# --------------
# The previous model was compiled to work on the TVM runtime, but did not
# include any platform specific optimization. In this section, we will show you
# how to build an optimized model using TVM to target your working platform.
#
# In some cases, we might not get the expected performance when running
# inferences using our compiled module. In cases like this, we can make use of
# the auto-tuner, to find a better configuration for our model and get a boost
# in performance. Tuning in TVM refers to the process by which a model is
# optimized to run faster on a given target. This differs from training or
# fine-tuning in that it does not affect the accuracy of the model, but only
# the runtime performance. As part of the tuning process, TVM will try running
# many different operator implementation variants to see which perform best.
# The results of these runs are stored in a tuning records file.
#
# In the simplest form, tuning requires you to provide three things:
#
# - the target specification of the device you intend to run this model on
# - the path to an output file in which the tuning records will be stored
# - a path to the model to be tuned.
#

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

################################################################################
# Set up some basic parameters for the runner. The runner takes compiled code
# that is generated with a specific set of parameters and measures the
# performance of it. ``number`` specifies the number of different
# configurations that we will test, while ``repeat`` specifies how many
# measurements we will take of each configuration. ``min_repeat_ms`` is a value
# that specifies how long need to run configuration test. If the number of
# repeats falls under this time, it will be increased. This option is necessary
# for accurate tuning on GPUs, and is not required for CPU tuning. Setting this
# value to 0 disables it. The ``timeout`` places an upper limit on how long to
# run training code for each tested configuration.

number = 10
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

################################################################################
# Create a simple structure for holding tuning options. We use an XGBoost
# algorithim for guiding the search. For a production job, you will want to set
# the number of trials to be larger than the value of 20 used here. For CPU we
# recommend 1500, for GPU 3000-4000. The number of trials required can depend
# on the particular model and processor, so it's worth spending some time
# evaluating performance across a range of values to find the best balance
# between tuning time and model optimization. Because running tuning is time
# intensive we set number of trials to 10, but do not recommend a value this
# small. The ``early_stopping`` parameter is the minimum number of trails to
# run before a condition that stops the search early can be applied. The
# measure option indicates where trial code will be built, and where it will be
# run. In this case, we're using the ``LocalRunner`` we just created and a
# ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
# the tuning data to.

tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning.json",
}

################################################################################
# .. admonition:: Defining the Tuning Search Algorithm
#
#   By default this search is guided using an `XGBoost Grid` algorithm.
#   Depending on your model complexity and amount of time available, you might
#   want to choose a different algorithm.


################################################################################
# .. admonition:: Setting Tuning Parameters
#
#   In this example, in the interest of time, we set the number of trials and
#   early stopping to 20 and 100. You will likely see more performance improvements if
#   you set these values to be higher but this comes at the expense of time
#   spent tuning. The number of trials required for convergence will vary
#   depending on the specifics of the model and the target platform.

# begin by extracting the tasks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

    # choose tuner
    tuner = "xgb"

    # create tuner
    if tuner == "xgb":
        tuner_obj = XGBTuner(task, loss_type="reg")
    elif tuner == "xgb_knob":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
    elif tuner == "xgb_itervar":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
    elif tuner == "xgb_curve":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
    elif tuner == "xgb_rank":
        tuner_obj = XGBTuner(task, loss_type="rank")
    elif tuner == "xgb_rank_knob":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
    elif tuner == "xgb_rank_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
    elif tuner == "xgb_rank_curve":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
    elif tuner == "xgb_rank_binary":
        tuner_obj = XGBTuner(task, loss_type="rank-binary")
    elif tuner == "xgb_rank_binary_knob":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
    elif tuner == "xgb_rank_binary_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
    elif tuner == "xgb_rank_binary_curve":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
    elif tuner == "ga":
        tuner_obj = GATuner(task, pop_size=50)
    elif tuner == "random":
        tuner_obj = RandomTuner(task)
    elif tuner == "gridsearch":
        tuner_obj = GridSearchTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)

    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )

################################################################################
# The output from this tuning process will look something like this:
#
# .. code-block:: bash
#
#   # [Task  1/24]  Current/Best:   10.71/  21.08 GFLOPS | Progress: (60/1000) | 111.77 s Done.
#   # [Task  1/24]  Current/Best:    9.32/  24.18 GFLOPS | Progress: (192/1000) | 365.02 s Done.
#   # [Task  2/24]  Current/Best:   22.39/ 177.59 GFLOPS | Progress: (960/1000) | 976.17 s Done.
#   # [Task  3/24]  Current/Best:   32.03/ 153.34 GFLOPS | Progress: (800/1000) | 776.84 s Done.
#   # [Task  4/24]  Current/Best:   11.96/ 156.49 GFLOPS | Progress: (960/1000) | 632.26 s Done.
#   # [Task  5/24]  Current/Best:   23.75/ 130.78 GFLOPS | Progress: (800/1000) | 739.29 s Done.
#   # [Task  6/24]  Current/Best:   38.29/ 198.31 GFLOPS | Progress: (1000/1000) | 624.51 s Done.
#   # [Task  7/24]  Current/Best:    4.31/ 210.78 GFLOPS | Progress: (1000/1000) | 701.03 s Done.
#   # [Task  8/24]  Current/Best:   50.25/ 185.35 GFLOPS | Progress: (972/1000) | 538.55 s Done.
#   # [Task  9/24]  Current/Best:   50.19/ 194.42 GFLOPS | Progress: (1000/1000) | 487.30 s Done.
#   # [Task 10/24]  Current/Best:   12.90/ 172.60 GFLOPS | Progress: (972/1000) | 607.32 s Done.
#   # [Task 11/24]  Current/Best:   62.71/ 203.46 GFLOPS | Progress: (1000/1000) | 581.92 s Done.
#   # [Task 12/24]  Current/Best:   36.79/ 224.71 GFLOPS | Progress: (1000/1000) | 675.13 s Done.
#   # [Task 13/24]  Current/Best:    7.76/ 219.72 GFLOPS | Progress: (1000/1000) | 519.06 s Done.
#   # [Task 14/24]  Current/Best:   12.26/ 202.42 GFLOPS | Progress: (1000/1000) | 514.30 s Done.
#   # [Task 15/24]  Current/Best:   31.59/ 197.61 GFLOPS | Progress: (1000/1000) | 558.54 s Done.
#   # [Task 16/24]  Current/Best:   31.63/ 206.08 GFLOPS | Progress: (1000/1000) | 708.36 s Done.
#   # [Task 17/24]  Current/Best:   41.18/ 204.45 GFLOPS | Progress: (1000/1000) | 736.08 s Done.
#   # [Task 18/24]  Current/Best:   15.85/ 222.38 GFLOPS | Progress: (980/1000) | 516.73 s Done.
#   # [Task 19/24]  Current/Best:   15.78/ 203.41 GFLOPS | Progress: (1000/1000) | 587.13 s Done.
#   # [Task 20/24]  Current/Best:   30.47/ 205.92 GFLOPS | Progress: (980/1000) | 471.00 s Done.
#   # [Task 21/24]  Current/Best:   46.91/ 227.99 GFLOPS | Progress: (308/1000) | 219.18 s Done.
#   # [Task 22/24]  Current/Best:   13.33/ 207.66 GFLOPS | Progress: (1000/1000) | 761.74 s Done.
#   # [Task 23/24]  Current/Best:   53.29/ 192.98 GFLOPS | Progress: (1000/1000) | 799.90 s Done.
#   # [Task 24/24]  Current/Best:   25.03/ 146.14 GFLOPS | Progress: (1000/1000) | 1112.55 s Done.

################################################################################
# Compiling an Optimized Model with Tuning Data
# ----------------------------------------------
#
# As an output of the tuning process above, we obtained the tuning records
# stored in ``resnet-50-v2-autotuning.json``. The compiler will use the results to
# generate high performance code for the model on your specified target.
#
# Now that tuning data for the model has been collected, we can re-compile the
# model using optimized operators to speed up our computations.

with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Verify that the optimized model runs and produces the same results:

dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

################################################################################
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
# We want to collect some basic performance data associated with this optimized
# model to compare it to the unoptimized model. Depending on your underlying
# hardware, number of iterations, and other factors, you should see a performance
# improvement in comparing the optimized model to the unoptimized model.

import timeit

timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))

################################################################################
# Final Remarks
# -------------
#
# In this tutorial, we gave a short example of how to use the TVM Python API
# to compile, run, and tune a model. We also discussed the need for pre and
# post-processing of inputs and outputs. After the tuning process, we
# demonstrated how to compare the performance of the unoptimized and optimize
# models.
#
# Here we presented a simple example using ResNet-50 v2 locally. However, TVM
# supports many more features including cross-compilation, remote execution and
# profiling/benchmarking.
