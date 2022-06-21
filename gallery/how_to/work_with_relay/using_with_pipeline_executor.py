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
Using Pipeline Executor in Relay
=================================
**Author**: `Hua Jiang <https://https://github.com/huajsj>`_

This is a short tutorial on how to use the "Pipeline Executor" with Relay.
"""
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing
import time

#######################################################################
# Create a simple network, this network can be a pre-trained model too.
# ---------------------------------------------------------------------
# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.
def get_network():
    out_channels = 16
    batch_size = 1
    data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
    weight = relay.var("weight")
    second_weight = relay.var("second_weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")
    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.conv2d(
        data=simple_net,
        weight=second_weight,
        kernel_size=(3, 3),
        channels=out_channels,
        padding=(1, 1),
    )
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    data_shape = (batch_size, 3, 224, 224)
    net, params = testing.create_workload(simple_net)
    return net, params, data_shape


net, params, data_shape = get_network()
###########################################
# Splitting the network into two subgraphs.
# -----------------------------------------
# We use an testing linear graph splitting function as a example. User also can create their
# own splitting function logic.
import inspect
import os

test_path = os.path.dirname(inspect.getfile(lambda: None))
os.sys.path.append(os.path.join(test_path, "../../../tests/python/relay"))
from test_pipeline_executor import graph_split

###########################################
# Splitting the network into two subgraphs.
split_config = [{"op_name": "nn.relu", "op_index": 0}]
subgraphs = graph_split(net["main"], split_config, params)
###########################################################
# The generated subgraphs should look something like below.

"""
#subgraphs[0])

 def @main(%data: Tensor[(1, 3, 224, 224), float32]) {
  %0 = nn.conv2d(%data, meta[relay.Constant][0] /* ty=Tensor[(16, 3, 3, 3), float32] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 224, 224), float32] */;
  %1 = nn.batch_norm(%0, meta[relay.Constant][1] /* ty=Tensor[(16), float32] */, meta[relay.Constant][2] /* ty=Tensor[(16), float32]*/, meta[relay.Constant][3] /* ty=Tensor[(16), float32] */, meta[relay.Constant][4] /* ty=Tensor[(16), float32] */) /* ty=(Tensor[(1,16, 224, 224), float32], Tensor[(16), float32], Tensor[(16), float32]) */;
  %2 = %1.0;
  nn.relu(%2) /* ty=Tensor[(1, 16, 224, 224), float32] */
 }

peline-tutorial

#subgraphs[1]

 def @main(%data_n_0: Tensor[(1, 16, 224, 224), float32]) {
  nn.conv2d(%data_n_0, meta[relay.Constant][0] /* ty=Tensor[(16, 16, 3, 3), float32] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 224, 224), float32] */
 }
"""

###########################################################
# Run the two subgraphs in pipeline with pipeline executor.
# ---------------------------------------------------------
# Define a function to do all the codegen and pipeline executor works.
# To run pipeline executor with dnnl, USE_PIPELINE_EXECUTOR need to get set as ON.
# and the 'USE_DNNL_CODEGEN' should set as ON in config.cmake and installing MKL-DNN.
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build

#########################################
# Create subgraph pipeline configuration.
# Associate the subgraph module with a target.
# Using BYOC to set the codegen of the second subgraph module.
# To use dnnl the 'USE_DNNL_CODEGEN' should set as ON in config.cmake and installing MKL-DNN.
mod0, mod1 = subgraphs[0], subgraphs[1]
mod0 = relay.transform.AnnotateTarget(["dnnl"])(mod0)
mod0 = relay.transform.MergeCompilerRegions()(mod0)
mod0 = relay.transform.PartitionGraph()(mod0)
#################################################
# Get the pipeline executor configuration object.
pipe_config = pipeline_executor_build.PipelineConfig()
###########################################################################
# Set the compile target of the second subgraph module for example as LLVM.
pipe_config[mod0].target = "llvm"
pipe_config[mod0].dev = tvm.cpu(0)
###############################################################################
# Set the cpu afinity for control flow, for example using cpu 0 for control flow.
pipe_config[mod1].cpu_affinity = "0"
##############################################################
# Set the compile target of the second subgraph module as LLVM.
pipe_config[mod1].target = "llvm"
pipe_config[mod1].dev = tvm.cpu(0)
#################################################################################
# Set the cpu afinity for control flow, for example using cpu 1 for control flow.
pipe_config[mod1].cpu_affinity = "1"
pipe_config["input"]["data"].connect(pipe_config[mod0]["input"]["data"])
pipe_config[mod0]["output"][0].connect(pipe_config[mod1]["input"]["data_n_0"])
pipe_config[mod1]["output"]["0"].connect(pipe_config["output"][0])
######################################
# The pipeline configuration as below.
"""
print(pipe_config)
 Inputs
  |data: mod0:data

 output
  |output(0) : mod1.output(0)

 connections
  |mod0.output(0)-> mod1.data_n_0
"""
##############################
# Build the pipeline executor.
# ----------------------------
with tvm.transform.PassContext(opt_level=3):
    pipeline_mod_factory = pipeline_executor_build.build(pipe_config)
###############################################
# Export the parameter configuration to a file.
directory_path = tvm.contrib.utils.tempdir().temp_dir
#############################################
# If the directory does not exist, create it.
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
config_file_name = pipeline_mod_factory.export_library(directory_path)
################################################################
# Use the load function to create and initialize PipelineModule.
# --------------------------------------------------------------
pipeline_module = pipeline_executor.PipelineModule.load_library(config_file_name)

############################
# Run the pipeline executor.
# --------------------------
# Allocated a input data.
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
pipeline_module.set_input("data", tvm.nd.array(data))
##########################################################################
# Run the two subgraph in pipeline mode and get the output asynchronously.
pipeline_module.run()
outputs = []
while not outputs:
    outputs = pipeline_module.get_output()
    time.sleep(0.001)
######################################
# Use graph_executor for verification.
# ------------------------------------
# Run these two subgraphs in sequence with graph_executor to get the output.
target = "llvm"
dev = tvm.device(target, 0)
lib0 = relay.build_module.build(mod0, target, params=params)
lib1 = relay.build_module.build(mod1, target, params=params)
module0 = runtime.GraphModule(lib0["default"](dev))
module1 = runtime.GraphModule(lib1["default"](dev))
module0.set_input("data", data)
module0.run()
out_shape = (1, 16, 224, 224)
out = module0.get_output(0, tvm.nd.empty(out_shape))
module1.set_input("data_n_0", out)
module1.run()
out = module1.get_output(0, tvm.nd.empty(out_shape))
####################
# Verify the result.
tvm.testing.assert_allclose(outputs[0].numpy(), out.numpy())
