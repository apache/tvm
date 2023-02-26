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
**Author**: `Hua Jiang <https://github.com/huajsj>`_

This is a short tutorial on how to use "Pipeline Executor" with Relay.
"""
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib.cutlass import finalize_modules

img_size = 8
#######################################################################
# Create a simple network, this network can be a pre-trained model too.
# ---------------------------------------------------------------------
# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, dense, and ReLU activation.
def get_network():
    out_channels = 16
    batch_size = 1
    data = relay.var("data", relay.TensorType((batch_size, 3, img_size, img_size), "float16"))
    dense_weight = relay.var(
        "dweight", relay.TensorType((batch_size, 16 * img_size * img_size), "float16")
    )
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")
    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.batch_flatten(simple_net)
    simple_net = relay.nn.dense(simple_net, dense_weight)
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = testing.create_workload(simple_net)
    return net, params, data_shape


net, params, data_shape = get_network()
###########################################
# Splitting the network into two subgraphs.
# -----------------------------------------
# This function called 'graph_split' from a unit test is just an example. User can create a customized logic
# to split the graph.
import inspect
import os

tutorial_dir = os.path.dirname(inspect.getfile(lambda: None))
os.sys.path.append(os.path.join(tutorial_dir, "../../../tests/python/relay"))
from test_pipeline_executor import graph_split

###########################################
# Splitting the network into two subgraphs.
split_config = [{"op_name": "nn.relu", "op_index": 0}]
subgraphs = graph_split(net["main"], split_config, params)
###########################################################
# The generated subgraphs should look something like below.

"""
#subgraphs[0])

 def @main(%data: Tensor[(1, 3, img_size, img_size), float16]) {
  %0 = nn.conv2d(%data, meta[relay.Constant][0] /* ty=Tensor[(16, 3, 3, 3), float16] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, img_size, img_size), float16] */;
  %1 = nn.batch_norm(%0, meta[relay.Constant][1] /* ty=Tensor[(16), float16] */, meta[relay.Constant][2] /* ty=Tensor[(16), float16]*/, meta[relay.Constant][3] /* ty=Tensor[(16), float16] */, meta[relay.Constant][4] /* ty=Tensor[(16), float16] */) /* ty=(Tensor[(1,16, img_size, img_size), float16], Tensor[(16), float16], Tensor[(16), float16]) */;
  %2 = %1.0;
  nn.relu(%2) /* ty=Tensor[(1, 16, img_size, img_size), float16] */
 }

#subgraphs[1]

 def @main(%data_n_0: Tensor[(1, 16, 8, 8), float16] /* ty=Tensor[(1, 16, 8, 8), float16] */) {
  %0 = nn.batch_flatten(%data_n_0) /* ty=Tensor[(1, 1024), float16] */;
  nn.dense(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 1024), float16] */, units=None) /* ty=Tensor[(1, 1), float16] */
 }

"""


#########################################
# Build the subgraph with cutlass target.
# ---------------------------------------

cutlass = tvm.target.Target(
    {
        "kind": "cutlass",
        "sm": int(tvm.target.Target("cuda").arch.split("_")[1]),
        "use_3xtf32": True,
        "split_k_slices": [1],
        "profile_all_alignments": False,
        "find_first_valid": True,
        "use_multiprocessing": True,
        "use_fast_math": False,
        "tmp_dir": "./tmp",
    },
    host=tvm.target.Target("llvm"),
)


def cutlass_build(mod, target, params=None, target_host=None, mod_name="default"):
    target = [target, cutlass]
    lib = relay.build_module.build(
        mod, target=target, params=params, target_host=target_host, mod_name=mod_name
    )
    return lib


###########################################################
# Run the two subgraphs in pipeline with pipeline executor.
# ---------------------------------------------------------
# Set 'USE_PIPELINE_EXECUTOR' as ON, and set USE_CUTLASS' as ON  in cmake.
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build

#########################################
# Create subgraph pipeline configuration.
# Associate a subgraph module with a target.
# Use CUTLASS BYOC to build the second subgraph module.
mod0, mod1 = subgraphs[0], subgraphs[1]
# Use cutlass as the codegen.
mod1 = partition_for_cutlass(mod1)
#################################################
# Get the pipeline executor configuration object.
pipe_config = pipeline_executor_build.PipelineConfig()
###########################################################################
# Set the compile target of the subgraph module.
pipe_config[mod0].target = "llvm"
pipe_config[mod0].dev = tvm.cpu(0)
##############################################################
# Set the compile target of the second subgraph module as cuda.
pipe_config[mod1].target = "cuda"
pipe_config[mod1].dev = tvm.device("cuda", 0)
pipe_config[mod1].build_func = cutlass_build
pipe_config[mod1].export_cc = "nvcc"
# Create the pipeline by connecting the subgraph modules.
# The global input will be forwarded to the input interface of the first module named mod0
pipe_config["input"]["data"].connect(pipe_config[mod0]["input"]["data"])
# The first output of mod0 will be forwarded to the input interface of mod1
pipe_config[mod0]["output"][0].connect(pipe_config[mod1]["input"]["data_n_0"])
# The first output of mod1 will be the first global output.
pipe_config[mod1]["output"][0].connect(pipe_config["output"][0])
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
os.makedirs(directory_path, exist_ok=True)
config_file_name = pipeline_mod_factory.export_library(directory_path)
################################################################
# Use the load function to create and initialize PipelineModule.
# --------------------------------------------------------------
pipeline_module = pipeline_executor.PipelineModule.load_library(config_file_name)

############################
# Run the pipeline executor.
# --------------------------
# Allocate input data.
data = np.random.uniform(-1, 1, size=data_shape).astype("float16")
pipeline_module.set_input("data", tvm.nd.array(data))
##########################################################################
# Run the two subgraph in the pipeline mode to get the output asynchronously
# or synchronously. In the following example, it is synchronous.
pipeline_module.run()
outputs = pipeline_module.get_output()
######################################
# Use graph_executor for verification.
# ------------------------------------
# Run these two subgraphs in sequence with graph_executor to get the output.
target = "llvm"
dev0 = tvm.device(target, 0)
lib0 = relay.build_module.build(mod0, target, params=params)
module0 = runtime.GraphModule(lib0["default"](dev0))
cuda = tvm.target.Target("cuda", host=tvm.target.Target("llvm"))
lib1 = relay.build_module.build(mod1, [cuda, cutlass], params=params)
lib1 = finalize_modules(lib1, "compile.so", "./tmp")

dev1 = tvm.device("cuda", 0)

module1 = runtime.GraphModule(lib1["default"](dev1))

module0.set_input("data", data)
module0.run()
out_shape = (1, 16, img_size, img_size)
out = module0.get_output(0, tvm.nd.empty(out_shape, "float16"))
module1.set_input("data_n_0", out)
module1.run()
out_shape = (1, 1)
out = module1.get_output(0, tvm.nd.empty(out_shape, "float16"))
####################
# Verify the result.
tvm.testing.assert_allclose(outputs[0].numpy(), out.numpy())
