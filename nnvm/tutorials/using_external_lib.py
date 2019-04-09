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
Using External Libraries in NNVM
================================
**Author**: `Masahiro Masuda <https://github.com/masahi>`_

This is a short tutorial on how to use external libraries such as cuDNN, or cuBLAS with NNVM.

NNVM uses TVM internally to generate target specific code. For example, with cuda backend TVM generates cuda kernels for all layers in the user provided network.
But sometimes it is also helpful to incorporate external libraries developed by various vendors into NNVM.
Luckily, TVM has a mechanism to transparently call into these libraries.
For NNVM users, all we need to do is just to set a target string appropriately.

Before we can use external libraries from NNVM, your TVM needs to be built with libraries you want to use.
For example, to use cuDNN, USE_CUDNN option in tvm/make/config.mk needs to be enabled, and cuDNN include and library directories need to be specified.

To begin with, we import NNVM and TVM.
"""
import tvm
import numpy as np
from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing import utils

######################################################################
# Create a simple network
# -----------------------
# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.

out_channels = 16
data = sym.Variable(name="data")
simple_net = sym.conv2d(data=data, kernel_size=(3,3), channels=out_channels, padding = (1, 1), use_bias=True)
simple_net = sym.batch_norm(data=simple_net)
simple_net = sym.relu(data=simple_net)

batch_size = 1
data_shape = (batch_size, 3, 224, 224)
net, params = utils.create_workload(simple_net, batch_size, data_shape[1:])

######################################################################
# Build and run with cuda backend
# -------------------------------
# We build and run this network with cuda backend, as usual.
# By setting the logging level to DEBUG, the result of NNVM graph compilation will be dumped as pseudo code.
import logging
logging.basicConfig(level=logging.DEBUG) # to dump TVM IR after fusion

target = "cuda"
graph, lib, params = nnvm.compiler.build(
    net, target, shape={"data": data_shape}, params=params)

ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.create(graph, lib, ctx)
module.set_input(**params)
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cuda = out.asnumpy()

######################################################################
# The generated pseudo code should look something like below.
# Note how bias add, batch normalization, and ReLU activation are fused into the convolution kernel.
# TVM generates a single, fused kernel from this representation.
#
# .. code-block:: text
#
#       produce compute {
#         // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 112
#         // attr [input1.shared] storage_scope = "shared"
#         allocate input1.shared[float32 * 16 * 3 * 3 * 3]
#         // attr [compute] storage_scope = "local"
#         allocate compute[float32 * 16 * 1 * 1 * 1 * 1]
#         // attr [pad_temp.global.global.shared] storage_scope = "shared"
#         allocate pad_temp.global.global.shared[float32 * 1 * 1 * 4 * 57 * 4]
#         // attr [iter_var(threadIdx.x, Range(min=0, extent=448), threadIdx.x)] thread_extent = 448
#         produce compute {
#           produce input1.shared {
#             for (ax0, 0, 16) {
#               if (likely((threadIdx.x < 27))) {
#                 input1.shared[(threadIdx.x + (ax0*27))] = input1[((((((blockIdx.x/112)*48) + (threadIdx.x/9))*9) + (threadIdx.x % 9)) + (ax0*27))]
#               }
#             }
#           }
#           compute[0] = 0.000000f
#           compute[1] = 0.000000f
#           compute[2] = 0.000000f
#           compute[3] = 0.000000f
#           compute[4] = 0.000000f
#           compute[5] = 0.000000f
#           compute[6] = 0.000000f
#           compute[7] = 0.000000f
#           compute[8] = 0.000000f
#           compute[9] = 0.000000f
#           compute[10] = 0.000000f
#           compute[11] = 0.000000f
#           compute[12] = 0.000000f
#           compute[13] = 0.000000f
#           compute[14] = 0.000000f
#           compute[15] = 0.000000f
#           for (rc, 0, 3) {
#             produce pad_temp.global.global.shared {
#               if (likely((threadIdx.x < 228))) {
#                 if (likely(((blockIdx.x*2) < (226 - (threadIdx.x/57))))) {
#                   pad_temp.global.global.shared[ramp((threadIdx.x*4), 1, 4)] = pad_temp[ramp(((((((blockIdx.x*2) + (threadIdx.x/57))*57) + (threadIdx.x % 57)) + (rc*12882))*4), 1, 4)]
#                 }
#               }
#             }
#             for (ry, 0, 3) {
#               for (rx, 0, 3) {
#                 compute[0] = (compute[0] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[((((rc*3) + ry)*3) + rx)]))
#                 compute[1] = (compute[1] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 27)]))
#                 compute[2] = (compute[2] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 54)]))
#                 compute[3] = (compute[3] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 81)]))
#                 compute[4] = (compute[4] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 108)]))
#                 compute[5] = (compute[5] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 135)]))
#                 compute[6] = (compute[6] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 162)]))
#                 compute[7] = (compute[7] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 189)]))
#                 compute[8] = (compute[8] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 216)]))
#                 compute[9] = (compute[9] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 243)]))
#                 compute[10] = (compute[10] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 270)]))
#                 compute[11] = (compute[11] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 297)]))
#                 compute[12] = (compute[12] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 324)]))
#                 compute[13] = (compute[13] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 351)]))
#                 compute[14] = (compute[14] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 378)]))
#                 compute[15] = (compute[15] + (pad_temp.global.global.shared[(((((threadIdx.x/224)*228) + (threadIdx.x % 224)) + (ry*228)) + rx)]*input1.shared[(((((rc*3) + ry)*3) + rx) + 405)]))
#               }
#             }
#           }
#         }
#         compute[(((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224))] = max((((compute[0] + input2[((blockIdx.x/112)*16)])*input3[((blockIdx.x/112)*16)]) + input4[((blockIdx.x/112)*16)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 50176)] = max((((compute[1] + input2[(((blockIdx.x/112)*16) + 1)])*input3[(((blockIdx.x/112)*16) + 1)]) + input4[(((blockIdx.x/112)*16) + 1)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 100352)] = max((((compute[2] + input2[(((blockIdx.x/112)*16) + 2)])*input3[(((blockIdx.x/112)*16) + 2)]) + input4[(((blockIdx.x/112)*16) + 2)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 150528)] = max((((compute[3] + input2[(((blockIdx.x/112)*16) + 3)])*input3[(((blockIdx.x/112)*16) + 3)]) + input4[(((blockIdx.x/112)*16) + 3)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 200704)] = max((((compute[4] + input2[(((blockIdx.x/112)*16) + 4)])*input3[(((blockIdx.x/112)*16) + 4)]) + input4[(((blockIdx.x/112)*16) + 4)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 250880)] = max((((compute[5] + input2[(((blockIdx.x/112)*16) + 5)])*input3[(((blockIdx.x/112)*16) + 5)]) + input4[(((blockIdx.x/112)*16) + 5)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 301056)] = max((((compute[6] + input2[(((blockIdx.x/112)*16) + 6)])*input3[(((blockIdx.x/112)*16) + 6)]) + input4[(((blockIdx.x/112)*16) + 6)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 351232)] = max((((compute[7] + input2[(((blockIdx.x/112)*16) + 7)])*input3[(((blockIdx.x/112)*16) + 7)]) + input4[(((blockIdx.x/112)*16) + 7)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 401408)] = max((((compute[8] + input2[(((blockIdx.x/112)*16) + 8)])*input3[(((blockIdx.x/112)*16) + 8)]) + input4[(((blockIdx.x/112)*16) + 8)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 451584)] = max((((compute[9] + input2[(((blockIdx.x/112)*16) + 9)])*input3[(((blockIdx.x/112)*16) + 9)]) + input4[(((blockIdx.x/112)*16) + 9)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 501760)] = max((((compute[10] + input2[(((blockIdx.x/112)*16) + 10)])*input3[(((blockIdx.x/112)*16) + 10)]) + input4[(((blockIdx.x/112)*16) + 10)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 551936)] = max((((compute[11] + input2[(((blockIdx.x/112)*16) + 11)])*input3[(((blockIdx.x/112)*16) + 11)]) + input4[(((blockIdx.x/112)*16) + 11)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 602112)] = max((((compute[12] + input2[(((blockIdx.x/112)*16) + 12)])*input3[(((blockIdx.x/112)*16) + 12)]) + input4[(((blockIdx.x/112)*16) + 12)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 652288)] = max((((compute[13] + input2[(((blockIdx.x/112)*16) + 13)])*input3[(((blockIdx.x/112)*16) + 13)]) + input4[(((blockIdx.x/112)*16) + 13)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 702464)] = max((((compute[14] + input2[(((blockIdx.x/112)*16) + 14)])*input3[(((blockIdx.x/112)*16) + 14)]) + input4[(((blockIdx.x/112)*16) + 14)]), 0.000000f)
#         compute[((((((blockIdx.x + ((blockIdx.x/112)*1792))*2) + (threadIdx.x/224))*224) + (threadIdx.x % 224)) + 752640)] = max((((compute[15] + input2[(((blockIdx.x/112)*16) + 15)])*input3[(((blockIdx.x/112)*16) + 15)]) + input4[(((blockIdx.x/112)*16) + 15)]), 0.000000f)
#       }
#

######################################################################
# Use cuDNN for a convolutional layer
# -----------------------------------
# We can use cuDNN to replace convolution kernels with cuDNN ones.
# To do that, all we need to do is to append the option " -libs=cudnn" to the target string.
net, params = utils.create_workload(simple_net, batch_size, data_shape[1:])
target = "cuda -libs=cudnn" # use cudnn for convolution
graph, lib, params = nnvm.compiler.build(
    net, target, shape={"data": data_shape}, params=params)

ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.create(graph, lib, ctx)
module.set_input(**params)
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cudnn = out.asnumpy()

######################################################################
# Note that if you use cuDNN, NNVM cannot fuse convolution with layers following it.
# This is because layer fusion happens at the level of TVM internal representation(IR).
# NNVM treats external libraries as black box, so there is no way to fuse them with TVM IR.
#
# The pseudo code below shows that cuDNN convolution + bias add + batch norm + ReLU turned into two stages of computation, one for cuDNN call and the other for the rest of operations.
#
# .. code-block:: text
#
#       allocate y[float32 * 1 * 16 * 224 * 224]
#       produce y {
#          // attr [0] extern_scope = 0
#          tvm_call_packed("tvm.contrib.cudnn.conv2d.forward", 1, 0, 1, 1, 1, 1, 1, 1, 1, tvm_stack_make_array(input0, tvm_stack_make_shape(1, 3, 224, 224), 0, 4, 0.000000f, 0), tvm_stack_make_array(input1, tvm_stack_make_shape(16, 3, 3, 3), 0, 4, 0.000000f, 0), tvm_stack_make_array(y, tvm_stack_make_shape(1, 16, 224, 224), 0, 4, 0.000000f, 0))
#        }
#       produce compute {
#          // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 1568
#          // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 512
#          compute[((((((blockIdx.x*512) + threadIdx.x)/50176) + ((((blockIdx.x*512) + threadIdx.x)/802816)*16))*50176) + ((((((blockIdx.x*512) + threadIdx.x)/224) % 224)*224) + (((blockIdx.x*64) + threadIdx.x) % 224)))] = max((((y[((((((blockIdx.x*512) + threadIdx.x)/50176) + ((((blockIdx.x*512) + threadIdx.x)/802816)*16))*50176) + ((((((blockIdx.x*512) + threadIdx.x)/224) % 224)*224) + (((blockIdx.x*64) + threadIdx.x) % 224)))] + input2[(((blockIdx.x*512) + threadIdx.x)/50176)])*input3[(((blockIdx.x*512) + threadIdx.x)/50176)]) + input4[(((blockIdx.x*512) + threadIdx.x)/50176)]), 0.000000f)
#        }
#

######################################################################
# Verify the result
# -----------------
# We can check that the results of two runs match.

tvm.testing.assert_allclose(out_cuda, out_cudnn, rtol=1e-5)

#####################################################################
# Conclusion
# ----------
# This tutorial covered the usage of cuDNN with NNVM.
# We also have support for cuBLAS. If cuBLAS is enabled, it will be used inside a fully connected layer (nnvm.symbol.dense).
# To use cuBLAS, set a target string as "cuda -libs=cublas".
# You can use both cuDNN and cuBLAS with "cuda -libs=cudnn,cublas".
#
# For ROCm backend, we have support for MIOpen and rocBLAS.
# They can be enabled with target "rocm -libs=miopen,rocblas".
#
# Being able to use external libraries is great, but we need to keep in mind some cautions.
#
# First, the use of external libraries may restrict your usage of TVM and NNVM.
# For example, MIOpen only supports NCHW layout and fp32 data type at the moment, so you cannot use other layouts or data type in TVM.
#
# Second, and more importantly, external libraries restrict the possibility of operator fusion during graph compilation, as shown above.
# TVM and NNVM aim to achieve the best performance on a variety of hardwares, with joint operator level and graph level optimization.
# To achieve this goal, we should continue developing better optimizations for TVM and NNVM, while using external libraries as a nice way to fall back to existing implementation when necessary.
