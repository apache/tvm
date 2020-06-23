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

""" Shape configurations for single operator / subgraph evaluation
This file is shared by tune_op_subgraph.py and scripts in scripts/baseline/
"""

matmul_shapes = [
    (1, 128, 128, 128),
    (1, 512, 32, 512),
    (1, 512, 512, 512),
    (1, 1024, 1024, 1024),
]

conv1d_shapes = [
    # derived from conv2d_shapes
    (1, 256, 64, 128, 3, 2, 1),
#    (1, 256, 64, 128, 1, 2, 0),
#    (1, 256, 64, 64, 1, 1, 0),
#    (1, 128, 128, 256, 3, 2, 1),
    (1, 128, 128, 256, 1, 2, 0),
#    (1, 128, 128, 128, 3, 1, 1),
#    (1, 64, 256, 512, 3, 2, 1),
#    (1, 64, 256, 512, 1, 2, 0),
    (1, 64, 256, 256, 5, 1, 2),
    (1, 32, 512, 512, 3, 1, 1),
]

conv2d_shapes = [
    # all conv2d layers in resnet-18
    (1, 224, 224, 3, 64, 7, 2, 3),
#    (1, 56, 56, 64, 128, 3, 2, 1),
#    (1, 56, 56, 64, 128, 1, 2, 0),
#    (1, 56, 56, 64, 64, 3, 1, 1),
    (1, 56, 56, 64, 64, 1, 1, 0),
#    (1, 28, 28, 128, 256, 3, 2, 1),
#    (1, 28, 28, 128, 256, 1, 2, 0),
#    (1, 28, 28, 128, 128, 3, 1, 1),
#    (1, 14, 14, 256, 512, 3, 2, 1),
#    (1, 14, 14, 256, 512, 1, 2, 0),
    (1, 14, 14, 256, 256, 3, 1, 1),
    (1, 7, 7, 512, 512, 3, 1, 1),
]

conv3d_shapes = [
    # Derived from cnov2d_shapes. Use depth=16 for all configurations
    (1, 16, 224, 224, 3, 64, 7, 2, 3),
#    (1, 16, 56, 56, 64, 128, 3, 2, 1),
#    (1, 16, 56, 56, 64, 128, 1, 2, 0),
#    (1, 16, 56, 56, 64, 64, 3, 1, 1),
    (1, 16, 56, 56, 64, 64, 1, 1, 0),
#    (1, 16, 28, 28, 128, 256, 3, 2, 1),
#    (1, 16, 28, 28, 128, 256, 1, 2, 0),
#    (1, 16, 28, 28, 128, 128, 3, 1, 1),
#    (1, 16, 14, 14, 256, 512, 3, 2, 1),
#    (1, 16, 14, 14, 256, 512, 1, 2, 0),
    (1, 16, 14, 14, 256, 256, 3, 1, 1),
    (1, 16, 7, 7, 512, 512, 3, 1, 1),
]

group_conv2d_shapes = [
    # Derived from cnov2d_shapes. Use group=4 for all configurations
    (1, 56, 56, 64, 128, 3, 2, 1 , 1, 4),
#    (1, 56, 56, 64, 128, 1, 2, 0 , 1, 4),
#    (1, 56, 56, 64, 64, 3, 1, 1  , 1, 4),
    (1, 56, 56, 64, 64, 1, 1, 0  , 1, 4),
#    (1, 28, 28, 128, 256, 3, 2, 1, 1, 4),
#    (1, 28, 28, 128, 256, 1, 2, 0, 1, 4),
#    (1, 28, 28, 128, 128, 3, 1, 1, 1, 4),
#    (1, 14, 14, 256, 512, 3, 2, 1, 1, 4),
#    (1, 14, 14, 256, 512, 1, 2, 0, 1, 4),
    (1, 14, 14, 256, 256, 3, 1, 1, 1, 4),
    (1, 7, 7, 512, 512, 3, 1, 1  , 1, 4),
]

dilation_conv2d_shapes = [
    # Derived from cnov2d_shapes. Use dilation=2 for all configurations
    (1, 224, 224, 3, 64, 7, 2, 3 , 2),
#    (1, 56, 56, 64, 128, 3, 2, 1 , 2),
#    (1, 56, 56, 64, 128, 1, 2, 0 , 2),
#    (1, 56, 56, 64, 64, 3, 1, 1  , 2),
    (1, 56, 56, 64, 64, 1, 1, 0  , 2),
#    (1, 28, 28, 128, 256, 3, 2, 1, 2),
#    (1, 28, 28, 128, 256, 1, 2, 0, 2),
#    (1, 28, 28, 128, 128, 3, 1, 1, 2),
#    (1, 14, 14, 256, 512, 3, 2, 1, 2),
#    (1, 14, 14, 256, 512, 1, 2, 0, 2),
    (1, 14, 14, 256, 256, 3, 1, 1, 2),
    (1, 7, 7, 512, 512, 3, 1, 1  , 2),
]

depthwise_conv2d_shapes = [
    # all depthwise conv2d layers in mobilenet
    (1, 112, 112, 32,  3, 1, 1),
    (1, 112, 112, 64,  3, 2, 1),
#    (1,  56,  56, 128, 3, 1, 1),
#    (1,  56,  56, 128, 3, 2, 1),
#    (1,  28,  28, 256, 3, 1, 1),
#    (1,  28,  28, 256, 3, 2, 1),
#    (1,  14,  14, 512, 3, 1, 1),
    (1,  14,  14, 512, 3, 2, 1),
    (1,   7,   7, 1024, 3, 1, 1),
]

conv2d_transpose_shapes = [
    # all conv2d tranpose layers in DCGAN
    (1, 4, 4, 512, 256, 4, 2, 1),
    (1, 8, 8, 256, 128, 4, 2, 1),
    (1, 16, 16, 128, 64, 4, 2, 1),
    (1, 32, 32, 64, 3, 4, 2, 1),
]

conv2d_capsule_shapes = [
    # all conv2d capsule layers in matrix capsules withemrouting (ICLR 2018)
    (1, 16, 16, 32, 32, 3, 2, 1),
    (1,  8,  8, 32, 32, 3, 1, 1),
    (1, 16, 16,  8, 16, 3, 2, 1),
    (1,  8,  8, 16, 16, 3, 1, 1),
]

conv2d_winograd_nhwc_shapes = [
    (1, 56, 56, 64, 64, 3, 1, 1),
    (1, 28, 28, 128, 128, 3, 1, 1),
    (1, 14, 14, 256, 256, 3, 1, 1),
    (1, 7, 7, 512, 512, 3, 1, 1),
]

conv2d_winograd_nchw_shapes = [
    (1, 64, 56, 56, 64, 3, 1, 1),
    (1, 128, 28, 28, 128, 3, 1, 1),
    (1, 256, 14, 14, 256, 3, 1, 1),
    (1, 512, 7, 7, 512, 3, 1, 1),
]

matmul_tensor_core_shapes = [
    (16, 512, 512, 'float16', 'float32', True),
    (32, 512, 512, 'float16', 'float32', True),
    (512, 512, 512, 'float16', 'float32', True),
]

norm_shapes = [
    (1, 256, 256),
    (1, 512, 512),
    (1, 1024, 1024),
    (1, 4096, 1024),
]

single_op_shape_dict = {
    'C1D': conv1d_shapes,
    'C2D': conv2d_shapes,
    'C3D': conv3d_shapes,
    'GMM': matmul_shapes,
    'GRP': group_conv2d_shapes,
    'DIL': dilation_conv2d_shapes,
    'DEP': depthwise_conv2d_shapes,
    'T2D': conv2d_transpose_shapes,
    'CAP': conv2d_capsule_shapes,
    'NRM': norm_shapes,

#    The following workloads are not in our sinle op evaluation plan.
#    They should be moved to `common.py` and be used by `tune_wkl.py`.
#    'C2D_NCHW': conv2d_nchw_shapes,
#    'C2DWG_NHWC': conv2d_winograd_nhwc_shapes,
#    'C2DWG_NCHW': conv2d_winograd_nchw_shapes,
#    'GMM_TC': matmul_tensor_core_shapes,
}

conv2d_bn_relu_shapes = [
    (1, 224, 224, 3, 64, 7, 2, 3),
    (1, 56, 56, 64, 128, 3, 2, 1),
    (1, 28, 28, 128, 256, 1, 2, 0),
    (1, 7, 7, 512, 512, 3, 1, 1, 1),
    (16, 224, 224, 3, 64, 7, 2, 3),
    (16, 56, 56, 64, 128, 3, 2, 1),
    (16, 28, 28, 128, 256, 1, 2, 0),
    (16, 7, 7, 512, 512, 3, 1, 1, 1),
]

transpose_batch_matmul_shapes = [
    (1,   128, 12, 64),
    (1,   128, 16, 64),
    (1,   64,  12, 128),
    (1,   128, 12, 128),
    (16,  128, 12, 64),
    (16,  128, 16, 64),
    (16,  64,  12, 128),
    (16,  128, 12, 128),
]

subgraph_shape_dict = {
    "conv2d_bn_relu": conv2d_bn_relu_shapes,
    "transpose_batch_matmul": transpose_batch_matmul_shapes,
}

resnet_shapes = [
    (1, ),
    (16, ),
]

mobilenet_v2_shapes = [
    (1, ),
    (16, ),
]

dcgan_shapes = [
    (1, ),
    (16, ),
]

dqn_shapes = [
    (1, ),
    (16, ),
]

bert_shapes = [
    (1, ),
    (16, ),
]

resnet18_3d_shapes = [
    (1, ),
    (16, ),
]

network_shape_dict = {
    'resnet_50': resnet_shapes,
    'mobilenet_v2': mobilenet_v2_shapes,
    'dcgan': dcgan_shapes,
    'dqn': dqn_shapes,
    'bert': bert_shapes,
    'resnet_18_3d': resnet18_3d_shapes,
}

