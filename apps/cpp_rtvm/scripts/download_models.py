#!/usr/bin/env python3

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

import os

import numpy as np

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.support import ndk

TARGET = "opencl"
HOST = {"kind": "llvm", "mtriple": "aarch64-linux-gnu"}


def get_network():
    @I.ir_module
    class Resnet:
        @R.function
        def main(
            data: R.Tensor((1, 3, 224, 224), dtype="float32"),
            resnetv22_batchnorm0_gamma: R.Tensor((3,), dtype="float32"),
            resnetv22_batchnorm0_beta: R.Tensor((3,), dtype="float32"),
            resnetv22_batchnorm0_running_mea: R.Tensor((3,), dtype="float32"),
            resnetv22_batchnorm0_running_var: R.Tensor((3,), dtype="float32"),
            resnetv22_conv0_weight: R.Tensor((64, 3, 7, 7), dtype="float32"),
            resnetv22_batchnorm1_gamma: R.Tensor((64,), dtype="float32"),
            resnetv22_batchnorm1_beta: R.Tensor((64,), dtype="float32"),
            resnetv22_batchnorm1_running_mea: R.Tensor((64,), dtype="float32"),
            resnetv22_batchnorm1_running_var: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm0_gamma: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm0_beta: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm0_running_mea: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm0_running_var: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_conv0_weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
            resnetv22_stage1_batchnorm1_gamma: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm1_beta: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm1_running_mea: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm1_running_var: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_conv1_weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
            resnetv22_stage1_batchnorm2_gamma: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm2_beta: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm2_running_mea: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm2_running_var: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_conv2_weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
            resnetv22_stage1_batchnorm3_gamma: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm3_beta: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm3_running_mea: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_batchnorm3_running_var: R.Tensor((64,), dtype="float32"),
            resnetv22_stage1_conv3_weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
            resnetv22_stage2_batchnorm0_gamma: R.Tensor((64,), dtype="float32"),
            resnetv22_stage2_batchnorm0_beta: R.Tensor((64,), dtype="float32"),
            resnetv22_stage2_batchnorm0_running_mea: R.Tensor((64,), dtype="float32"),
            resnetv22_stage2_batchnorm0_running_var: R.Tensor((64,), dtype="float32"),
            resnetv22_stage2_conv0_weight: R.Tensor((128, 64, 3, 3), dtype="float32"),
            resnetv22_stage2_batchnorm1_gamma: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm1_beta: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm1_running_mea: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm1_running_var: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_conv1_weight: R.Tensor((128, 128, 3, 3), dtype="float32"),
            resnetv22_stage2_conv2_weight: R.Tensor((128, 64, 1, 1), dtype="float32"),
            resnetv22_stage2_batchnorm2_gamma: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm2_beta: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm2_running_mea: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm2_running_var: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_conv3_weight: R.Tensor((128, 128, 3, 3), dtype="float32"),
            resnetv22_stage2_batchnorm3_gamma: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm3_beta: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm3_running_mea: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_batchnorm3_running_var: R.Tensor((128,), dtype="float32"),
            resnetv22_stage2_conv4_weight: R.Tensor((128, 128, 3, 3), dtype="float32"),
            resnetv22_stage3_batchnorm0_gamma: R.Tensor((128,), dtype="float32"),
            resnetv22_stage3_batchnorm0_beta: R.Tensor((128,), dtype="float32"),
            resnetv22_stage3_batchnorm0_running_mea: R.Tensor((128,), dtype="float32"),
            resnetv22_stage3_batchnorm0_running_var: R.Tensor((128,), dtype="float32"),
            resnetv22_stage3_conv0_weight: R.Tensor((256, 128, 3, 3), dtype="float32"),
            resnetv22_stage3_batchnorm1_gamma: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm1_beta: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm1_running_mea: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm1_running_var: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_conv1_weight: R.Tensor((256, 256, 3, 3), dtype="float32"),
            resnetv22_stage3_conv2_weight: R.Tensor((256, 128, 1, 1), dtype="float32"),
            resnetv22_stage3_batchnorm2_gamma: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm2_beta: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm2_running_mea: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm2_running_var: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_conv3_weight: R.Tensor((256, 256, 3, 3), dtype="float32"),
            resnetv22_stage3_batchnorm3_gamma: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm3_beta: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm3_running_mea: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_batchnorm3_running_var: R.Tensor((256,), dtype="float32"),
            resnetv22_stage3_conv4_weight: R.Tensor((256, 256, 3, 3), dtype="float32"),
            resnetv22_stage4_batchnorm0_gamma: R.Tensor((256,), dtype="float32"),
            resnetv22_stage4_batchnorm0_beta: R.Tensor((256,), dtype="float32"),
            resnetv22_stage4_batchnorm0_running_mea: R.Tensor((256,), dtype="float32"),
            resnetv22_stage4_batchnorm0_running_var: R.Tensor((256,), dtype="float32"),
            resnetv22_stage4_conv0_weight: R.Tensor((512, 256, 3, 3), dtype="float32"),
            resnetv22_stage4_batchnorm1_gamma: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm1_beta: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm1_running_mea: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm1_running_var: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_conv1_weight: R.Tensor((512, 512, 3, 3), dtype="float32"),
            resnetv22_stage4_conv2_weight: R.Tensor((512, 256, 1, 1), dtype="float32"),
            resnetv22_stage4_batchnorm2_gamma: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm2_beta: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm2_running_mea: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm2_running_var: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_conv3_weight: R.Tensor((512, 512, 3, 3), dtype="float32"),
            resnetv22_stage4_batchnorm3_gamma: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm3_beta: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm3_running_mea: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_batchnorm3_running_var: R.Tensor((512,), dtype="float32"),
            resnetv22_stage4_conv4_weight: R.Tensor((512, 512, 3, 3), dtype="float32"),
            resnetv22_batchnorm2_gamma: R.Tensor((512,), dtype="float32"),
            resnetv22_batchnorm2_beta: R.Tensor((512,), dtype="float32"),
            resnetv22_batchnorm2_running_mea: R.Tensor((512,), dtype="float32"),
            resnetv22_batchnorm2_running_var: R.Tensor((512,), dtype="float32"),
            reshape_attr_tensor164: R.Tensor((2,), dtype="int64"),
            resnetv22_dense0_weight: R.Tensor((1000, 512), dtype="float32"),
            resnetv22_dense0_bias: R.Tensor((1000,), dtype="float32"),
        ) -> R.Tensor((1, 1000), dtype="float32"):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 224, 224), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    data,
                    resnetv22_batchnorm0_gamma,
                    resnetv22_batchnorm0_beta,
                    resnetv22_batchnorm0_running_mea,
                    resnetv22_batchnorm0_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv1: R.Tensor((1, 3, 224, 224), dtype="float32") = lv[0]
                lv4: R.Tensor((1, 64, 112, 112), dtype="float32") = R.nn.conv2d(
                    lv1,
                    resnetv22_conv0_weight,
                    strides=[2, 2],
                    padding=[3, 3, 3, 3],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv5: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv4,
                    resnetv22_batchnorm1_gamma,
                    resnetv22_batchnorm1_beta,
                    resnetv22_batchnorm1_running_mea,
                    resnetv22_batchnorm1_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv6: R.Tensor((1, 64, 112, 112), dtype="float32") = lv5[0]
                lv9: R.Tensor((1, 64, 112, 112), dtype="float32") = R.nn.relu(lv6)
                lv10: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.max_pool2d(
                    lv9,
                    pool_size=[3, 3],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[1, 1, 1, 1],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv11: R.Tuple(
                    R.Tensor((1, 64, 56, 56), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv10,
                    resnetv22_stage1_batchnorm0_gamma,
                    resnetv22_stage1_batchnorm0_beta,
                    resnetv22_stage1_batchnorm0_running_mea,
                    resnetv22_stage1_batchnorm0_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv12: R.Tensor((1, 64, 56, 56), dtype="float32") = lv11[0]
                lv15: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv12)
                lv16: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    lv15,
                    resnetv22_stage1_conv0_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv17: R.Tuple(
                    R.Tensor((1, 64, 56, 56), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv16,
                    resnetv22_stage1_batchnorm1_gamma,
                    resnetv22_stage1_batchnorm1_beta,
                    resnetv22_stage1_batchnorm1_running_mea,
                    resnetv22_stage1_batchnorm1_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv18: R.Tensor((1, 64, 56, 56), dtype="float32") = lv17[0]
                lv21: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv18)
                lv22: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    lv21,
                    resnetv22_stage1_conv1_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv23: R.Tensor((1, 64, 56, 56), dtype="float32") = R.add(lv22, lv10)
                lv24: R.Tuple(
                    R.Tensor((1, 64, 56, 56), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv23,
                    resnetv22_stage1_batchnorm2_gamma,
                    resnetv22_stage1_batchnorm2_beta,
                    resnetv22_stage1_batchnorm2_running_mea,
                    resnetv22_stage1_batchnorm2_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv25: R.Tensor((1, 64, 56, 56), dtype="float32") = lv24[0]
                lv28: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv25)
                lv29: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    lv28,
                    resnetv22_stage1_conv2_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv30: R.Tuple(
                    R.Tensor((1, 64, 56, 56), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv29,
                    resnetv22_stage1_batchnorm3_gamma,
                    resnetv22_stage1_batchnorm3_beta,
                    resnetv22_stage1_batchnorm3_running_mea,
                    resnetv22_stage1_batchnorm3_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv31: R.Tensor((1, 64, 56, 56), dtype="float32") = lv30[0]
                lv34: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv31)
                lv35: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    lv34,
                    resnetv22_stage1_conv3_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv36: R.Tensor((1, 64, 56, 56), dtype="float32") = R.add(lv35, lv23)
                lv37: R.Tuple(
                    R.Tensor((1, 64, 56, 56), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv36,
                    resnetv22_stage2_batchnorm0_gamma,
                    resnetv22_stage2_batchnorm0_beta,
                    resnetv22_stage2_batchnorm0_running_mea,
                    resnetv22_stage2_batchnorm0_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv38: R.Tensor((1, 64, 56, 56), dtype="float32") = lv37[0]
                lv41: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv38)
                lv42: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(
                    lv41,
                    resnetv22_stage2_conv0_weight,
                    strides=[2, 2],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv43: R.Tuple(
                    R.Tensor((1, 128, 28, 28), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv42,
                    resnetv22_stage2_batchnorm1_gamma,
                    resnetv22_stage2_batchnorm1_beta,
                    resnetv22_stage2_batchnorm1_running_mea,
                    resnetv22_stage2_batchnorm1_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv44: R.Tensor((1, 128, 28, 28), dtype="float32") = lv43[0]
                lv47: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv44)
                lv48: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(
                    lv47,
                    resnetv22_stage2_conv1_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv49: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(
                    lv41,
                    resnetv22_stage2_conv2_weight,
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv50: R.Tensor((1, 128, 28, 28), dtype="float32") = R.add(lv48, lv49)
                lv51: R.Tuple(
                    R.Tensor((1, 128, 28, 28), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv50,
                    resnetv22_stage2_batchnorm2_gamma,
                    resnetv22_stage2_batchnorm2_beta,
                    resnetv22_stage2_batchnorm2_running_mea,
                    resnetv22_stage2_batchnorm2_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv52: R.Tensor((1, 128, 28, 28), dtype="float32") = lv51[0]
                lv55: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv52)
                lv56: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(
                    lv55,
                    resnetv22_stage2_conv3_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv57: R.Tuple(
                    R.Tensor((1, 128, 28, 28), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv56,
                    resnetv22_stage2_batchnorm3_gamma,
                    resnetv22_stage2_batchnorm3_beta,
                    resnetv22_stage2_batchnorm3_running_mea,
                    resnetv22_stage2_batchnorm3_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv58: R.Tensor((1, 128, 28, 28), dtype="float32") = lv57[0]
                lv61: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv58)
                lv62: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.conv2d(
                    lv61,
                    resnetv22_stage2_conv4_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv63: R.Tensor((1, 128, 28, 28), dtype="float32") = R.add(lv62, lv50)
                lv64: R.Tuple(
                    R.Tensor((1, 128, 28, 28), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                    R.Tensor((128,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv63,
                    resnetv22_stage3_batchnorm0_gamma,
                    resnetv22_stage3_batchnorm0_beta,
                    resnetv22_stage3_batchnorm0_running_mea,
                    resnetv22_stage3_batchnorm0_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv65: R.Tensor((1, 128, 28, 28), dtype="float32") = lv64[0]
                lv68: R.Tensor((1, 128, 28, 28), dtype="float32") = R.nn.relu(lv65)
                lv69: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(
                    lv68,
                    resnetv22_stage3_conv0_weight,
                    strides=[2, 2],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv70: R.Tuple(
                    R.Tensor((1, 256, 14, 14), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv69,
                    resnetv22_stage3_batchnorm1_gamma,
                    resnetv22_stage3_batchnorm1_beta,
                    resnetv22_stage3_batchnorm1_running_mea,
                    resnetv22_stage3_batchnorm1_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv71: R.Tensor((1, 256, 14, 14), dtype="float32") = lv70[0]
                lv74: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv71)
                lv75: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(
                    lv74,
                    resnetv22_stage3_conv1_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv76: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(
                    lv68,
                    resnetv22_stage3_conv2_weight,
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv77: R.Tensor((1, 256, 14, 14), dtype="float32") = R.add(lv75, lv76)
                lv78: R.Tuple(
                    R.Tensor((1, 256, 14, 14), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv77,
                    resnetv22_stage3_batchnorm2_gamma,
                    resnetv22_stage3_batchnorm2_beta,
                    resnetv22_stage3_batchnorm2_running_mea,
                    resnetv22_stage3_batchnorm2_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv79: R.Tensor((1, 256, 14, 14), dtype="float32") = lv78[0]
                lv82: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv79)
                lv83: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(
                    lv82,
                    resnetv22_stage3_conv3_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv84: R.Tuple(
                    R.Tensor((1, 256, 14, 14), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv83,
                    resnetv22_stage3_batchnorm3_gamma,
                    resnetv22_stage3_batchnorm3_beta,
                    resnetv22_stage3_batchnorm3_running_mea,
                    resnetv22_stage3_batchnorm3_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv85: R.Tensor((1, 256, 14, 14), dtype="float32") = lv84[0]
                lv88: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv85)
                lv89: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.conv2d(
                    lv88,
                    resnetv22_stage3_conv4_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv90: R.Tensor((1, 256, 14, 14), dtype="float32") = R.add(lv89, lv77)
                lv91: R.Tuple(
                    R.Tensor((1, 256, 14, 14), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                    R.Tensor((256,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv90,
                    resnetv22_stage4_batchnorm0_gamma,
                    resnetv22_stage4_batchnorm0_beta,
                    resnetv22_stage4_batchnorm0_running_mea,
                    resnetv22_stage4_batchnorm0_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv92: R.Tensor((1, 256, 14, 14), dtype="float32") = lv91[0]
                lv95: R.Tensor((1, 256, 14, 14), dtype="float32") = R.nn.relu(lv92)
                lv96: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(
                    lv95,
                    resnetv22_stage4_conv0_weight,
                    strides=[2, 2],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv97: R.Tuple(
                    R.Tensor((1, 512, 7, 7), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv96,
                    resnetv22_stage4_batchnorm1_gamma,
                    resnetv22_stage4_batchnorm1_beta,
                    resnetv22_stage4_batchnorm1_running_mea,
                    resnetv22_stage4_batchnorm1_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv98: R.Tensor((1, 512, 7, 7), dtype="float32") = lv97[0]
                lv101: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv98)
                lv102: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(
                    lv101,
                    resnetv22_stage4_conv1_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv103: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(
                    lv95,
                    resnetv22_stage4_conv2_weight,
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv104: R.Tensor((1, 512, 7, 7), dtype="float32") = R.add(lv102, lv103)
                lv105: R.Tuple(
                    R.Tensor((1, 512, 7, 7), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv104,
                    resnetv22_stage4_batchnorm2_gamma,
                    resnetv22_stage4_batchnorm2_beta,
                    resnetv22_stage4_batchnorm2_running_mea,
                    resnetv22_stage4_batchnorm2_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv106: R.Tensor((1, 512, 7, 7), dtype="float32") = lv105[0]
                lv109: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv106)
                lv110: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(
                    lv109,
                    resnetv22_stage4_conv3_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv111: R.Tuple(
                    R.Tensor((1, 512, 7, 7), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv110,
                    resnetv22_stage4_batchnorm3_gamma,
                    resnetv22_stage4_batchnorm3_beta,
                    resnetv22_stage4_batchnorm3_running_mea,
                    resnetv22_stage4_batchnorm3_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv112: R.Tensor((1, 512, 7, 7), dtype="float32") = lv111[0]
                lv115: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv112)
                lv116: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.conv2d(
                    lv115,
                    resnetv22_stage4_conv4_weight,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                )
                lv117: R.Tensor((1, 512, 7, 7), dtype="float32") = R.add(lv116, lv104)
                lv118: R.Tuple(
                    R.Tensor((1, 512, 7, 7), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                    R.Tensor((512,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv117,
                    resnetv22_batchnorm2_gamma,
                    resnetv22_batchnorm2_beta,
                    resnetv22_batchnorm2_running_mea,
                    resnetv22_batchnorm2_running_var,
                    axis=1,
                    epsilon=9.9999997473787516e-06,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                lv119: R.Tensor((1, 512, 7, 7), dtype="float32") = lv118[0]
                lv122: R.Tensor((1, 512, 7, 7), dtype="float32") = R.nn.relu(lv119)
                lv123: R.Tensor((1, 512, 1, 1), dtype="float32") = R.mean(
                    lv122, axis=[2, 3], keepdims=True
                )
                lv124: R.Tensor((1, 512), dtype="float32") = R.reshape(lv123, R.shape([1, 512]))
                lv125: R.Tensor((512, 1000), dtype="float32") = R.permute_dims(
                    resnetv22_dense0_weight, axes=[1, 0]
                )
                lv126: R.Tensor((1, 1000), dtype="float32") = R.matmul(lv124, lv125)
                gv: R.Tensor((1, 1000), dtype="float32") = R.add(lv126, resnetv22_dense0_bias)
                R.output(gv)
            return gv

    return Resnet


def generate_model_artifacts(mod):
    tgt = tvm.target.Target(TARGET, host=HOST)

    # tvm.compile applies the target specific default Relax and TIR pipelines.
    ex = tvm.compile(mod, tgt)

    ex.export_library(
        tgt.kind.name + "_vm_mod.so",
        fcompile=ndk.create_shared,
        options=["-shared", "-fPIC", "-lm"],
    )

    input_dir = "inputs/"
    os.makedirs(input_dir, exist_ok=True)
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.ty.shape.values)
        if str(arg.ty.dtype).startswith("uint"):
            np.save(
                input_dir + arg.name + ".npy",
                np.random.randint(0, 8, size=shape).astype(str(arg.ty.dtype)),
            )
        elif str(arg.ty.dtype).startswith("int"):
            np.save(
                input_dir + arg.name + ".npy",
                np.random.randint(-8, 8, size=shape).astype(str(arg.ty.dtype)),
            )
        else:
            np.save(
                input_dir + arg.name + ".npy",
                np.random.uniform(0, 1, size=shape).astype(str(arg.ty.dtype)),
            )


if __name__ == "__main__":
    mod = get_network()
    generate_model_artifacts(mod)
