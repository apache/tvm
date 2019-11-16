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
import tvm
from tvm.contrib import cudnn
import numpy as np


def verify_conv2d(data_dtype, conv_dtype, tensor_format=0):
    in_channel = 4
    out_channel = 32
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    batch = 3
    height = 32
    weight = 32

    if not tvm.module.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cudnn.conv2d.output_shape", True):
        print("skip because cudnn is not enabled...")
        return

    xshape = [batch, in_channel, height, weight]
    wshape = cudnn.conv2d_w_shape(in_channel,
                          out_channel,
                          filter_h,
                          filter_w)

    X = tvm.placeholder(xshape, name='X', dtype=data_dtype)
    W = tvm.placeholder(wshape, name='W', dtype=data_dtype)
    Y = cudnn.conv2d_forward(X,
                             W,
                             stride_h,
                             stride_w,
                             pad_h,
                             pad_w,
                             dilation_h,
                             dilation_w,
                             conv_mode=1,
                             tensor_format=tensor_format,
                             conv_dtype=conv_dtype,
                             algo=-1)
    yshape = [x.value for x in Y.shape]
    s = tvm.create_schedule(Y.op)

    def verify():
        ctx = tvm.gpu(0)
        f = tvm.build(s, [X, W, Y], "cuda", target_host="llvm", name="conv2d")
        x = tvm.nd.array(np.random.uniform(-1, 1, xshape).astype(data_dtype),
                         ctx)
        w = tvm.nd.array(np.random.uniform(-1, 1, wshape).astype(data_dtype),
                         ctx)
        y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(data_dtype),
                         ctx)
        f(x, w, y)

    verify()

def test_conv2d():
    verify_conv2d("float32", "float32", tensor_format=0)
    verify_conv2d("float16", "float32", tensor_format=1)
    verify_conv2d("float16", "float16", tensor_format=0)
    verify_conv2d("int8", "int32", tensor_format=1)


if __name__ == "__main__":
    test_conv2d()
