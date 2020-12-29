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
import tvm.testing
from tvm import te
from tvm.contrib import miopen
import numpy as np


@tvm.testing.requires_rocm
def test_conv2d():
    in_channel = 3
    out_channel = 64
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [1, in_channel, 128, 128]
    if not tvm.get_global_func("tvm.contrib.miopen.conv2d.setup", True):
        print("skip because miopen is not enabled...")
        return
    wshape = (out_channel, in_channel, filter_h, filter_w)

    X = te.placeholder(xshape, name="X")
    W = te.placeholder(wshape, name="W")
    Y = miopen.conv2d_forward(
        X, W, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, conv_mode=0, data_type=1
    )

    yshape = [x.value for x in Y.shape]
    from tvm import topi

    s = te.create_schedule(Y.op)

    def verify():
        ctx = tvm.rocm(0)
        f = tvm.build(s, [X, W, Y], "rocm", target_host="llvm", name="conv2d")
        x = tvm.nd.array(np.random.uniform(-1, 1, xshape).astype(np.float32), ctx)
        w = tvm.nd.array(np.random.uniform(-1, 1, wshape).astype(np.float32), ctx)
        y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32), ctx)
        f(x, w, y)

        Y_ref = topi.nn.conv2d_nchw(
            X, W, (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w)
        )
        s_ref = te.create_schedule(Y_ref.op)
        f_ref = tvm.build(s_ref, [X, W, Y_ref], "rocm", target_host="llvm")
        y_ref = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32), ctx)
        f_ref(x, w, y_ref)
        print("Max abs diff:", np.max(np.abs(y.asnumpy() - y_ref.asnumpy())))
        tvm.testing.assert_allclose(y.asnumpy(), y_ref.asnumpy(), atol=1e-3)

    verify()


if __name__ == "__main__":
    test_conv2d()
