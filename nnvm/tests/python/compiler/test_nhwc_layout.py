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
import numpy as np
import tvm
from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list

def get_sym(layout, kernel_layout, channels):
    data = sym.Variable(name="data")
    data = sym.conv2d(data=data, kernel_size=(3,3), channels=channels, padding=(1, 1),
                      layout=layout, kernel_layout=kernel_layout, use_bias=True)
    data = sym.max_pool2d(data=data, pool_size=(2, 2), strides=(2, 2), layout=layout)
    data = sym.upsampling(data=data, scale=2, layout=layout)
    softmax_axis = 1
    if layout == "NHWC":
        softmax_axis = 3
    data = sym.softmax(data=data, axis=softmax_axis)
    return data


def build_and_run(sym, params, data, out_shape):
    ctx = tvm.cpu(0)
    graph, lib, params = nnvm.compiler.build(sym, "llvm", shape={"data":data.shape}, params=params)
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(out_shape))
    return out.asnumpy()


def test_nhwc():
    data_shape = (1, 3, 224, 224)
    out_channel = 8
    nchw_sym = get_sym("NCHW", "OIHW", out_channel)
    nhwc_sym = get_sym("NHWC", "HWIO", out_channel)
    conv_weight = np.random.uniform(-1, 1, (out_channel, 3, 3, 3)).astype(np.float32)
    conv_bias = np.random.uniform(-1, 1, (out_channel)).astype(np.float32)
    nchw_params = {
        "conv2d0_weight" : tvm.nd.array(conv_weight, ctx=tvm.cpu(0)),
        "conv2d0_bias" : tvm.nd.array(conv_bias, ctx=tvm.cpu(0))
    }
    nhwc_params = {
        "conv2d1_weight" : tvm.nd.array(conv_weight.transpose(2, 3, 1, 0), ctx=tvm.cpu(0)),
        "conv2d1_bias" : tvm.nd.array(conv_bias, ctx=tvm.cpu(0))
    }

    data = np.random.uniform(-1, 1, data_shape).astype(np.float32)
    oshape = (1, out_channel, 224, 224)
    oshape_nhwc = (1, 224, 224, out_channel)
    nchw_output = build_and_run(nchw_sym, nchw_params, data, oshape)
    nhwc_output = build_and_run(nhwc_sym, nhwc_params, data.transpose(0, 2, 3, 1), oshape_nhwc)
    tvm.testing.assert_allclose(nchw_output, nhwc_output.transpose(0, 3, 1, 2), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_nhwc()
