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
from tvm import te
import numpy as np
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing import run_infer_type


def quantize_test_driver(in_dtype, quant_args, axis, out_dtype, in_data):
    shape = in_data.shape
    input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
    output_zero_point = relay.const(quant_args["out_zero_point"])
    output_scale = relay.const(quant_args["out_scale"])
    quantized_output = relay.qnn.op.quantize(
        input_data,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        axis=axis,
        out_dtype=out_dtype,
    )
    mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
    mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", params=None)
    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.set_input(input_data=in_data)
    rt_mod.set_input(**params)
    rt_mod.run()
    res = rt_mod.get_output(0).asnumpy()
    return res


def test_float32_to_uint8():
    data = (
        np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64])
        .astype("float32")
        .reshape((2, 5))
    )
    scale_np = np.float32(0.5)
    zp_np = np.int32(127)
    quant_args = {"out_zero_point": zp_np, "out_scale": scale_np}
    q_out = quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="uint8",
        in_data=data,
    )
    input_data = relay.var("input_data", shape=data.shape, dtype='float32')
    scale = relay.var("scale", shape=[relay.Any()], dtype="float32")
    zp = relay.var("zp", shape=[relay.Any()], dtype="int32")
    dtype = relay.var("dtype", shape=[1], dtype='int32')
    sim_q_out = relay.qnn.op.simulated_quantize(
        input_data,
        scale,
        zp,
        axis=-1,
        out_dtype=dtype,
    )
    mod = tvm.IRModule.from_expr(sim_q_out)
    print(mod)
    print(relay.transform.InferType()(mod))
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", params=None)
    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.set_input(input_data=data, scale=scale_np, zp=zp_np)
    rt_mod.set_input(**params)
    rt_mod.run()
    res = rt_mod.get_output(0).asnumpy()

    print(sim_q_out)


if __name__ == "__main__":
    test_float32_to_uint8()