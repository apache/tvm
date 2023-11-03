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
from tvm.contrib import graph_executor
from tvm.relay.testing import run_infer_type


def quantize_test_driver(in_dtype, quant_args, axis, out_dtype, in_data, verify_output_data):
    shape = in_data.shape
    input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
    output_zero_point = relay.const(quant_args["out_zero_point"])
    output_scale = relay.const(quant_args["out_scale"])
    quantized_output = relay.qnn.quantize(
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
        rt_mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
        rt_mod.set_input(input_data=in_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).numpy()
        np.testing.assert_equal(res, verify_output_data)
        assert res.dtype == out_dtype


def test_float32_to_uint8():
    data = (
        np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64])
        .astype("float32")
        .reshape((2, 5))
    )
    output = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]).astype("uint8").reshape((2, 5))
    quant_args = {"out_zero_point": np.int32(127), "out_scale": np.float32(0.5)}
    quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="uint8",
        in_data=data,
        verify_output_data=output,
    )


def test_float32_to_int8():
    data = (
        np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64])
        .astype("float32")
        .reshape((2, 5))
    )
    output = (
        np.array([-128, -127, -126, -125, -124, 123, 124, 125, 126, 127])
        .astype("int8")
        .reshape((2, 5))
    )
    quant_args = {"out_zero_point": np.int32(-1), "out_scale": np.float32(0.5)}
    quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="int8",
        in_data=data,
        verify_output_data=output,
    )


def test_float32_to_uint16():
    data = (
        np.array([-6553, -6552.8, -6552.6, -6552.4, -6552.2, 6553.2, 6553.4, 6553.6, 6553.8, 6554])
        .astype("float32")
        .reshape((2, 5))
    )
    output = (
        np.array([0, 1, 2, 3, 4, 65531, 65532, 65533, 65534, 65535])
        .astype("uint16")
        .reshape((2, 5))
    )
    quant_args = {"out_zero_point": np.int32(32765), "out_scale": np.float32(0.2)}
    quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="uint16",
        in_data=data,
        verify_output_data=output,
    )


def test_scalar_float32_to_int8():
    data = np.array(-63.5).astype("float32")
    output = np.array(-128).astype("int8")
    quant_args = {"out_zero_point": np.int32(-1), "out_scale": np.float32(0.5)}
    quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="int8",
        in_data=data,
        verify_output_data=output,
    )


def test_channelwise_axis_0():
    data = (
        np.array([-63.5, -63, -62.5, -62, -61.5, 30, 31, 31.5, 31.75, 32])
        .astype("float32")
        .reshape((2, 5))
    )
    output = np.array([0, 1, 2, 3, 4, 243, 247, 249, 250, 251]).astype("uint8").reshape((2, 5))
    quant_args = {
        "out_zero_point": np.array([127, 123]).astype("int32"),
        "out_scale": np.array([0.5, 0.25]).astype("float32"),
    }

    quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=0,
        out_dtype="uint8",
        in_data=data,
        verify_output_data=output,
    )


def test_channelwise_axis_1():
    data = np.transpose(
        np.array([-63.5, -63, -62.5, -62, -61.5, 30, 31, 31.5, 31.75, 32])
        .astype("float32")
        .reshape((2, 5))
    )
    output = np.transpose(
        np.array([0, 1, 2, 3, 4, 243, 247, 249, 250, 251]).astype("uint8").reshape((2, 5))
    )
    quant_args = {
        "out_zero_point": np.array([127, 123]).astype("int32"),
        "out_scale": np.array([0.5, 0.25]).astype("float32"),
    }

    quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="uint8",
        in_data=data,
        verify_output_data=output,
    )


def test_dynamic_quantize():
    x = relay.var("x", shape=(1, 2, 3, 4), dtype="float32")
    scale_var = relay.var("scale", shape=(), dtype="float32")
    zp_var = relay.var("zp", shape=(), dtype="int32")

    q_x = relay.qnn.quantize(x, scale_var * scale_var, zp_var + zp_var)
    tt = run_infer_type(q_x)

    assert tt.checked_type == relay.TensorType((1, 2, 3, 4), "int8")
    func = relay.Function([x, scale_var, zp_var], q_x)
    data = np.random.uniform(size=(1, 2, 3, 4)).astype("float32")
    scale = np.array(1).astype("float32")
    zp = np.array(0).astype("int32")

    mod = tvm.ir.IRModule.from_expr(func)

    for target, dev in tvm.testing.enabled_targets():
        # TODO: (electriclilies) enable AlterOpLayout when it is fixed
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            lib = relay.build(mod, target=target)

    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(**{"x": data, "scale": scale, "zp": zp})
    module.run()


if __name__ == "__main__":
    test_float32_to_uint8()
    test_float32_to_int8()
    test_float32_to_uint16()
    test_scalar_float32_to_int8()
    test_channelwise_axis_0()
    test_channelwise_axis_1()
    test_dynamic_quantize()
